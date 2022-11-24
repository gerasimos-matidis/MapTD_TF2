import tensorflow as tf
import keras
import cv2
import numpy as np
import argparse
import math
from tensorflow.image import non_max_suppression as nms
from maptd_model import maptd_model
#from predict_v2 import create_tile_set, convert_geometry_to_boxes
import matplotlib.pyplot as plt
from visualize_modified import render_boxes

def reconstruct_box(point, dist, theta, tile_overlap=1024):
    """
    Reconstructs a rotated rectangle from the corresponding point, distance to
    edges and angle

    Parameters
      point: the (i, j) coordinate of the pixel as a numpy array
      dist : distances from the pixel to [top, left, bottom, and right] edges 
              as a numpy array
      theta: the angle of rotation (according to OpenCV), a scalar

    Returns
      box: a rotated rectangle defined by four (i,j) vertices in bottom-left, 
            bottom-right, top-right, and top-left order; a numpy array with
            shape [4,2]  (i.e., 4 corner points with 2 coordinates per point)

    Note
     i corresponds to row number, j corresponds to column number
    """
    
    left_trans = np.asarray([-math.cos(theta), -math.sin(theta)])*dist[0]
    bottom_trans = np.asarray([math.sin(theta), -math.cos(theta)])*dist[1]
    right_trans = np.asarray([math.cos(theta), math.sin(theta)])*dist[2]
    top_trans = np.asarray([-math.sin(theta), math.cos(theta)])*dist[3]
    
    v0_trans = left_trans + bottom_trans
    v1_trans = bottom_trans + right_trans
    v2_trans = right_trans + top_trans
    v3_trans = top_trans + left_trans

    point = point*4 # Compensate for downsampling in target map construction
    v0 = point + v0_trans
    v1 = point + v1_trans
    v2 = point + v2_trans
    v3 = point + v3_trans
    box = np.asarray([v1, v2, v3, v0])
    
    return box


def create_tile_set(image, tile_shape):
    """
    Create a set of tiles and their corresponding relative positions (shifts)
    A tile is a smaller section of the larger image.

    Parameters
       image      : image to tile, as numpy array, shape [H, W, 3]
       tile_shape : tuple of the tiles size as (width, height)

    Returns
       tiles  : list of tile images of shape tile_shape, 
                 where each tile image is a numpy array of shape 
                 (height, width, 3) and tile image is a portion of image
       shifts : list of (y,x) shifts necessary to translate points on each tile
                  to the original image
    """

    def tile_ticks( img_sz, tile_sz ):
        """ Calculate tile origin points and sizes.
            Tiles must overlap by at least the args.tile_overlap
        """
        tile_overlap = 1024
        
        ticks = list()
        sizes = list()
        
        pos = 0

        if tile_sz > img_sz:
            return [0], [img_sz]

        while (True):
            ticks.append( pos )
            sizes.append( tile_sz )
            
            next_pos = pos + tile_sz - tile_overlap

            if (next_pos + tile_sz) >= img_sz:
                trunc_tile_sz = img_sz - next_pos
                next_pos = img_sz - trunc_tile_sz

                ticks.append( next_pos )
                sizes.append( trunc_tile_sz )

                return ticks, sizes
            
            pos = next_pos

    # Main procedure
    tiles = list()
    shifts = list()
    tile_width = tile_shape[0]
    tile_height = tile_shape[1]
    im_width = len(image[0])
    im_height = len(image)

    y_tiles = tile_ticks( im_height, tile_height )
    x_tiles = tile_ticks( im_width, tile_width )

    print(y_tiles)
    print(x_tiles)
    
    # Loop over all tile (position, size) pairs
    for y,h in zip(y_tiles[0],y_tiles[1]):
        for x,w in zip(x_tiles[0],x_tiles[1]):
            tile = image[y:y+h, x:x+w]
            shift = (y,x)
    
            tiles.append( tile )
            shifts.append( shift )

    return tiles, shifts


def convert_geometry_to_boxes(score_map, geo_map, detect_thresh):
    """Convert the predicted geometry map into rotated rectangles

    Parameters:
       score_map     : the predicted score map, a numpy array of size
                         [batch_size, tile_size/4, tile_size/4, 1]
       geo_map       : the predicted geometry map, a numpy array of size
                         [batch_size, tile_size/4, tile_size/4, 5]
       detect_thresh : the minimum score of a bounding box for it to be 
                        counted as prediction, a scalar
    Returns:
       boxes : a Nx9 numpy array with the rectangle coordinates and scores;
                 each 1x9 row contains the four (i,j) pairs of vertices 
                 (in bottom-left, bottom-right, top-right, and top-left order) 
                 followed by the detection score
    """
    score_map = np.squeeze(score_map)
    geo_map = np.squeeze(geo_map)
    
    boxes = list()
    for i in range(len(score_map)):
        for j in range(len(score_map[0])):
            if (score_map[i, j] < detect_thresh):
                continue
            point = np.asarray([i, j])
            dist = geo_map[i, j, 0:4]
            theta = -geo_map[i, j, 4] # Negate to convert from OpenCV's ij to xy
            box = reconstruct_box(point, dist, theta)
            box = np.append(box, score_map[i, j])
            boxes.append(box)

    boxes = np.asarray(boxes)
    boxes = np.reshape(boxes, (len(boxes), 9))
    
    return boxes


def predict_v2(model, image_file, tile_shape, detect_thresh=0.8, pyramid_levels=1):

    """Use a restored model to detect text in the given image

    Parameters
       sess          : TensorFlow Session object
       image_file    : path of the image to run through the model, a string
       pyramid_levels: number of pyramid levels (decimations) before NMS
       input_images  : TensorFlow placeholder for image batch
       f_score       : TensorFlow tensor for model output (cf. model.outputs)
       f_geometry    : TensorFlow tensor for model output (cf. model.outputs)
       tile_shape    : tuple (width,height) of the tile size
    """

    image = cv2.imread(image_file)
    image = image[:, :, ::-1] # Convert from OpenCV's BGR to RGB
    image = image[:6144, :7168, :]
    boxes = np.zeros((0,9)) # Initialize array to hold resulting detections

    for level in range(pyramid_levels):
        if level != 0:
            image = cv2.resize( image, (0,0), fx=0.5, fy=0.5,
                                interpolation=cv2.INTER_CUBIC )

        image_tiles, shifts = create_tile_set(image, tile_shape)

        
        for i in range(len(image_tiles)):
            print('predicting tile',i+1,'of', len(image_tiles))
            tile = np.expand_dims(image_tiles[i], axis=0)
            shift = shifts[i]
            print('tile_shape', tile.shape)
            score, geometry = model(tile, training=False)
            tile_boxes = convert_geometry_to_boxes(
                score_map=score,
                geo_map=geometry,
                detect_thresh=detect_thresh)

            if len(tile_boxes) != 0:
                # Shift boxes to global image coords from tile-specific coords
                shift = np.asarray([shift[0],shift[1],
                                    shift[0],shift[1],
                                    shift[0],shift[1],
                                    shift[0],shift[1],
                                    0])
                tile_boxes = tile_boxes[:,:]+shift
                # Resize tile boxes to global image coords from pyramid-level
                tile_boxes[:,:-1] *= (2**level)
                boxes = np.concatenate((boxes, tile_boxes), axis=0)

    #idx_by_score = tf.argsort(boxes[:, -1], axis=-1, direction='DESCENDING')
    #sorted_boxes = boxes[idx_by_score]
    """
    selected_indices = tf.image.non_max_suppression(boxes[:, [0, 1, 5, 6]], 
                                                    boxes[:, -1], 100000, 
                                                    iou_threshold=0.5, 
                                                    score_threshold=0.7)

    final_boxes = boxes[selected_indices]  
    """
    return boxes

modelpath = 'D:/Gerasimos/Toponym_Recognition/MapTD_General/MapTD_TF2/data/ckpts/models/0/modified_model'
model = keras.models.load_model(modelpath)
image_path = 'D:/Gerasimos/1024-5028149.tiff'
boxes = predict_v2(model, image_path, (1024, 1024))
print(boxes.shape)
positive_rows = []
for row, values in enumerate(boxes):
    if np.min(values) >= 0:
        positive_rows.append(row) 

boxes = boxes[positive_rows]

int_boxes = boxes[:, :8].astype(int)
scores = boxes[:, 8]

selected_indices = nms(int_boxes[:, [0, 1, 4, 5]], scores, 350000, iou_threshold=0.5)

selected_boxes = tf.gather(int_boxes, selected_indices)

reshaped_boxes = np.zeros((selected_boxes.shape[0], 4, 2))

for i in range(selected_boxes.shape[0]):
    reshaped_boxes[i][0] = selected_boxes[i, :2]
    reshaped_boxes[i][1] = selected_boxes[i, 2:4]
    reshaped_boxes[i][2] = selected_boxes[i, 4:6]
    reshaped_boxes[i][3] = selected_boxes[i, 6:]
    
reshaped_boxes = reshaped_boxes.astype(int)

xy_reshaped = reshaped_boxes[:, :, ::-1]

img = plt.imread('D:/Gerasimos/1024-5028149.tiff')

render_boxes(img, xy_reshaped)



