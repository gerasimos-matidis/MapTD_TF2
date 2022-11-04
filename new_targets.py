# NOTE: For local use. This script contains modified code of the authors refered 
# below and it is not meant to be published. In case of publication please 
# inform me at mat.gera@hotmail.com, so as I will remove it.

# MapTD
# Copyright (C) 2018 Jerod Weinman, Nathan Gifford, Abyaya Lamsal, 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numpy.linalg import norm
import cv2

"""In all the routines below, we assume the rectangle coordinates are
sorted in counter-clockwise order, so that the first point corresponds
to the left point of the text baseline (presuming a left-to-right reading
order) as in these examples.


  3 *---------* 2         3 *                 2 *
    | T E X T |            / `                 / `* 1
  0 *---------* 1       0 * T `               / r/
                           ` h `             / a/
                            ` e * 2       3 * B/
                             ` /             `* 0
                              * 1           
"""
def get_angle ( line_segment ):
    """Angle of the line segment and an array of its cosine and sin
    
    Parameters:
       line_segment: A 1x2 np array of the directed line segment (x,y)
    Returns:
       angle: The scalar angle of the directed line segment (in radians)
       cos_sin: A 1x2 np array containing the cosine and sine of that angle
"""
    angle =  np.arctan2( line_segment[1], line_segment[0] )
    cos_sin = np.array([np.cos(angle), np.sin(angle)])
    return angle, cos_sin


def shrink_rect( rect, shrink_ratio=0.3): 
    """ Shrink the edges of a rectangle by a fixed relative factor. The
        effect should be equivalent to scaling the height and width of a
        rotated box represented as a center, size, and rotated angle.
    
    Parameters:
        rect: A 4x2 numpy array indicating the coordinates of the four rectangle
              vertices
        shrink_ratio: A scalar in (0,0.5) indicating by how much to move each 
                      point along the line segment representing a rectangle 
                      side. [default 0.3]
    Returns:
        shrunk: A 4x2 numpy array with the modified rectangle points
    """

    # Modeled on Eq. (3) in Zhou et al. (EAST), but the mod is outside the +/- 1
    # due to Python's zero-based indexing
    reference_lengths = [ min( norm( rect[c] - rect[(c+1)%4] ),
                               norm( rect[c] - rect[(c-1)%4] ) ) 
                          for c in range(4) ]

    shrunk = rect.copy().astype(np.float32) # Create a clean copy for mutation

    # Find the longer pair of edges --- 
    # {<p0,p1>,<p3,p2>} versus {<p0,p3>,<p1,p2>}
    len_01_32 = norm(rect[0] - rect[1]) + norm(rect[3] - rect[2])
    len_03_12 = norm(rect[0] - rect[3]) + norm(rect[1] - rect[2])

    # Local helper function to shrink a line segment <start,end>
    def shrink(start,end):
        cos_sin = get_angle(rect[end]-rect[start])[1]
        shrunk[start] += shrink_ratio * reference_lengths[start] * cos_sin
        shrunk[end]   -= shrink_ratio * reference_lengths[end]   * cos_sin
    # Local helper function to shrink all edges in given order
    def shrink_edges(edges):
        for edge in edges:
            shrink(edge[0],edge[1])

    # Move the longer axes first then shorter axes 
    if len_01_32 > len_03_12:
        shrink_edges( [[0,1],[3,2],[0,3],[1,2]] )
    else:
        shrink_edges( [[0,3],[1,2],[0,1],[3,2]] )

    return shrunk


def dist_to_line(p0, p1, points):
    """ Calculate the distance of points to the line segment <p0,p1> """
    norm1 = norm( p1-p0 )
    if norm1 == 0:
        print(p0, p1) # NOTE: Gerasimos wrote the "print" as a function, it was a decorator (python 2)
        norm1 = 1.0
    return np.abs( np.cross(p1-p0, points-p0) / norm1 )

def generate(image_size, rectangles):
    """ Generate the label maps for training from the preprocessed rectangles 
        intersecting the cropped subimage. 

    Parameters:
       image_size: A two-element tuple [image_height,image_width]
       rects: An 4x2xN numpy array containing the coordinates of the four 
              rectangle vertices. The zeroth dimension runs clockwise around the
              rectangle (as given by sort_rectangle), the first dimension is 
              (x,y), and the last dimension is the particular rectangle.
    Returns:
       score_map : An image_size/4 array of ground truth labels (in {0,1}) for 
                     shrunk versions of the given rectangles
       geo_map   : An image_size/4 x 5 array of geometries for the shrunk 
                     rectangles; the final dimension contains the distances to 
                     the top, left, bottom, and right rectangle eges, as well as
                     the oriented angle of the top edge in [0,2*pi)
       train_mask: An image_size/4 x 1 array of weights (in {0,1}) to include in
                     the loss function calculations.
    """

    # ---------------------------------------------------------------------------
    # Set up return values 
    # Where a given rectangle is located
    rect_mask = np.zeros( image_size, dtype=np.uint8) 

    # Pixel-wise positive/negative class indicators for loss calculation
    score_map  = np.zeros( image_size, dtype=np.uint8 ) 
    
    # Distances to four rectangle edges and angle
    geo_map = np.zeros( [image_size[0],image_size[1],5], dtype=np.float32)

    # Which pixels are used or ignored during training. Initially 2 (unknown)
    training_mask = np.ones( [image_size[0], image_size[1], 1], dtype=np.uint8 )

    shrunk_rectangles = np.zeros_like(rectangles)
    for i, rect in enumerate(rectangles):
        # Shrink the rectangle, and put in a fillPoly-friendly format
        shrunk_rectangles[i] = shrink_rect(rect)
       
    # Set ground truth pixels to detect
    score_map = cv2.fillPoly(score_map, pts=shrunk_rectangles.astype(np.int32), color=1)

    cv2.fillPoly(training_mask, pts=rectangles.astype(np.int32), color=0)
    cv2.fillPoly(training_mask, pts=shrunk_rectangles.astype(np.int32), color=1)


    return np.expand_dims(score_map[::4, ::4], axis=-1), training_mask[::4, ::4, :]
