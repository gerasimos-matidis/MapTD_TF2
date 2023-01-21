from tkinter.filedialog import askopenfilename
import os
from shapely.geometry import Polygon
import numpy as np

from data_tools import parse_boxes_from_text, parse_boxes_from_json

pred_file = askopenfilename(title='Select the file with the predictions', 
                        initialdir='predictions_n_evaluation')

gt_dir = 'data/general_dataset/json'
map_name = os.path.splitext(os.path.basename(pred_file))[0]
gt_file = os.path.join(gt_dir, map_name + '.json')

prediction_boxes, _, _, _ = parse_boxes_from_text(pred_file, slice_first=True)
prediction_boxes = [Polygon(pred_box) for pred_box in prediction_boxes]
ground_truth_boxes, _, _ = parse_boxes_from_json(gt_file, slice_first=True)
ground_truth_boxes = [Polygon(gt_box) for gt_box in ground_truth_boxes]

def get_iou(p1, p2):
    if p1.intersects(p2):
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter/union
    else:
        return 0

iou_matrix = np.zeros([len(ground_truth_boxes), len(prediction_boxes)])
for i, gt_box in enumerate(ground_truth_boxes):
    for j, pred_box in enumerate(prediction_boxes):
        iou = get_iou(gt_box, pred_box)
        if iou > 0.5:
            iou_matrix[i, j] = iou

idx = iou_matrix.nonzero()
non_zeros = iou_matrix[idx]
print(f'(Map: {map_name}) The average IoU of the matched boxes is: ', 
      np.mean(non_zeros))
