import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disables INFO & WARNING logs
from visualize_modified import render_boxes, load_boxes
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
from utils import center_crop

default_path = '/media/gerasimos/Νέος τόμος/Gerasimos/Toponym_Recognition/MapTD_General/MapTD_TF2'

impath = askopenfilename(title='Select the image', initialdir=default_path)
txtpath = askopenfilename(title='Select the .txt file')

img = plt.imread(impath)
boxes = load_boxes(txtpath)

render_boxes(img, boxes)

savepath = os.path.join(txtpath.split(os.path.basename(txtpath))[0], 'pred_ims', f'new_{os.path.basename(txtpath).split(".")[0]}.png')
print('Saving the image with the rendered boxes in: ', savepath)
plt.savefig(savepath, dpi=1000, bbox_inches='tight')


