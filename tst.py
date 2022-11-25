import matplotlib.pyplot as plt


img_path = '/media/gerasimos/Νέος τόμος/Gerasimos/Toponym_Recognition/MapTD_General/D5005-5028149.tiff'

img = plt.imread(img_path)
img = img[:6144, :7168, :]
plt.imsave('b_D5005-5028149.tiff', img)

#plt.savefig('b_D5005-5028149.tiff', dpi=1000, bbox_inches='tight')