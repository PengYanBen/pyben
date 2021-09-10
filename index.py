import cv2
import numpy as np
from vendor.graphical.treatment import *




source_path = './static/img/2.jpg'
out_path = './static/img/out.jpg'

img = cv2.imread(source_path).astype(np.float32)
#img = tf.Transform(img).otsu_binarization()
img = tf.Transform(img).HSVTranspose()
# Save result
cv2.imwrite(out_path, img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





