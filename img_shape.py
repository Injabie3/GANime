#If you encounter error "cannot broadcast array from (x,y,3) to (x,y)" during training, run this script on the images

import glob
import cv2
import os

path = "D:/419/datasets/fixed"
i = 0

for img in glob.glob("D:/419/datasets/anime/*/*.png"):
    print(os.path.join(path , 'face_' + str(i)+ '.png'))
    img = cv2.imread(img)
    (b, g, r)=cv2.split(img)
    img=cv2.merge([r,g,b])    
    cv2.imwrite(os.path.join(path , 'face_' + str(i)+ '.png'), img)
    i = i + 1