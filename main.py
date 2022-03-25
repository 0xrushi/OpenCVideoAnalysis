import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from face_detector import YoloDetector
import cv2

model = YoloDetector(target_size=720,gpu=0,min_face=90)
orgimg = np.array(Image.open('frames_saved/frame30.jpg')).astype(np.uint8)
bboxes, points = model.predict(orgimg)
print(bboxes)

for c in bboxes[0]:
    print(c)
    rect = c
    x,y,w,h = rect
    cv2.rectangle(orgimg,(x,y),(w,h),(0,255,0),2)
    cv2.putText(orgimg,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
cv2.imshow("Show",orgimg)
cv2.waitKey()  
cv2.destroyAllWindows()


# plt.imshow(orgimg)
# plt.show()