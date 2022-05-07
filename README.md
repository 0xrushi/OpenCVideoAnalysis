# Automate video analysis with deep learning

This project is a free and open-source video analytics service that can be easily integrated into applications without prior deep learning knowledge. The service can seamlessly scale to process petabytes of data.

## Features

**Face compare and search**
**Gender detection**
**Age detection**
**Video segment detection**

**Customizable pretrained models for each function**
**Flexible deployment**

## Use cases

**Smart home alerts**
**Multimedia analysis**
**Spatial analysis**


#### Please refer to my article for more detailed analysis [A comparison of different Face Detection APIs in the industry](https://rushichaudhari.github.io/posts/2022-05-06-why-do-we-need-an-opensource-face-detection-apis/)

---

# How to run

`virtualenv --python=/usr/bin/python3.8 ./python38venv`

`source python38venv/bin/activate`

`export PYTHONPATH="${PYTHONPATH}:/mnt/hdd2/gender_detect/yoloface/"`

--- 
## Things using

yoloface: https://github.com/sthanhng/yoloface
vit transformers: https://huggingface.co/docs/transformers/model_doc/vit
arcface

## Things used before

face_recognition: https://github.com/ageitgey/face_recognition
face.evoLVe: https://github.com/ZhaoJ9014/face.evoLVe
## Things failed (didn't work well with pretrained weights)

Insightface: https://github.com/deepinsight/insightface

Deepface: https://github.com/serengil/deepface

CompreFace: https://github.com/exadel-inc/CompreFace
```
# compreface
curl -i -X POST -H "Content-Type: multipart/form-data" -F "file=@ /mnt/hdd2/gender_detect/1.png" -F "compareImage=@ /mnt/hdd2/gender_detect/3.png" http://127.0.0.1:8000/api/simface/
```

## An example of model2_on_all_videos.py on 19288/1524962.mp4

**Conclusion:**
- Not all frames are processed eg. 0, 10, 40 but as the time delay is very less, we can still draw out some conclusions
- After frameid 730 there are no bounding boxes detected, 730th frame = 730/30 seconds = 24 seconds, which is true in 19288/1524962.mp4 no person appears after 24 seconds
- Frame530 has two entries, because there are two faces in the frame

```
─➤  cat 1524962/export.csv
frameid,bbloc,img_path,name
0,,just_yolo_frames/19288/1524962/frame0.jpg,
10,,just_yolo_frames/19288/1524962/frame10.jpg,
20,"(116, 808, 223, 700)",just_yolo_frames/19288/1524962/frame20.jpg,
30,"(82, 726, 211, 597)",just_yolo_frames/19288/1524962/frame30.jpg,frame20
40,,just_yolo_frames/19288/1524962/frame40.jpg,
50,"(96, 669, 225, 540)",just_yolo_frames/19288/1524962/frame50.jpg,frame20
60,,just_yolo_frames/19288/1524962/frame60.jpg,
70,"(82, 655, 211, 526)",just_yolo_frames/19288/1524962/frame70.jpg,frame20
80,,just_yolo_frames/19288/1524962/frame80.jpg,
90,"(82, 683, 211, 554)",just_yolo_frames/19288/1524962/frame90.jpg,frame20
100,,just_yolo_frames/19288/1524962/frame100.jpg,
110,"(96, 669, 225, 540)",just_yolo_frames/19288/1524962/frame110.jpg,frame20
120,,just_yolo_frames/19288/1524962/frame120.jpg,
130,"(82, 669, 211, 540)",just_yolo_frames/19288/1524962/frame130.jpg,frame20
140,,just_yolo_frames/19288/1524962/frame140.jpg,
150,"(82, 655, 211, 526)",just_yolo_frames/19288/1524962/frame150.jpg,frame20
160,,just_yolo_frames/19288/1524962/frame160.jpg,
170,"(82, 640, 211, 511)",just_yolo_frames/19288/1524962/frame170.jpg,frame20
180,,just_yolo_frames/19288/1524962/frame180.jpg,
190,"(82, 640, 211, 511)",just_yolo_frames/19288/1524962/frame190.jpg,frame20
200,,just_yolo_frames/19288/1524962/frame200.jpg,
210,"(82, 655, 211, 526)",just_yolo_frames/19288/1524962/frame210.jpg,frame20
220,,just_yolo_frames/19288/1524962/frame220.jpg,
230,"(260, 474, 322, 411)",just_yolo_frames/19288/1524962/frame230.jpg,
230,"(274, 792, 336, 729)",just_yolo_frames/19288/1524962/frame230.jpg,
240,,just_yolo_frames/19288/1524962/frame240.jpg,
250,"(254, 478, 329, 403)",just_yolo_frames/19288/1524962/frame250.jpg,
250,"(271, 793, 345, 718)",just_yolo_frames/19288/1524962/frame250.jpg,frame230
260,,just_yolo_frames/19288/1524962/frame260.jpg,
270,"(262, 478, 337, 403)",just_yolo_frames/19288/1524962/frame270.jpg,frame250
270,"(281, 785, 343, 722)",just_yolo_frames/19288/1524962/frame270.jpg,frame230
280,,just_yolo_frames/19288/1524962/frame280.jpg,
290,"(96, 640, 225, 511)",just_yolo_frames/19288/1524962/frame290.jpg,frame20
300,,just_yolo_frames/19288/1524962/frame300.jpg,
310,"(96, 655, 225, 526)",just_yolo_frames/19288/1524962/frame310.jpg,frame20
320,,just_yolo_frames/19288/1524962/frame320.jpg,
330,"(96, 655, 225, 526)",just_yolo_frames/19288/1524962/frame330.jpg,frame20
340,,just_yolo_frames/19288/1524962/frame340.jpg,
350,"(96, 640, 225, 511)",just_yolo_frames/19288/1524962/frame350.jpg,frame20
360,,just_yolo_frames/19288/1524962/frame360.jpg,
370,"(96, 640, 225, 511)",just_yolo_frames/19288/1524962/frame370.jpg,frame20
380,,just_yolo_frames/19288/1524962/frame380.jpg,
390,"(110, 626, 239, 497)",just_yolo_frames/19288/1524962/frame390.jpg,frame20
400,,just_yolo_frames/19288/1524962/frame400.jpg,
410,"(110, 626, 239, 497)",just_yolo_frames/19288/1524962/frame410.jpg,frame20
420,,just_yolo_frames/19288/1524962/frame420.jpg,
430,"(96, 626, 225, 497)",just_yolo_frames/19288/1524962/frame430.jpg,frame20
440,,just_yolo_frames/19288/1524962/frame440.jpg,
450,"(110, 612, 239, 483)",just_yolo_frames/19288/1524962/frame450.jpg,frame20
460,,just_yolo_frames/19288/1524962/frame460.jpg,
470,"(110, 626, 239, 497)",just_yolo_frames/19288/1524962/frame470.jpg,frame20
480,,just_yolo_frames/19288/1524962/frame480.jpg,
490,"(116, 614, 270, 459)",just_yolo_frames/19288/1524962/frame490.jpg,frame20
500,,just_yolo_frames/19288/1524962/frame500.jpg,
510,"(125, 612, 254, 483)",just_yolo_frames/19288/1524962/frame510.jpg,frame20
520,,just_yolo_frames/19288/1524962/frame520.jpg,
530,"(196, 382, 325, 253)",just_yolo_frames/19288/1524962/frame530.jpg,frame250
530,"(182, 984, 311, 855)",just_yolo_frames/19288/1524962/frame530.jpg,frame230
540,,just_yolo_frames/19288/1524962/frame540.jpg,
550,"(112, 617, 379, 349)",just_yolo_frames/19288/1524962/frame550.jpg,frame20
560,,just_yolo_frames/19288/1524962/frame560.jpg,
570,"(171, 587, 439, 319)",just_yolo_frames/19288/1524962/frame570.jpg,frame20
580,,just_yolo_frames/19288/1524962/frame580.jpg,
590,"(142, 646, 409, 379)",just_yolo_frames/19288/1524962/frame590.jpg,frame20
600,,just_yolo_frames/19288/1524962/frame600.jpg,
610,"(142, 617, 409, 349)",just_yolo_frames/19288/1524962/frame610.jpg,frame20
620,,just_yolo_frames/19288/1524962/frame620.jpg,
630,"(171, 587, 439, 319)",just_yolo_frames/19288/1524962/frame630.jpg,frame20
640,,just_yolo_frames/19288/1524962/frame640.jpg,
650,"(170, 598, 491, 277)",just_yolo_frames/19288/1524962/frame650.jpg,frame20
660,,just_yolo_frames/19288/1524962/frame660.jpg,
670,"(196, 999, 325, 870)",just_yolo_frames/19288/1524962/frame670.jpg,frame230
670,"(182, 368, 311, 239)",just_yolo_frames/19288/1524962/frame670.jpg,frame250
680,,just_yolo_frames/19288/1524962/frame680.jpg,
690,"(196, 999, 325, 870)",just_yolo_frames/19288/1524962/frame690.jpg,frame230
690,"(182, 368, 311, 239)",just_yolo_frames/19288/1524962/frame690.jpg,frame250
700,,just_yolo_frames/19288/1524962/frame700.jpg,
710,"(116, 752, 270, 597)",just_yolo_frames/19288/1524962/frame710.jpg,frame20
720,,just_yolo_frames/19288/1524962/frame720.jpg,
730,"(153, 812, 282, 683)",just_yolo_frames/19288/1524962/frame730.jpg,frame20
740,,just_yolo_frames/19288/1524962/frame740.jpg,
750,,just_yolo_frames/19288/1524962/frame750.jpg,
760,,just_yolo_frames/19288/1524962/frame760.jpg,
770,,just_yolo_frames/19288/1524962/frame770.jpg,
780,,just_yolo_frames/19288/1524962/frame780.jpg,
790,,just_yolo_frames/19288/1524962/frame790.jpg,
800,,just_yolo_frames/19288/1524962/frame800.jpg,
810,,just_yolo_frames/19288/1524962/frame810.jpg,
820,,just_yolo_frames/19288/1524962/frame820.jpg,
830,,just_yolo_frames/19288/1524962/frame830.jpg,
840,,just_yolo_frames/19288/1524962/frame840.jpg,
850,,just_yolo_frames/19288/1524962/frame850.jpg,
860,,just_yolo_frames/19288/1524962/frame860.jpg,
870,,just_yolo_frames/19288/1524962/frame870.jpg,
880,,just_yolo_frames/19288/1524962/frame880.jpg,
890,,just_yolo_frames/19288/1524962/frame890.jpg,
```