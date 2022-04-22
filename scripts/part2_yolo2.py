import os
from tkinter import Y
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from outerutils.constants import *
import sys
sys.path.insert(0, f'{ROOT_FOLDER}/yoloface/')
from face_detector import YoloDetector

import cv2
import pandas as pd
import face_recognition
from PyVGGFace.lib import VGGFace
from SiameseNet.model import SiameseNetwork, preprocess_siamese

class VGFace:
    def __init__(self):
        self.model = VGGFace().double()
        model_dict = torch.load('weights/vggface.pth', map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_dict)
    def load_img(self, image_arr):
        # img = cv2.imread(img_path)
        img = cv2.resize(image_arr, (224, 224))
        img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224).double()
        img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
        return img
    def get_encoding(self, image_arr):
        img = self.load_img(image_arr)
        return self.model(img)

def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    # top, right, bottom, left = location
    left, top, right, bottom = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    return frame
class Database:
    def __init__(self, IMAGES_PATH=None):
        self.database = {}
        self.IMAGES_PATH = IMAGES_PATH
        self.vgg_face = VGFace()

        for filename in glob.glob(os.path.join(IMAGES_PATH, '*.png')):
            # load image
            print(filename)
            image_rgb = Image.open(filename)

            # use the name in the filename as the identity key
            identity = os.path.splitext(os.path.basename(filename))[0]

            # get the face encoding and link it to the identity
            # encodings = self.get_face_embeddings_from_image(image_rgb)
            
            self.database[identity] = filename
            
    def get_face_embeddings_from_image(self, image, convert_to_rgb=False):
        """
        Take a raw image and run both the face detection and face embedding model on it
        """
        # Convert from BGR to RGB if needed
        if convert_to_rgb:
            image = image[:, :, ::-1]

        return self.vgg_face.get_encoding(image).detach().numpy()[0]
    
    def insert(self, img_path, frame=None):
        # get filename without pre path and extension
        identity = os.path.splitext(os.path.basename(img_path))[0]
        if frame is None:
            frame = np.asarray(Image.open(img_path))
        # get the face encoding and link it to the identity
        # encodings = self.get_face_embeddings_from_image(frame)
        self.database[identity] = img_path
    def get(self):
        return self.database

# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

siamese_model = SiameseNetwork()
siamese_model.load_state_dict(torch.load('weights/siamese_net.pt'))

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)    

def predict_gender(face_img):
    """
    Predict the gender of the face shown in the image
    Input: face_img, numpy array
    Return: gender label
    """
    print("face_img.shape", face_img.shape)
    if face_img.shape[1] > 105:
        face_img = image_resize(face_img, width=105)

    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()

    i = gender_preds[0].argmax()
    gender = GENDER_LIST[i]
    gender_confidence_score = gender_preds[0][i]
    label = f"{gender}-{gender_confidence_score*100:.1f}%"
    print(label)
    return label

def predict_age(face_img):
    """
    Predict the age of the face shown in the image
    Input: face_img, numpy array
    Return: gender label
    """
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    age_preds =  age_net.forward()
    i = age_preds[0].argmax()
    age = AGE_INTERVALS[i]
    age_confidence_score = age_preds[0][i]
    # Draw the box
    label = f"Age: {age}-{age_confidence_score*100:.1f}%"
    return label

model = YoloDetector(target_size=720, gpu=1, min_face=90)


def run_on_frame(db, frame, video_name, frameid, df, timestamp):
    orgimg = frame
    bboxes, _ = model(orgimg)
    # print(f"bboxes are {bboxes}, locations are {locations}")
    # print(f"original image shape is {orgimg.shape}, \n bboxes are {bboxes}")
    JUST_SAVE_BOUNDING_BOXES = True
    MAX_DISTANCE = 100
    img_path = 'just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid)
    pp, name = None, ""
    # the face_recognitino library uses keys and values of your database separately
    known_face_image_files = list(db.database.values())
    # print('known_face_image_files ', known_face_image_files)
    known_face_names = list(db.database.keys())
    print(f"known_face_names {known_face_names}")

    # iterate through all the faces
    for bbox in bboxes[0]:
        # print(c)
        rect = bbox
        x1,y1,x2,y2 = rect
        h = y2-y1
        w = x2-x1
        # print(f"h is {h} \n w is {w}")
        cv2.rectangle(orgimg,(x1,y1),(x2,y2),(0,255,0), 2)
        # keep in mind index become reversed during cropping
        face = orgimg[y1:y2, x1:x2].copy()
        
        img0 = preprocess_siamese(face)
        if len(known_face_image_files)<1:
            known_face_image_files.append(img_path)
            known_face_names.append(os.path.splitext(os.path.basename(img_path))[0])
            db.insert(img_path, face)
            cv2.imwrite('./just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid), face)
        for i, img1_filename in enumerate(known_face_image_files):
            img1 = preprocess_siamese(np.asarray(Image.open(img1_filename)))
            output1, output2 = siamese_model(img0, img1)
            # output1 = round(float(output1), 2)
            # output2 = round(float(output2), 2)
            distance = F.pairwise_distance(output1, output2)[0]
            print(f"distanbce is {distance}")
            if distance < 1.7:
                name = known_face_names[i]
                break
            else:
                name = None
                db.insert(img_path, face)
                
        pp  = paint_detected_face_on_image(orgimg, bbox, name)
        if pp is not None:
            print('just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid))
            cv2.imwrite('./just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid), face)
            cv2.imwrite('./just_yolo_frames2/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid), orgimg)
        print(f"name is {name}")
        df.loc[len(df)] = [frameid, round(timestamp, 2), bbox, img_path, name]

        if not JUST_SAVE_BOUNDING_BOXES:
            custom_plot(face)
            # apply gender prediction
            gender_label = predict_gender(face)
            # apply age prediction
            age_label = predict_age(face)
            labeltext = f"Person  {gender_label} \n {age_label}"
            y0, dy = y2+20, 22
            # The loop below is to put text one below other, we cannot use \n directly| change y0 and dy as per your screen size
            for i, line in enumerate(labeltext.split('\n')):
                y = y0 + i*dy
                cv2.putText(orgimg, line, (x1+10, y), 1, 1.8, (0,255,0))

            # custom_plot(orgimg)

def run_on_video(video_name, df):
    cap = cv2.VideoCapture(video_name)
    count = 0
    FRAME_SKIP = 5

    SAVE_PATH = 'just_yolo_frames/{0}'.format(os.path.splitext(video_name)[0])
    SAVE_PATH2 = 'just_yolo_frames2/{0}'.format(os.path.splitext(video_name)[0])

    db = Database(IMAGES_PATH)


    # create save path if doesn't exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(SAVE_PATH2):
        os.makedirs(SAVE_PATH2)

    while cap.isOpened():
        ret, frame = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ret:
            run_on_frame(db, frame, video_name, count, df, timestamp)
            # cv2.imwrite('just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], count), frame)
            count += FRAME_SKIP # i.e. at 10 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            df.to_csv(f"{SAVE_PATH}/export.csv", index=False)
            break

# path for initial images in the databases, images here should be unique
IMAGES_PATH = '/mnt/hdd2/gender_detect/unique'
df = pd.DataFrame(columns = ["frameid", "timestamp", "bbloc", "img_path", "name"])

# videos_list = ['19288/1524962.mp4']
videos_list = glob.glob('19288/*.mp4')[20:40]
print(videos_list)
for video in videos_list:
    run_on_video(video, df)