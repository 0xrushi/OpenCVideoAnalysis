import logging as logging2 
import argparse
import os
from tkinter import Y
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import sys
sys.path.insert(0, '/mnt/hdd2/gender_detect')
from outerutils.constants import *
sys.path.insert(0, f'{ROOT_FOLDER}/main_models/yoloface/')
from face_detector import YoloDetector

import cv2
import pandas as pd
# from main_models.PyVGGFace.lib import VGGFace
from main_models.race_model.model import get_race_model

sys.path.insert(0, f'{ROOT_FOLDER}/main_models/arcface/')
from main_models.arcface import arcface

from transformers import ViTFeatureExtractor, ViTForImageClassification

os.chdir(ROOT_FOLDER)


class Database:
    def __init__(self, IMAGES_PATH=None):
        self.database = {}
        self.IMAGES_PATH = IMAGES_PATH
        self.threshold = 0.1
        for filename in glob.glob(os.path.join(IMAGES_PATH, '*.png')):
            # load image
            # print(filename)

            # use the name in the filename as the identity key
            identity = os.path.splitext(os.path.basename(filename))[0]

            self.database[identity] = filename
    
    def insert(self, img_path, frame=None):
        # get filename without previous path and extension
        identity = os.path.splitext(os.path.basename(img_path))[0]
        if frame is None:
            frame = np.asarray(Image.open(img_path))
        self.database[identity] = img_path

    def check_if_exists(self, new_image_path):
        # returns true and image_name if image exists in the database
        maxsim = 0
        name = None
        for key in self.database:
            try:
                print(f"trying inmages {self.database[key]}, {new_image_path}")
                _, sim = arcface.inference(self.database[key], new_image_path)
                if sim > maxsim and sim > self.threshold:
                    maxsim = sim
                    name = key
            except Exception as e:
                print(e)
        return maxsim > 0, name
    def get(self):
        return self.database

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

# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
# Load age prediction model
# age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
# Init model, transforms
age_net = model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
age_transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
# load race model
race_net  = get_race_model()

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

def predict_race(face_img):
    races = ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic']
    img = np.asarray(Image.fromarray(face_img).resize((224, 224)))
    img = np.expand_dims(img, 0)
    out = race_net(img)
    return races[np.argmax(out)]

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
    labels = {0:"0-2", 1: "3-9" , 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7:"60-69",8:"more than 70"} 
    # Transform our image and pass it through the model
    inputs = age_transforms(face_img, return_tensors='pt')
    output = age_net(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    preds = proba.argmax(1)

    values, indices = torch.topk(proba, k=1)

    preds, age_confidence_score = list({labels[i.item()]: v.item() for i, v in zip(indices.numpy()[0], values.detach().numpy()[0])}.items())[0]

    # Draw the box
    label = f"Age: {preds}-{age_confidence_score*100:.1f}%"
    return label

# def predict_age(face_img):
#     """
#     Predict the age of the face shown in the image
#     Input: face_img, numpy array
#     Return: gender label
#     """
#     blob = cv2.dnn.blobFromImage(
#         image=face_img, scalefactor=1.0, size=(227, 227),
#         mean=MODEL_MEAN_VALUES, swapRB=False
#     )
#     age_net.setInput(blob)
#     age_preds =  age_net.forward()
#     i = age_preds[0].argmax()
#     age = AGE_INTERVALS[i]
#     age_confidence_score = age_preds[0][i]
#     # Draw the box
#     label = f"Age: {age}-{age_confidence_score*100:.1f}%"
#     return label

model = YoloDetector(target_size=720, gpu=0, min_face=90)


def run_on_frame(frame, video_name, frameid, df, timestamp, db):
    orgimg = frame
    bboxes, _ = model(orgimg)
    # print(f"bboxes are {bboxes}, locations are {locations}")
    # print(f"original image shape is {orgimg.shape}, \n bboxes are {bboxes}")
    logger.debug('Processing frame %s from video %s ',frameid, video_name)
    JUST_SAVE_BOUNDING_BOXES = False
    img_path = 'output/just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid)
    pp, name = None, ""

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
        
        pp  = paint_detected_face_on_image(orgimg, bbox, name)
        if pp is not None:
            face_save_path = 'output/just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid)
            face_save_path2 = 'output/just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid)

            logger.debug('Valid face found, saving face at %s', face_save_path)
            # print(face_save_path)
            cv2.imwrite(face_save_path, face)
            exists, name_from_db = db.check_if_exists(face_save_path)
            if not exists:
                logger.debug('Face at %s not found in DB, adding...', face_save_path)
                db.insert(face_save_path)
                print("inserted in db ", face_save_path)
            else:
                name = name_from_db

        print(f"name is {name}")

        if not JUST_SAVE_BOUNDING_BOXES:
            # custom_plot(face)
            # apply gender prediction
            gender_label = predict_gender(face)
            # apply age prediction
            age_label = predict_age(face)
            # apply race prediction
            race_label = predict_race(face)

            labeltext = f"Person  {gender_label} \n {age_label} \n {race_label} \n {name}"
            y0, dy = y2+20, 22
            # The loop below is to put text one below other, we cannot use \n directly| change y0 and dy as per your screen size
            for i, line in enumerate(labeltext.split('\n')):
                y = y0 + i*dy
                cv2.putText(orgimg, line, (x1+10, y), 1, 1.8, (0,255,0))
            logger.debug('Saving gender, age,race %s', 'output/just_yolo_frames2/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid))
            cv2.imwrite('output/just_yolo_frames2/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid), orgimg)
            df.loc[len(df)] = [frameid, round(timestamp, 2), bbox, img_path, name, gender_label, age_label, race_label]
            # custom_plot(orgimg)

def run_on_video(video_name, df):
    logger.debug('Processing video %s', video_name)
    cap = cv2.VideoCapture(video_name)
    count = 0
    FRAME_SKIP = 5

    SAVE_PATH = 'output/just_yolo_frames/{0}'.format(os.path.splitext(video_name)[0])
    SAVE_PATH2 = 'output/just_yolo_frames2/{0}'.format(os.path.splitext(video_name)[0])

    print(SAVE_PATH, SAVE_PATH2)

    # raise SystemExit('lavda')

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
            run_on_frame(frame, video_name, count, df, timestamp, db)
            # cv2.imwrite('just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], count), frame)
            count += FRAME_SKIP # i.e. at 10 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            df.to_csv(f"{SAVE_PATH}/export.csv", index=False)
            break



if __name__ == '__main__':
    # path for initial images in the databases, images here should be unique
    IMAGES_PATH = f'{ROOT_FOLDER}/unique'

    # videos_list = ['19288/1524962.mp4']
    # videos_list = glob.glob('data/19288/*.mp4')[0:100]
    videos_list = glob.glob('data/lebron/*.mp4')
    # videos_list = glob.glob('19288/1524935.mp4')
    # videos_list = glob.glob('19288/1575178.mp4')


    # videos_list = np.array_split(videos_list, 12)
    print("videos list ", videos_list)
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--index', type=int, required=True)

    args = parser.parse_args()

    # videos_list = videos_list[args.index]
    
    logging2.basicConfig(
    # filename=f'yololog1.log',
    level=logging2.DEBUG, 
    format='%(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        logging2.FileHandler(f"logs/yololog{args.index}.log"),
        logging2.StreamHandler()
    ], force=True)
    logger = logging2.getLogger("server_log")

    print(videos_list)
    # Process each video
    for video in videos_list:
        df = pd.DataFrame(columns = ["frameid", "timestamp", "bbloc", "img_path", "name", "gender", "age", "race"])
        run_on_video(video, df)