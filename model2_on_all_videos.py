import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
# sys.path.insert(0, '/mnt/hdd2/gender_detect/yoloface/')
sys.path.insert(0, '/mnt/hdd2/gender_detect/face.evoLVe/applications/align')

from detector import detect_faces
from visualization_utils import show_results
import face_recognition

import cv2
import numpy as np
import glob
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd

import cv2

class Database:
    def __init__(self, IMAGES_PATH=None):
        self.database = {}
        self.IMAGES_PATH = IMAGES_PATH 
        for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
            # load image
            print(filename)
            image_rgb = face_recognition.load_image_file(filename)

            # use the name in the filename as the identity key
            identity = os.path.splitext(os.path.basename(filename))[0]

            # get the face encoding and link it to the identity
            locations, encodings = self.get_face_embeddings_from_image(image_rgb)
            
            self.database[identity] = encodings[0]
            
    def get_face_embeddings_from_image(self, image, convert_to_rgb=False):
        """
        Take a raw image and run both the face detection and face embedding model on it
        """
        # Convert from BGR to RGB if needed
        if convert_to_rgb:
            image = image[:, :, ::-1]

        # run the face detection model to find face locations
        face_locations = face_recognition.face_locations(image)

        # run the embedding model to get face embeddings for the supplied locations
        face_encodings = face_recognition.face_encodings(image, face_locations)

        return face_locations, face_encodings
    
    def insert(self, img_path, frame=None):
        # get filename without pre path and extension
        identity = os.path.splitext(os.path.basename(img_path))[0]
        if frame is None:
            frame = np.asarray(Image.open(img_path))
        # get the face encoding and link it to the identity
        locations, encodings = self.get_face_embeddings_from_image(frame)
        self.database[identity] = encodings[0]
    def get(self):
        return self.database

def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location

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

def run_on_frame(db, frame, video_name, frameid, df, timestamp):
    """
    Start the face recognition via the webcam
    """
    MAX_DISTANCE = 0.6 
    img_path = 'just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid)
    pp, name = None, ""
    # the face_recognitino library uses keys and values of your database separately
    known_face_encodings = list(db.database.values())
    known_face_names = list(db.database.keys())
    # run detection and embedding models
    face_locations, face_encodings = db.get_face_embeddings_from_image(frame, convert_to_rgb=True)

    # Loop through each face in this frame of video and see if there's a match
    for location, face_encoding in zip(face_locations, face_encodings):
        # get the distances from this encoding to those of all reference images
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        # select the closest match (smallest distance) if it's below the threshold value
        if np.any(distances <= MAX_DISTANCE):
            best_match_idx = np.argmin(distances)
            name = known_face_names[best_match_idx]
        else:
            name = None
            db.insert(img_path, frame)
        # put recognition info on the image
        pp  = paint_detected_face_on_image(frame, location, name)
        if pp is not None:
            print('just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid))
            cv2.imwrite('./just_yolo_frames/{0}/frame{1}.jpg'.format(os.path.splitext(video_name)[0], frameid), pp)
        print(name)
        df.loc[len(df)] = [frameid, round(timestamp, 2), location, img_path, name]
    if not face_locations or not face_encodings:
        df.loc[len(df)] = [frameid,  round(timestamp, 2), None, img_path, None]
    # plt.imshow(pp)
    # plt.show()

def run_on_video(video_name, db, df):
    cap = cv2.VideoCapture(video_name)
    # count the number of frames
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    FRAME_SKIP = 10

    SAVE_PATH = 'just_yolo_frames/{0}'.format(os.path.splitext(video_name)[0])
    # create save path if doesn't exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    while cap.isOpened():
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
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
IMAGES_PATH = '/mnt/hdd2/gender_detect/just_yolo_frames/19288/unique'
db = Database(IMAGES_PATH)
df = pd.DataFrame(columns = ["frameid", "timestamp", "bbloc", "img_path", "name"])

# run_on_video('19288/1524962.mp4', db, df)
videos_list = glob.glob('19288/*.mp4')[:4]
print(videos_list)
for video in videos_list:
    run_on_video(video, db, df)

# print(db.get().keys())
# print(df)
