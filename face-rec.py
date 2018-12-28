import numpy as np
import cv2
import pickle
import os

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
image_rm = image_dir + "\\"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {}
count = 0
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
for root, dirs, files in os.walk(image_dir):
    obj = root.replace(image_rm, "")
    for file in files:
        if file.endswith("bmp") and file in ['2.bmp', '5.bmp', '6.bmp', '8.bmp', '10.bmp']:
            path = os.path.join(root, file)
            print('Face rec ... {0}', path)
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)
            for (x, y, w, h) in faces:
                # print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
                roi_color = frame[y:y+h, x:x+w]

                # recognize? deep learned model predict keras tensorflow pytorch scikit learn
                id_, conf = recognizer.predict(roi_gray)
                if conf <= 100:
                    name = labels[id_]
                    if name == obj:
                        count += 1
                        print(f"Counts: {count}")
