import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("bmp") and file in ['1.bmp', '2.bmp', '4.bmp', '7.bmp', '9.bmp']:
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label)  # some number
            # verify this image, turn into a NUMPY arrray, GRAY
            # x_train.append(path)
            pil_image = Image.open(path).convert("L")  # grayscale
            # img = cv2.imread(path)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array)
            print(faces)
            # if type(faces) is np.ndarray:
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            # cv2.imshow("Adding faces to traning set...", roi)
            # cv2.waitKey(10)


# print(y_labels)
# print(x_train)

with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")
print("\n [INFO] Faces trained. Exiting Program...")
# cv2.destroyAllWindows()
