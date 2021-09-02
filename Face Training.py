

import cv2
import numpy as np
from PIL import Image 
import os


path = 'Dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("FaceRecognition.xml")
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]    
#%%

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = cv2.imread(imagePath,0)
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids
#%%
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
#%%

recognizer.write('trainer.yml') 

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
