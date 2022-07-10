from cProfile import label
from unicodedata import name
import cv2
import os
import numpy as np

dataPath = "./Reconocimiento Facial/Base de Datos"
peopleList = os.listdir(dataPath)
print('Lista de Personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        cv2.imshow('image',image)
        cv2.waitKey(10)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print('Entrenando...')

face_recognizer.train(facesData,np.array(labels))

face_recognizer.write('modeloEigenFace.xml')
print('Modelo Almacenado....')