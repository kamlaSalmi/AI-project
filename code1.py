import numpy as np
import cv2
import matplotlib.image as im
import numpy
import matplotlib.pyplot as plt
import os

folder = r"datasets\main folder" # import the data from the file
c = os.listdir(folder) #  save list of folders
num_of_criminals = len(c)
print(c)

fd = cv2.CascadeClassifier(r"haarcascade_frontalface_alt.xml") # model to detect the face
# function to get the image and resize it 
def get_image(imgname):
    img = im.imread(imgname) 
    corners = fd.detectMultiScale(img,1.3,4) 
    (x,y,w,h) = corners[0]
    img = img[y-20:y+h+20,x-20:x+w+20] # cropping the face
    img = cv2.resize(img,(100,100)) 
    return img

trainimg = []
trainlb = []
# getting all images saved in img2
for subfolder in c:
    direc = folder+'\\'+subfolder
    img = os.listdir(direc)
    print(img)
    for imgpath in img:
        imgpath = direc + "\\" + imgpath
        img2= get_image(imgpath)
        trainimg.append(img2)
        trainlb.append(subfolder)


# getting images of unknown 
folder2 = r"datasets\unknown"
d = os.listdir(folder2)
unknown_person = len(d)

def get_image2(imgname1):
    img1 = im.imread(folder2+'\\'+imgname1) 
    corners = fd.detectMultiScale(img1,1.3,4) 
    (x,y,w,h) = corners[0]
    img1 = img1[y-20:y+h+20,x-20:x+w+20]
    img1 = cv2.resize(img1,(100,100)) 
    return img1

trainimgun = []
trainlbun = []

for imgname1 in d:
    imgun = get_image2(imgname1)
    trainimg.append(imgun)
    trainlb.append("unknown")
    
#merging all images list in one list 
trainimg = trainimg + trainimgun
trainlb = trainlb + trainlbun

trainimg = numpy.array(trainimg)
# reshape the image using -1 when the no. of images not counted
trainimg = trainimg.reshape(-1,1,100,100)
trainimg = trainimg/255

trainlb = numpy.array(trainlb).reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
trainlb = OneHotEncoder().fit_transform(trainlb).toarray()
print(trainimg.shape)
print(trainlb.shape)

# creating the model
from keras import models,layers
from keras import backend
backend.set_image_data_format("channels_first")

model = models.Sequential()

model.add(layers.Conv2D(filters=20,kernel_size=(5,5),activation='relu',
                       input_shape=(1,100,100)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=40,kernel_size=(5,5),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(80,activation='relu'))
model.add(layers.Dense(4,activation='sigmoid'))# 4 labels

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],
             optimizer='adam')
model.fit(trainimg,trainlb,batch_size=None,epochs=10,verbose=True)


model.save("criminal_detection.h5") # saving the model