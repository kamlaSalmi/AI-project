import cv2
import matplotlib.image as im
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('criminal_detection.h5') # importing the trained model

labels = ['subject01','subject02',"subject03","unknown"]

fd = cv2.CascadeClassifier(r"haarcascade_frontalface_alt.xml")

    
"""
path = "datasets\main folder\c1\subject01.wink.jpg"
img = im.imread(path)
output = load_image2(img)
print(output)
"""
# getting the face 

def get_face(img):
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None
    else:
        (x,y,w,h) = corners[0]
        img = img[x:x+h,y:y+w] # cropping the image
        img = cv2.resize(img,(100,100))
    return (x,y,w,h),img
# using the model for predecting     
def img_prde (img):
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corner,img2 = get_face(img2)
    if corner!=None:
        (x,y,w,h)=corner
        output = model.predict_classes(img2.reshape(1,1,100,100))
        person = labels[output[0]]
        if person!="unknown":
            cv2.putText(img,person,(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
    return img
# this step to stop running the code 2 if the app1.py working 
if __name__=="__main__":
    vid = cv2.VideoCapture(0)
    while True:
        ret,img = vid.read()
        if ret==True:
            img = img_prde(img) # predecting the img using the function
            cv2.imshow("img",img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows()
