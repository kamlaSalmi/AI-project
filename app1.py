from flask import Flask, render_template,Response
import cv2
from keras.models import load_model
import tensorflow as tf
model = load_model('criminal_detection.h5') 
labels = ['subject01','subject02',"subject03","unknown"]
fd = cv2.CascadeClassifier(r"haarcascade_frontalface_alt.xml")
graph = tf.get_default_graph()
def get_face(img):
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None
    else:
        (x,y,w,h) = corners[0]
        img = img[x:x+h,y:y+w] # cropping the image
        img = cv2.resize(img,(100,100))
    return (x,y,w,h),img
    
def img_prde (img):
    global graph
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corner,img2 = get_face(img2)
    if corner!=None:
        (x,y,w,h)=corner
        with graph.as_default():
            output = model.predict(img2.reshape(1,1,100,100))
        person = labels[int(output[0][0])]
        if person!="unknown":
            cv2.putText(img,person,(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
    return img




app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

def gen():
    vid = cv2.VideoCapture(0)
    while True:
        ret,img = vid.read()
        if ret ==True:
            img = img_prde(img)
            ret,jpeg = cv2.imencode(".jpg",img)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")
    