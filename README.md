# Criminal Face recognition 
in our project we choose facial recognition system to detect the criminals with several objective:
•	Detect specific face from all faces 
•	Highlight suspect face
•	Generate alert based on location
•	Illegal detection immigrants 
•	Run this on server 
We create codes to assist police detect the fugitive criminals and illegal residents by using public cameras to detect the criminals by face recognitions by using algorithm historical crime data and construct patterns based on analysis of particular.  In the several coming point it will explain the technique of using the code.
•	cameras recognize the face in public places even in the streets with real-time video footage and trying to match the people it sees with images on black list.
•	Highlight the faces and identify the person by get information like name age and nationality and generate alert based on location.
•	develop the image by use infrared camera solutions to overcome the issues others face with changes in lighting.

Code 1:
In code-1 we imported the images of different faces  saved in folder and put it in a list to convert it to an array   and reshape it  and scale the image so we prepare the data to be used in training the model we will create, after that we used model.sequential for deep learning and create model layer by layer by using keras package, The model receives black and white 64×64 images as input, then has a sequence of two convolutional and pooling layers as feature extractors, followed by a fully connected layer to interpret the features and an output layer with a sigmoid activation for two-class predictions.
Then we trained the model with the images we prepared previously some images for criminal for example and images for unknown people. Finally, we saved the model (criminal-detection.h5) to use it in code 2.

Code 2:
In code-2 we loaded the model that we created before (criminal-detection.h5) and used it to predict the images by using CascadeClassifier to detect objects in a video stream, we feed using the camera throw function we created to detect faces using haarcascade_frontalface , or the image we provide to test the accuracy of the model.

App.1Py:
In this code we created a server camera using Flask package,  and used the code 2 to predict the faces with the live camera and to show us if the person is criminal or not and draw square around the criminal person and the name of that person or word like” criminal , black list,” and store the name and the time of this person appear on cameras.

Index.html:
In this file we create the web server by using Flask package to open the camera and detect the criminal faces and apply a code to open different cameras of different devices in same page. 

Here is the result of running the code above and using as input the video stream of a build-in webcam:

