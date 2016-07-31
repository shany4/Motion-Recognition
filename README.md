# Motion-Recognition

-This is an exercise I made for motion recognition based on OpenCV

-The motion recognition system is based on the Lucas-Kanade optical flow method. The Lucas-Kanade optical flow method (LK) is a method for optical flow estimation that analysis every single frame of a motion graphic, and compute different value of x, y coordinates and time, in order to find the vector of each pixel in the graphic. The method comes up with three assumptions. (1) Lighting is constant; (2) movement is tiny; (3) speed and distance are same for each set of pixels in two frames. The system will able to display movement direction of object, which is captured by webcam and distinguish movement to the left and movement to the right. 

-Compile: g++ -lopencv_core -lopencv_highgui -lopencv_imgproc MotionRecognition.cpp
(Make sure you have built openCV already)
