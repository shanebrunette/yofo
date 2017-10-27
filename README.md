# You Only Follow Once

Human Following with the Human Support Robot

## Getting Started

Download the code into a catkin workspace as a new package. Run catkin make on the workspace (if there are errors you may need to alter the name on the CMakeLists and package.xml files). 

This code contains two scripts required for use in the src folder. 

Place the robot in the correct following position by extending the base to full height and lowering the arm:

```
python stand_tall.py
```

Run the simple follower without head movement or face detection. This script takes two arguments. 
* Argument 1 - Face Detection - 0 for off or 1 for on
* Argument 2 - Head Movement - 0 for off or 1 for on 

```
python yofo.py 0 0 
```


Other files in this folder include:
* human.py - a python script which details the human object used for following
* haarcascade_frontalface_default.xml - Code from the Intel Corporation Open Source Computer Vision Library, which is used for Face Detection
 

### Prerequisites

This code runs using ROS Indigo, Python 2.7.6, OpenCV 2.4 and YOLO v2. The topics listed in the code are designed to run on the Toyota Human Support Robot, and will need to be altered if another robot is used. 

### Installing and Setup

ROS Indigo can be installed from [here](http://wiki.ros.org/indigo/Installation)

YOLO from Darknet can be installed from [here](https://pjreddie.com/darknet/install/), the main website for YOLO is [here](https://pjreddie.com/darknet/yolo/)

OpenCV 2.4 for Linux can be installed from [here](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)

Note: The VM provided by UNSW for COMP3431 already has ROS Indigo and OpenCV 2.4 installed and configured. If you are using this VM only install YOLO. 

## Demonstration 

Videos of this code being demonstrated on the Toyota Human Support Robot can be found [here](https://youtu.be/AZw5d_0etys) 

## Authors

* **Shane Brunette** - shanebrunette@gmail.com
* **Alison McCann** - alison.r.mccann@gmail.com

## Acknowledgments

* The YOLO Convolutional Neural Network was used for Object Detection. 
* Code from the Intel Corporation Open Source Computer Vision Library was used for Face Detection
* OpenCV was used for Differentiation based on Color Probability and Image Cropping
