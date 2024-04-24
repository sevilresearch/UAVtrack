# Project Title: Development of a Tracking Pipeline Using AirSim and YOLOv7
Overview: This project integrates Unreal Engine 4.27 with AirSim and YOLOv7 to track a moving object autonomously using advanced computer vision techniques. The project's goal is to demonstrate AI's capabilities in identifying and following objects in a simulated environment. The object of interest for this demonstration is a basic blue sedan, defined and trained using Roboflow.

## Getting Started
**Prerequisites** 
  - Unreal Engine 4.27
  - Microsoft Visual Studio 2022
  - PyCharm or any Python IDE
  - Git

## Initial Setup:
### AirSim in Unreal Engine:
Install AirSim by following the guide provided by Microsoft:
[Building AirSim on Windows](https://microsoft.github.io/AirSim/build_windows/).
Create a custom environment in Unreal Engine as described here:
[Custom Environments in AirSim](https://microsoft.github.io/AirSim/unreal_custenv/).
Ensure that the 'AirSim_uproject' and 'AirSim.sln' files are created using Microsoft Visual Studio 2022.

### IDE Setup:
Download and install PyCharm or another Python IDE capable of running Python scripts.
To understand the available plugins and quadrotor functions, familiarize yourself with the function documentation in the official AirSim GitHub repository.

### Plugin Installation:
Install all necessary plugins for Unreal Engine as described in the AirSim documentation.
Test communication between your Python IDE and Unreal Engine using 'AirSimTest1.py'.
YOLOv7 Integration

**Download YOLOv7:**
Follow the instructions on the YOLOv7 GitHub page to download the necessary files:
[YOLOv7 Official GitHub](https://github.com/WongKinYiu/yolov7).
Ensure you download the .weights and .cfg files suitable for YOLOv7.

### Roboflow Dataset Creation:
Capture several images of any object you wish to track; in this case, we use a standard blue sedan model from the Unreal Engine assets.
Use Roboflow to upload and configure your dataset for the object detection model.

### Model Training and Detection:
Run 'detect.py' to check if the quadrotor cameras can accurately draw bounding boxes around the blue sedan.
Debug and run 'ObjectDetectionTest.py' and 'Training_ObjectDetection.py' to refine and compile your dataset.

## Running the Tracker
Execute YOLO_Test.py to validate if the quadrotor successfully follows the target based on the trained model.

## Troubleshooting and Debugging
Ensure all environment paths and dependencies are correctly configured.
Regularly update Unreal Engine and AirSim to their latest versions to avoid compatibility issues.

## Conclusion
This project showcases integrating computer vision tools such as YOLOv7 to simulate real-world AI applications within a controlled environment, offering a scalable model for further research and development in autonomous tracking systems.
