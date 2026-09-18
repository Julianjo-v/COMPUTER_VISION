# Raspberry Pi Zero 2 W Computer Vision Shape Classifier

A real-time computer vision project designed primarily for the **Raspberry Pi Zero 2 W**. The system uses a camera to detect objects, extract geometric features from their contours, and classify their shapes using a trained Decision Tree machine learning model.

The project was designed to demonstrate how computer vision and machine learning can be implemented on a small, low-power embedded computer.

Although the main target is the Raspberry Pi Zero 2 W, the program can also be run on a regular computer with Python and a compatible camera.

## Project Overview

The goal of this project was to create a **small and lightweight shape recognition system** that could run on a Raspberry Pi Zero 2 W.

The Pi Zero 2 W captures images from a camera and processes them using OpenCV. Instead of using a large neural network, the system extracts a small set of geometric features from each detected object.

These features are then provided to a trained Decision Tree classifier, which predicts the shape of the object.

The predicted shape is displayed on the live camera feed together with a bounding box around the detected object.

## Hardware

### Main Hardware

* **Raspberry Pi Zero 2 W**
* Camera / USB webcam
* MicroSD card
* Power supply

The Raspberry Pi Zero 2 W is the primary platform for this project.

Its small size and low power requirements make it suitable for creating compact embedded computer vision applications.

## System Pipeline

The computer vision system follows this process:

```text
                 Camera
                    ↓
          Raspberry Pi Zero 2 W
                    ↓
             Capture Image
                    ↓
              BGR → HSV
                    ↓
         Saturation Threshold
                    ↓
        Morphological Filtering
                    ↓
         Connected Components
                    ↓
           Contour Detection
                    ↓
        Extract Shape Features
                    ↓
          Decision Tree Model
                    ↓
          Predicted Shape
                    ↓
       Bounding Box + Label
```

## How It Works

### 1. Camera Capture

The Raspberry Pi Zero 2 W captures frames from a connected camera.

The program uses a resolution of:

```text
640 × 480 pixels
```

The camera index can be changed depending on the camera being used:

```python
cap = cv2.VideoCapture(1)
```

If the camera is detected as device `0`, it can be changed to:

```python
cap = cv2.VideoCapture(0)
```

### 2. HSV Image Segmentation

Each camera frame is converted from BGR to HSV color space.

The saturation channel is extracted:

```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]
```

A threshold is then applied to separate the objects from the background:

```python
_, binary = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
```

Morphological operations are used to clean up the resulting binary image.

### 3. Object Detection

Connected components are used to identify individual objects.

Small components are removed using an area threshold. This helps prevent small areas of noise from being classified as objects.

Contours are then extracted from the remaining components.

### 4. Shape Feature Extraction

For each detected object, the program calculates five geometric features:

| Feature     | Purpose                                               |
| ----------- | ----------------------------------------------------- |
| Area        | Measures the size of the object                       |
| Perimeter   | Measures the length of the object's contour           |
| Circularity | Describes how circular the object is                  |
| Compactness | Describes the relationship between perimeter and area |
| Convexity   | Compares the contour to its convex hull               |

These measurements provide the machine learning model with information about the object's shape.

### 5. Decision Tree Classification

The five extracted features are passed to a trained Decision Tree model:

```text
Area
Perimeter
Circularity
Compactness
Convexity
       ↓
Decision Tree
       ↓
Shape Prediction
```

The trained model is loaded using Joblib.

The model bundle contains:

* Decision Tree classifier
* Label encoder
* Feature names

The label encoder converts the model's numerical prediction back into the name of the detected shape.

### 6. Displaying the Result

Once the shape has been classified, the program displays the result on the camera image.

A bounding box is drawn around the object and the predicted shape is displayed above it.

Example:

```text
          circle
       ┌─────────┐
       │         │
       │    ●    │
       │         │
       └─────────┘
```

## Why the Raspberry Pi Zero 2 W?

The Raspberry Pi Zero 2 W was chosen as the main platform because it provides a small and low-power computer capable of running Python and OpenCV.

The project also demonstrates an important embedded systems concept: **designing a computer vision application that can operate with limited hardware resources**.

Rather than processing images using a computationally expensive deep-learning model, this project uses:

* Image segmentation
* Contour detection
* Geometric shape descriptors
* A lightweight Decision Tree classifier

This approach reduces the amount of computation required on the Raspberry Pi Zero 2 W.

## Running on a Regular Computer

The project is primarily intended for the Raspberry Pi Zero 2 W, but the same Python program can also be tested on a normal computer.

A computer can be useful for:

* Developing the program
* Training the Decision Tree
* Testing the computer vision pipeline
* Collecting training data
* Debugging the system

Once the system is working, the program can be transferred to the Raspberry Pi Zero 2 W.

## Technologies Used

### Hardware

* Raspberry Pi Zero 2 W
* Camera / USB webcam

### Software

* Python
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Joblib

## Project Structure

```text
computer-vision-shape-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── shape_classifier.py
│
├── models/
│   └── decision_tree_model_FIXED.joblib
│
└── images/
    └── example_detection.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/computer-vision-shape-classifier.git
```

Enter the project folder:

```bash
cd computer-vision-shape-classifier
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Requirements

```text
opencv-python
numpy
pandas
scikit-learn
joblib
```

## Running the Program

Run the classifier:

```bash
python shape_classifier.py
```

The camera window will open and the system will begin detecting and classifying objects.

Press:

```text
q
```

to exit the program.

## Future Improvements

Possible improvements to the project include:

* Optimizing OpenCV processing specifically for the Pi Zero 2 W
* Improving detection under different lighting conditions
* Adding additional shape classes
* Supporting multiple objects simultaneously
* Improving the training dataset
* Adding prediction confidence
* Using the Raspberry Pi Camera Module
* Connecting the classifier to a robotic object-sorting system
* Measuring processing speed and FPS on the Pi Zero 2 W
* Reducing memory and CPU usage

## Project Goal

The main goal of this project was to build a **real-time computer vision shape classifier on a Raspberry Pi Zero 2 W**.

The project demonstrates how a small embedded computer can combine a camera, image processing, and machine learning to recognize physical objects in real time.

The system uses traditional computer vision to extract meaningful shape information and a lightweight Decision Tree to classify the detected objects.

## Author

Raspberry Pi Zero 2 W Computer Vision and Machine Learning Project.
