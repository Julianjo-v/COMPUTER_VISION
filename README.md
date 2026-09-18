# Computer Vision Shape Classifier

A real-time computer vision project that uses a webcam to detect objects, extract their geometric features, and classify their shapes using a trained Decision Tree machine learning model.

## Overview

This project combines OpenCV and machine learning to recognize different shapes from a live webcam feed.

The program first processes the webcam image to separate objects from the background. It then detects the objects using connected components and contours. For each detected object, several geometric features are calculated and passed to a trained Decision Tree classifier.

The predicted shape is displayed directly on the webcam image with a bounding box.

## How It Works

The classification process follows these steps:

1. Capture an image from the webcam.
2. Resize the image to 640 × 480 pixels.
3. Convert the image from BGR to HSV color space.
4. Use the saturation channel to create a binary mask.
5. Apply morphological opening and closing to clean the mask.
6. Detect connected components in the binary image.
7. Find the contour of each detected object.
8. Remove objects that are too small.
9. Calculate geometric shape descriptors.
10. Pass the descriptors to the trained Decision Tree model.
11. Convert the model prediction back into the shape name.
12. Display the result with a bounding box and label.

## Features Used by the Model

The Decision Tree uses five features to classify each object:

| Feature     | Description                                           |
| ----------- | ----------------------------------------------------- |
| Area        | The area inside the object's contour                  |
| Perimeter   | The length of the object's contour                    |
| Circularity | Measures how close the object is to a circle          |
| Compactness | Describes the relationship between perimeter and area |
| Convexity   | Compares the object's contour to its convex hull      |

### Circularity

Circularity is calculated using:

```text
Circularity = 4π × Area / Perimeter²
```

A value closer to 1 generally indicates a more circular shape.

### Compactness

Compactness is calculated using:

```text
Compactness = Perimeter² / Area
```

### Convexity

Convexity is calculated by comparing the object's perimeter with the perimeter of its convex hull:

```text
Convexity = Perimeter / Hull Perimeter
```

## Machine Learning Model

The project uses a trained Decision Tree classifier.

The trained model is stored in a `.joblib` file and contains:

* Decision Tree classifier
* Label encoder
* Feature names

The feature names stored with the model are used to make sure the input data is provided to the classifier in the correct order.

## Technologies Used

* Python
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Joblib

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

The main Python libraries used by this project are:

```text
opencv-python
numpy
pandas
scikit-learn
joblib
```

## Running the Program

Run the Python script:

```bash
python shape_classifier.py
```

The program will open the webcam and display the detected objects.

Press:

```text
q
```

to exit the program.

## Webcam Configuration

The current program uses:

```python
cap = cv2.VideoCapture(1)
```

If your computer only has one webcam or the program cannot find the camera, change it to:

```python
cap = cv2.VideoCapture(0)
```

## Example Output

The program displays a bounding box around each detected object and shows the predicted shape above the object.

Example:

```text
+-----------------------+
|                       |
|       OBJECT          |
|                       |
|      "circle"         |
|                       |
+-----------------------+
```

## Project Purpose

The goal of this project is to demonstrate how traditional computer vision techniques can be combined with machine learning for real-time object classification.

Instead of using a deep neural network directly on the image, the system extracts meaningful geometric features from each object's contour and uses those features as the input to a Decision Tree classifier.

## Future Improvements

Possible improvements include:

* Improving detection under different lighting conditions
* Supporting more object classes
* Improving the segmentation process
* Adding confidence information
* Testing with multiple objects at the same time
* Improving the robustness of the classifier
* Adding a graphical user interface
* Collecting a larger and more diverse training dataset

## Author

Created as a computer vision and machine learning project.
