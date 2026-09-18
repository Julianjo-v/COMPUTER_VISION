# Raspberry Pi Shape Classifier

A lightweight, real-time computer-vision project that detects coloured objects from a camera feed and classifies their geometric shape. It is designed for a Raspberry Pi Zero 2 W, but the full workflow can also run on a computer with Python and a webcam.

Rather than using a large neural network, the project segments objects by colour, measures their contours, and sends five numeric shape descriptors to a Decision Tree classifier. This keeps the runtime small enough for embedded hardware.

## What the project does

1. Captures a 640 × 480 camera frame.
2. Converts the frame from BGR to HSV and thresholds the saturation channel to isolate coloured objects.
3. Uses morphological opening/closing and connected-component analysis to remove noise and find objects.
4. Calculates contour descriptors: area, perimeter, circularity, compactness, and convexity.
5. Uses the saved Decision Tree model to predict a label and draws that label and a bounding box on the live feed.

```text
Camera → HSV segmentation → clean binary mask → contours
       → shape descriptors → Decision Tree → label + bounding box
```

## Repository layout

```text
Training images/
  Real_images/                             Source images used for the current training set
  Digital_images/                          Additional generated reference images
src/
  Shape classification/
    shape_classification_resize.py         Extracts descriptors at the live-camera resolution
    shape_classification_regular.py        Older, non-resized descriptor extractor
  Training/
    shape_descriptors_project_640x480.csv  Generated descriptor dataset
    decision_tree_640x480.py               Trains and saves the Decision Tree bundle
  shape_descriptor/
    shape_descriptor.py                    Live webcam classifier
    decision_tree_model_final.joblib       Saved model used by the live classifier
```

## Requirements

- Python 3
- A USB webcam or Raspberry Pi-compatible camera for live classification
- Raspberry Pi Zero 2 W is the target hardware; a desktop computer works for development and training

Install the Python packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run the complete workflow

Run these commands from the repository root. Quotation marks are needed because one directory name contains a space.

### 1. Build the training dataset

This reads images from `Training images/Real_images`, resizes each to 640 × 480 to match live inference, and writes descriptors to `src/Training/shape_descriptors_project_640x480.csv`.

```bash
python "src/Shape classification/shape_classification_resize.py"
```

The image filename becomes the class label. The included real-image dataset contains `BEAD`, `CIRCLE`, `DIAMOND`, `KNOB`, and `SQUARE` examples.

### 2. Train the model

This reads the generated CSV and overwrites `src/shape_descriptor/decision_tree_model_final.joblib`, which is the model consumed by the webcam program.

```bash
python src/Training/decision_tree_640x480.py
```

### 3. Start live classification

```bash
python src/shape_descriptor/shape_descriptor.py
```

Press `q` while the camera window is active to exit.

If the camera does not open, change `cv2.VideoCapture(0)` in `src/shape_descriptor/shape_descriptor.py` to the camera index assigned by the operating system, such as `1`.

## Feature descriptors

| Feature | Meaning |
| --- | --- |
| Area | Pixels inside the contour |
| Perimeter | Length of the contour boundary |
| Circularity | How closely the contour resembles a circle |
| Compactness | Relationship between perimeter and area |
| Convexity | Contour perimeter compared with its convex hull |

The training and live scripts use the same feature order and the same 640 × 480 preprocessing settings. The saved model bundle also stores the feature names and label encoder so predictions are converted back to readable shape labels.

## Tuning and limitations

- Objects should have sufficient colour saturation compared with the background; the current saturation threshold is `40`.
- The minimum accepted object area is 0.7% of a 640 × 480 frame, which filters small blobs/noise.
- Predictions are only as good as the small training set and will be sensitive to lighting, object colour, size, angle, and background.
- The Decision Tree does not provide prediction confidence in the current interface.

## Technology

Python, OpenCV, NumPy, Pandas, scikit-learn, and Joblib.

## Author

Julian Joseph Amaro
