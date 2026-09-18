import cv2
import numpy as np
import pandas as pd
import joblib
import os

# Load trained model bundle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "decision_tree_model_final.joblib")

bundle = joblib.load(MODEL_PATH)
clf = bundle["decision_tree"]
label_encoder = bundle["label_encoder"]
feature_names = bundle["feature_names"]

print("Model feature order:", feature_names)

# Initialize webcam
cap = cv2.VideoCapture(0)  # change to 0 if needed to run on computer 1 if on raspberry
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

TOTAL_PIXELS = TARGET_WIDTH * TARGET_HEIGHT
MIN_AREA = int(TOTAL_PIXELS * 0.007)

print("Press 'q' to exit")

cv2.namedWindow("Live Classified Objects", cv2.WINDOW_NORMAL)

# Main loop
while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    output = image.copy()

    # HSV SEGMENTATION (IDENTICAL)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    _, binary = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components (ONLY for masks)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    for i in range(1, num_labels):
        component_mask = (labels == i).astype("uint8") * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = contours[0]

        # CONTOUR AREA FILTER
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        # CONTOUR BOUNDING BOX
        x, y, w, h = cv2.boundingRect(cnt)

        # SHAPE DESCRIPTORS
        perimeter = cv2.arcLength(cnt, True)

        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        compactness = (perimeter ** 2) / area

        hull = cv2.convexHull(cnt)
        hull_perimeter = cv2.arcLength(hull, True)
        convexity = perimeter / hull_perimeter if hull_perimeter > 0 else 0


        # ROUNDING (IDENTICAL)
        area = round(area, 1)
        perimeter = round(perimeter, 1)
        circularity = round(circularity, 3)
        compactness = round(compactness, 2)
        convexity = round(convexity, 3)

        # Build feature row
        descriptor_row = pd.DataFrame(
            [[area, perimeter, circularity, compactness, convexity]],
            columns=feature_names
        )

        # Predict
        prediction = clf.predict(descriptor_row)
        label = label_encoder.inverse_transform(prediction)[0]


        # Draw bounding box + label

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Live Classified Objects", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
