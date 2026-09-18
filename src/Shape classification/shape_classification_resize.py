# =======================================================================
# This script processes a folder of training images to extract geometric
# shape descriptors and writes them to a standardized CSV file
# (shape_descriptors_project_640x480.csv) for machine learning.
# =======================================================================

import cv2
import numpy as np
import csv
import os
from pathlib import Path

# Paths are based on this file, so the script can be started from any folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
folder_path = PROJECT_ROOT / "Training images" / "Real_images"
OUTPUT_CSV = PROJECT_ROOT / "src" / "Training" / "shape_descriptors_project_640x480.csv"
image_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith((".jpg", ".png"))
]

# ---------------- CSV ----------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "ID", "Label",
        "Area", "Perimeter",
        "X", "Y", "Width", "Height",
        "Circularity", "Compactness", "Convexity"
    ])

    # ---------------- PARAMETERS ----------------
    SAT_THRESHOLD = 40
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    TOTAL_PIXELS = TARGET_WIDTH * TARGET_HEIGHT
    MIN_AREA = int(TOTAL_PIXELS * 0.007)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ---------------- PROCESS IMAGES ----------------
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
        output = image.copy()

        # ---------------- HSV SEGMENTATION ----------------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]

        _, binary = cv2.threshold(
            s, SAT_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ---------------- CONNECTED COMPONENTS ----------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        image_label = os.path.splitext(image_file)[0]

        # ---------------- COMPONENT LOOP ----------------
        for i in range(1, num_labels):
            _, _, _, _, area_cc = stats[i]

            if area_cc < MIN_AREA:
                continue

            component_mask = (labels == i).astype("uint8") * 255
            contours, _ = cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            cnt = contours[0]

            # ---------------- CONTOUR BOUNDING BOX ----------------
            x, y, w, h = cv2.boundingRect(cnt)

            # ---------------- SHAPE DESCRIPTORS ----------------
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if area <= 0 or perimeter <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            compactness = (perimeter ** 2) / area

            hull = cv2.convexHull(cnt)
            hull_perimeter = cv2.arcLength(hull, True)
            convexity = perimeter / hull_perimeter if hull_perimeter > 0 else 0

            # ---------------- WRITE CSV ----------------
            writer.writerow([
                i, image_label,
                f"{area:.1f}", f"{perimeter:.1f}",
                x, y, w, h,
                f"{circularity:.3f}",
                f"{compactness:.2f}",
                f"{convexity:.3f}"
            ])

            # ---------------- DRAW (OPTIONAL) ----------------
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)

        # Optional preview
        # cv2.imshow("Shape Extraction", output)
        # cv2.waitKey(0)

# cv2.destroyAllWindows()
