import cv2
import numpy as np
import csv
import os

# ---------------- PATH ----------------
folder_path = "Training images/Real_images"

image_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith((".jpg", ".png"))
]

# ---------------- CSV ----------------
f = open("shape_descriptors_project_2.csv", "w", newline="")
writer = csv.writer(f)

writer.writerow([
    "ID", "Label", "Area", "Perimeter",
    "X", "Y", "Width", "Height",
    "Circularity", "Compactness", "Convexity"
])

# ---------------- PARAMETERS ----------------
SAT_THRESHOLD = 40    # saturation threshold (30–60)
MIN_AREA = 15000      # remove small blobs

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ---------------- PROCESS IMAGES ----------------
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        continue

    output = image.copy()

    # ---------------- HSV SHAPE SEGMENTATION ----------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    _, binary = cv2.threshold(
        s, SAT_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    # Morphology (clean silhouette)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---------------- CONNECTED COMPONENTS ----------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Label from filename
    image_label = os.path.splitext(image_file)[0]

    # ---------------- COMPONENT LOOP ----------------
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        cx, cy = centroids[i]

        component_mask = (labels == i).astype("uint8") * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = contours[0]

        # ---------------- SHAPE DESCRIPTORS ----------------
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter > 0 else 0
        )

        compactness = (
            (perimeter ** 2) / area
            if area > 0 else 0
        )

        hull = cv2.convexHull(cnt)
        hull_perimeter = cv2.arcLength(hull, True)

        convexity = (
            perimeter / hull_perimeter
            if hull_perimeter > 0 else 0
        )

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
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(
            output, f"ID:{i}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

    # Optional preview per image
    # cv2.imshow("Shape Extraction", output)
    # cv2.waitKey(0)

# ---------------- CLEANUP ----------------
f.close()
# cv2.destroyAllWindows()
