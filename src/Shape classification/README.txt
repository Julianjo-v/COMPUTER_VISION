For better performance in the raspberry pi use the resize

-=- Image Preprocessing: Loads color images, resizes them to 640x480,
    converts them to the HSV color space, and isolates the Saturation channel
    to convert them into clean, black-and-white binary masks using thresholding
    and morphological filtering.
-=- Object Segmentation: Detects distinct shapes/objects in the binary image using
    connected component analysis and contours, filtering out small background noise.
-=- Feature Extraction: Calculates geometric properties for each detected object,
    including Area, Perimeter, Bounding Box $(X, Y, W, H)$, Circularity, Compactness,
    and Convexity.
-=- Data Export: Saves these extracted numerical metrics alongside object labels into a
    standardized CSV file (shape_descriptors_project_640x480.csv) for machine learning models.

