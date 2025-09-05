# task-11.3
this is for mia task 11.3




approach:

-Process the image with traditional image processing pipelines.

-Extract contours that represent object boundaries.

-Approximate and analyze geometric properties to classify shapes.

-Use color thresholding to identify the dominant color of each shape.

algorithims used:

The input image is first converted from BGR → HSV color space. HSV provides better separation of chromatic information (hue) and intensity, making color thresholding more robust to lighting changes.

Color masks are generated using cv2.inRange, isolating specific color ranges.

The masks are cleaned using morphological operations (closing and opening) to remove noise and fill small gaps.

Using cv2.findContours, contours (object boundaries) are extracted from the cleaned binary masks.

Each contour is then analyzed for its perimeter, area, and polygonal approximation.

Polygonal approximation (cv2.approxPolyDP) is applied to contours to reduce them to simpler geometric forms.

Decision rules:

3 vertices → triangle

4 vertices → square or rectangle (aspect ratio check)

>4 vertices → check circularity to classify as circle

Color is determined by the mask from which the contour was extracted.

Multiple HSV ranges were defined for each target color (e.g., red requires two ranges due to hue wrapping).




challenges:

Thin or irregular shapes: Long, thin rectangles sometimes approximated incorrectly as triangles. Adjusting polygon approximation and filtering small contours helped improve results.

Yellow detection: Yellow in HSV can overlap with red/green under different lighting conditions. Narrowing HSV ranges and adjusting morphological cleaning reduced false negatives.

Lighting sensitivity: Classical methods are sensitive to shadows and illumination, requiring careful calibration of thresholds.



insights:
classical computer vision can massivly struggle to complete modern complicated tasks like self driving cars or facial recognition
but it still has its place its very well suited to tasks like this one as it is light weight,doesnt require alot of computing power ,and doesnt require training

note:the final image is included in this repo as output.jpg
