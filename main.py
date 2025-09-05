#!/usr/bin/env python3
"""
main.py
Improved Shape + Color detection using classical OpenCV methods.

Usage:
    python main.py --image test.jpg --output annotated.png --min-area 300 --scale 60

What changed vs the previous version (why):
- We now detect shapes by first segmenting common colors in HSV (red, orange, yellow,
  green, blue, purple) using widened HSV ranges. Color-first segmentation fixes
  mislabelling caused by averaging hue across background pixels.
- For each color mask we apply morphological closing/opening to fill thin shapes
  (important for thin rectangles) so polygon approximation doesn't collapse edges.
- We use convex hull + a smaller approximation epsilon (0.02 * perimeter) which
  reduces accidental merging of vertices (fixes thin rectangles being approximated
  as triangles).
- Circle detection uses a slightly stricter circularity threshold (0.75).
- We also detect black/white shapes using simple V/S thresholds.
- Display resizing remains available via --scale.

This file replaces the previous main.py content.
"""

import cv2
import numpy as np
import argparse
import os


# Define HSV color ranges (H: 0-179 in OpenCV)
COLOR_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (179, 255, 255))],
    "orange": [((11, 70, 50), (20, 255, 255))],
    "yellow": [((21, 70, 50), (40, 255, 255))],
    "green": [((41, 70, 50), (85, 255, 255))],
    "blue": [((86, 70, 50), (125, 255, 255))],
    "purple": [((126, 70, 50), (159, 255, 255))],
}


def preprocess_mask(mask):
    """Apply morphological ops to clean and fill the mask.

    This helps with thin shapes where edges or anti-aliasing create holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def detect_shape_from_contour(cnt):
    """Classify shape from a single contour (expects reasonably clean contour).

    Uses convex hull + approxPolyDP with a smaller epsilon (0.02*peri) to avoid
    merging different corners. For 4-sided shapes we use minAreaRect to handle
    rotated rectangles/squares.
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    if peri == 0:
        return "unidentified", hull

    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    v = len(approx)

    if v == 3:
        return "triangle", approx

    if v == 4:
        rect = cv2.minAreaRect(hull)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            ar = 0
        else:
            ar = max(w, h) / float(min(w, h))
        if 0.95 <= ar <= 1.05:
            return "square", approx
        else:
            return "rectangle", approx

    # use circularity for >4 vertices
    area = cv2.contourArea(hull)
    circularity = 0 if peri == 0 else 4.0 * np.pi * area / (peri * peri)
    if circularity >= 0.75:
        return "circle", approx

    return "unidentified", approx


def annotate_image(img, results):
    out = img.copy()
    for r in results:
        cnt = r["contour"]
        shape = r["shape"]
        color = r["color"]
        cx, cy = r["centroid"]

        cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
        label = f"{color} {shape}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx - 5, cy - th - 5), (cx + tw + 5, cy + 5), (255, 255, 255), -1)
        cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    return out


def find_shapes_by_color(image_bgr, min_area=300):
    """Segment the image by color masks and detect shapes per color.

    Returns a list of dicts: {contour, shape, color, centroid, area}
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    processed_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    results = []

    for color_name, ranges in COLOR_RANGES.items():
        # build combined mask for color (handles multi-range like red)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            m = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, m)

        # remove parts already claimed by previous colors (avoid duplicates)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(processed_mask))

        # clean/fill
        mask = preprocess_mask(mask)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            shape, approx = detect_shape_from_contour(cnt)

            # centroid
            M = cv2.moments(cnt)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                pts = approx.reshape(-1, 2)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

            results.append({
                "contour": cnt,
                "shape": shape,
                "color": color_name,
                "centroid": (cx, cy),
                "area": area,
            })

            # mark processed regions to avoid double-detection
            cv2.drawContours(processed_mask, [cnt], -1, 255, -1)

    # Detect black/white shapes (low saturation / low value)
    v = cv2.split(hsv)[2]
    s = cv2.split(hsv)[1]

    # black: V <= 50
    black_mask = cv2.inRange(v, 0, 50)
    black_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(processed_mask))
    black_mask = preprocess_mask(black_mask)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        shape, approx = detect_shape_from_contour(cnt)
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            pts = approx.reshape(-1, 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        results.append({"contour": cnt, "shape": shape, "color": "black", "centroid": (cx, cy), "area": area})
        cv2.drawContours(processed_mask, [cnt], -1, 255, -1)

    # white: low S and high V
    white_mask = cv2.inRange(s, 0, 50)
    vmask = cv2.inRange(v, 200, 255)
    white_mask = cv2.bitwise_and(white_mask, vmask)
    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(processed_mask))
    white_mask = preprocess_mask(white_mask)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        shape, approx = detect_shape_from_contour(cnt)
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            pts = approx.reshape(-1, 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        results.append({"contour": cnt, "shape": shape, "color": "white", "centroid": (cx, cy), "area": area})
        cv2.drawContours(processed_mask, [cnt], -1, 255, -1)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default="test.jpg", help="path to input image")
    ap.add_argument("-o", "--output", required=False, default="output.png", help="path to save annotated output")
    ap.add_argument("-m", "--min-area", required=False, type=int, default=300, help="minimum contour area to keep")
    ap.add_argument("-s", "--scale", required=False, type=int, default=60, help="display scale percent for the annotated window")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"Input image not found: {args.image}")
        return

    image = cv2.imread(args.image)
    results = find_shapes_by_color(image, min_area=args.min_area)

    annotated = annotate_image(image, results)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to {args.output}")
    print("Detected shapes:")
    for r in results:
        print(f" - {r['shape']:10s} | {r['color']:7s} | area={int(r['area']):6d} | centroid={r['centroid']}")

    # Resize for display
    scale_percent = args.scale
    width = int(annotated.shape[1] * scale_percent / 100)
    height = int(annotated.shape[0] * scale_percent / 100)
    resized = cv2.resize(annotated, (width, height))

    cv2.imshow("Annotated", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
