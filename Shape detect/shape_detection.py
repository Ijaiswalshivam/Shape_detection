import cv2
import numpy as np
import matplotlib.pyplot as plt

def angle(pt1, pt2, pt0):
    """Calculate the angle between the line pt1-pt0 and pt2-pt0"""
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to open image file {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(binary, 50, 150)
    return edges, image

def detect_shapes(image_path):
    edges, image = preprocess_image(image_path)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if area < 100:  # Filter out small areas
            continue
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        angles = []
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]
            pt0 = approx[(i - 1) % len(approx)][0]
            ang = np.arccos(angle(pt1, pt2, pt0)) * 180 / np.pi
            angles.append(ang)
        
        if len(approx) == 3:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif len(approx) == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif len(approx) == 5:
            if all(108 <= ang <= 112 for ang in angles):
                shape = "Pentagon"
            else:
                shape = "Irregular Pentagon"
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif len(approx) == 6:
            if all(117 <= ang <= 123 for ang in angles):
                shape = "Hexagon"
            else:
                shape = "Irregular Hexagon"
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif len(approx) == 10:
            if all(140 <= ang <= 160 for ang in angles):
                shape = "Star"
            else:
                shape = "Irregular Star"
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif len(approx) > 10:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            cv2.putText(image, 'Circle', (center[0] - radius, center[1] - radius), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, 'Polygon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Shapes")
    plt.axis('off')
    plt.show()


detect_shapes(r'C:\Users\ijais\OneDrive\Desktop\Shape detect\OIP.jpeg')
