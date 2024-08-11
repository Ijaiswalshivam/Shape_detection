# Shape Detection with OpenCV
### This Python script detects and labels geometric shapes in images using OpenCV and NumPy. It supports shapes like triangles, squares, rectangles, pentagons, hexagons, stars, and circles.

## Features
### Shape Detection:
Identifies and labels various geometric shapes.
### Contour Approximation: 
Utilizes contour approximation to classify shapes.
### Visualization: 
Displays the processed image with labeled shapes using Matplotlib.
## Requirements
### Ensure you have the following Python packages installed:
opencv-python
numpy
matplotlib

## Usage
### Save the Script: 
Copy the code into a file named shape_detection.py.
### Prepare an Image: 
Place the image you want to analyze in a known directory.
### Run the Script:
Execute the script with Python, specifying the image path.
### View Results: 
The script will display the image with detected shapes labeled.

## How It Works
### Preprocessing: 
Converts the image to grayscale, blurs it, applies adaptive thresholding, and detects edges.
### Shape Detection: 
Finds contours, approximates them to polygons, and classifies shapes based on vertex count and angles.
### Visualization: 
Draws contours and labels detected shapes, then shows the result using Matplotlib.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
