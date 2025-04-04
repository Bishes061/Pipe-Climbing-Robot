# Pipe Climbing Robot with Pipe Diameter Detection using OpenCV

## 🧠 Overview

This project focuses on developing a **pipe climbing robot** capable of navigating vertically placed cylindrical pipes. The core feature of the project involves **measuring the pipe diameter using computer vision techniques**. The system detects pipe contours from camera input and converts pixel width to real-world dimensions for precise diameter estimation.

## 🎯 Objectives

- Design and build a robot that can climb vertical pipes.
- Implement a reliable algorithm to calculate pipe diameter from image data.
- Integrate computer vision capabilities with the robot's navigation or control system.

## 🧰 Tech Stack

- **Programming Language:** Python
- **Libraries/Frameworks:** OpenCV, NumPy
- **Hardware:** Pipe climbing robot, camera module
- **Platform:** Raspberry Pi / Arduino (optional, based on implementation)

## 📷 Diameter Measurement Technique

The pipe diameter is calculated using image processing as follows:

1. **Image Capture:** A camera mounted on the robot captures pipe images.
2. **Preprocessing:**
   - Convert to grayscale.
   - Apply Gaussian Blur.
   - Use edge detection (Canny).
3. **Contour Detection:**
   - Detect the outer boundary of the pipe.
   - Use bounding box or ellipse fitting to estimate shape.
4. **Pixel to Real-World Conversion:**
   - Use a reference object of known size or a pre-calibrated scale.
   - Convert the pixel width of the pipe to real-world diameter.

> 📏 Accuracy depends on the calibration and resolution of the camera.

## 🚀 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Bishes061/pipe-climbing-robot.git
   cd pipe-climbing-robot
