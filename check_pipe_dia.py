# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('img2.jpg')

# def detect_pipe_and_measure_distance(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
#     # Perform edge detection
#     edges = cv2.Canny(blurred, 50, 150)
    
#     # Find contours in the edge-detected image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw contours on the original image
#     cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
#     # Calculate distances between contours and plot them
#     if len(contours) > 1:
#         centers = []
#         for contour in contours:
#             # Get the bounding box for each contour
#             x, y, w, h = cv2.boundingRect(contour)
#             # Calculate the center of the bounding box
#             center_x = x + w // 2
#             center_y = y + h // 2
#             centers.append((center_x, center_y))
        
#         # Calculate distances and draw lines with labels
#         for i in range(len(centers)):
#             for j in range(i + 1, len(centers)):
#                 # Calculate distance between centers
#                 distance = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
                
#                 # Draw a red line between the centers
#                 cv2.line(image, centers[i], centers[j], (0, 0, 255), 2)
                
#                 # Calculate midpoint of the line for placing the text
#                 mid_x = (centers[i][0] + centers[j][0]) // 2
#                 mid_y = (centers[i][1] + centers[j][1]) // 2
                
#                 # Put the distance text near the midpoint
#                 cv2.putText(image, f"{int(distance)}", (mid_x, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     # Display the results
#     cv2.imshow('Detected Contours with Distances', image)
#     cv2.imwrite("out.jpg",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# detect_pipe_and_measure_distance(image)




import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '../Pipe Dimension/Sample Images/Img4.png'  # Change this to your actual image path
image = cv2.imread(image_path)

# Define the real-world scale (e.g., 1 pixel = 0.5 mm)
real_world_scale = 0.5  # Replace this with the actual scale based on calibration

def show_image(img, title="Image"):
    """ Helper function to display an image using Matplotlib """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def detect_pipe_and_measure_distance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    # Display edges
    show_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), title="Edges")

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    # Calculate distances between contours and plot them
    if len(contours) > 1:
        centers = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            diameter_pixels = max(w, h)
            diameter_real = diameter_pixels * real_world_scale  # Convert to real-world units
            
            # Draw diameter label
            cv2.putText(image, f"Diameter: {diameter_real:.2f} mm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        # Calculate distances between detected objects
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance_pixels = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + 
                                          (centers[i][1] - centers[j][1]) ** 2)
                distance_real = distance_pixels * real_world_scale
                
                # Draw red line between detected objects
                cv2.line(image, centers[i], centers[j], (0, 0, 255), 2)
                
                # Display distance label at midpoint
                mid_x = (centers[i][0] + centers[j][0]) // 2
                mid_y = (centers[i][1] + centers[j][1]) // 2
                cv2.putText(image, f"{distance_real:.2f} mm", (mid_x, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show final result with detected pipes and distances
    show_image(image, title="Detected Pipes with Measurements")

    # Save output image
    cv2.imwrite("out.jpg", image)

detect_pipe_and_measure_distance(image)






