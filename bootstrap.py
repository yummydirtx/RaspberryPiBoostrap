# Imports
import cv2
import torch
from picamera2 import Picamera2
import time
import os

# load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='helmet.pt')

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def detect_objects(image):
	'''
	Dont change this function
	'''
	temp_path = 'temp.png'
	
	cv2.imwrite(temp_path, image)
	imgs = [temp_path]

	# Inference
	results = model(imgs)
	df = results.pandas().xyxy[0]
	# print(df)
	detected_objects = []
	for index, row in df.iterrows():
		p1 = (int(row['xmin']), int(row['ymin']))
		p2 = (int(row['xmax']), int(row['ymax']))
		detected_objects.append((p1, p2, round(row['confidence'] * 100, 2)))

	os.remove(temp_path)
	
	return detected_objects


def draw_on_image(image, objectsDetected, color=(255, 0, 0), thickness=2,
                       fontScale=1, font=cv2.FONT_HERSHEY_SIMPLEX):
	''' 
	TODO Task 1
	
	Code to this fucntion to draw squares to the objects detected then 
	return the image. Use the cv2.rectangle function. Also add text using
	cv2.putText to put the confidences of the object detected.

	objectsDetected is a collection of tuples of a start_point, an 
	end_point and a confidence. start_point and end_point are integers
	and confidence is a float
	
	
	References:
	https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
	'''
	
	for start_point, end_point, confidence in objectsDetected:
		pass

	return image
	

def main():
	''' 
	TODO Task 2
		modify this fucntion to take a photo uses the pi camera instead 
		of loading an image

	TODO Task 3
		modify this function further to loop and show a video
	'''
	# Load the image
	image = cv2.imread("helmet.jpg")

	# Detect Objects
	objectsDetected = detect_objects(image)

	# Draw on the image
	image = draw_on_image(image, objectsDetected)
	
	# Save the output image
	cv2.imwrite('outout.png', image)

	
		
if __name__ == "__main__":
    main()

