# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def draw_pose(image, landmarks):
	''' 
	TODO Task 1
	
	Code to this fucntion to draw circles on the landmarks and lines
	connecting the landmarks then return the image.
	
	Use the cv2.line and cv2.circle functions.

	landmarks is a collection of 33 dictionaries with the following keys
		x: float values in the interval of [0.0,1.0]
		y: float values in the interval of [0.0,1.0]
		z: float values in the interval of [0.0,1.0]
		visibility: float values in the interval of [0.0,1.0]
		
	References:
	https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
	https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
	'''

	# copy the image
	landmark_image = image.copy()
	
	# get the dimensions of the image
	height, width, _ = image.shape

	# loop through the landmarks
	for landmark in landmarks.landmark:
		x = int(landmark.x * width)
		y = int(landmark.y * height)
		cv2.circle(landmark_image, (x, y), 5, (0, 255, 0), -1)
	# Draw lines between connected landmarks
	for connection in mp.solutions.pose.POSE_CONNECTIONS:
		start_idx, end_idx = connection
		start = landmarks.landmark[start_idx]
		end = landmarks.landmark[end_idx]
		x1, y1 = int(start.x * width), int(start.y * height)
		x2, y2 = int(end.x * width), int(end.y * height)
		cv2.line(landmark_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
	
	return landmark_image

def main():
	''' 
	TODO Task 2
		modify this fucntion to take a photo uses the pi camera instead 
		of loading an image

	TODO Task 3
		modify this function further to loop and show a video
	'''

	# Create a pose estimation model 
	mp_pose = mp.solutions.pose

	# Start detecting the poses
	with mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as pose:

		# Capture an image from the Pi Camera
		image = pi_camera.capture_array()

		# Convert the image from RGB to BGR (OpenCV uses BGR)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# To improve performance, mark the image as not writeable
		image.flags.writeable = False

		# Get the landmarks
		results = pose.process(image)

		if results.pose_landmarks is not None:
			# Draw the pose landmarks on the image
			result_image = draw_pose(image, results.pose_landmarks)
			cv2.imwrite('output.png', result_image)
			print(results.pose_landmarks)
		else:
			print('No Pose Detected')


if __name__ == "__main__":
	main()
	print('done')
