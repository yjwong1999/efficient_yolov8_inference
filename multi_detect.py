import cv2 # Import OpenCV Library
from ultralytics import YOLO # Import Ultralytics Package

import threading # Threading module import

# Define the video files for the trackers
video_file1 = "dataset_cam1.mp4" # Video file path
video_file2 = 0  # WebCam Path

# Load the YOLOv8 models
model1 = YOLO("yolov8n.pt") # YOLOv8n Model
model2 = YOLO("yolov8n_face.pt") # YOLOv8s Model

def run_tracker_in_thread(filename, model, file_index):
    """
    This function is designed to run a video file or webcam stream
    concurrently with the YOLOv8 model, utilizing threading.

    - filename: The path to the video file or the webcam/external
    camera source.
    - model: The file path to the YOLOv8 model.
    - file_index: An argument to specify the count of the
    file being processed.
    """

    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        cv2.imshow("Tracking_Stream_"+str(file_index), res_plotted)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release video sources
    video.release()
	
# Create the tracker thread

# Thread used for the video file
tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file1, model1, 1),
                                   daemon=True)

# Thread used for the webcam
tracker_thread2 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file2, model2, 2),
                                   daemon=True)	
								   
# Start the tracker thread

# Start thread that run video file
tracker_thread1.start()

# Start thread that run webcam
tracker_thread2.start()

								   # Wait for the tracker thread to finish

# Tracker thread 1
tracker_thread1.join()

# Tracker thread 2
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
