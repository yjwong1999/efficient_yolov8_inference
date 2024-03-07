import cv2 # Import OpenCV Library
from ultralytics import YOLO # Import Ultralytics Package
import concurrent.futures
import threading # Threading module import

# Define the video files for the trackers
video_file1 = "dataset_cam1.mp4" # Video file path
video_file2 = 0  # WebCam Path

# Load the YOLOv8 models
model1 = YOLO("yolov8n.pt") # YOLOv8n Model
model2 = YOLO("yolov8n_face.pt") # YOLOv8s Model

resize_width = 1280   # Adjust based on your needs
resize_height = 720  # Adjust based on your needs

# predict
def predict(chosen_model, img, classes=[], conf=0.5):
   #resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   if classes:
       results = chosen_model.predict(img, classes=classes, conf=conf, save_txt=False)
   else:
       results = chosen_model.predict(img, conf=conf, save_txt=False)

   return results


# predict and detect
def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
   # resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   results = predict(chosen_model, img, classes, conf=conf)

   for result in results:
       for box in result.boxes:
           #if lable is person make the box greeen and print confidence level on the box in huge font
           if result.names[int(box.cls[0])] == "person":
               cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
               cv2.putText(img, f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
           else:
               cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
               cv2.putText(img, f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
   return img, results


# process frame
def process_frame(inputs):
   model, frame = inputs
   result_frame, _ = predict_and_detect(model, frame)
   return result_frame



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

    cap = cv2.VideoCapture(filename)  # Read the video file
    skip_frames = 2  # Number of frames to skip before processing the next one
    frame_count = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
               break
            frame_count = 1+frame_count
            if frame_count % skip_frames != 0:
                continue  # Skip this frame

            # Submit the frame for processing
            future = executor.submit(process_frame, (model, frame,))
            result_frame = future.result()

            # Display the processed frame
            cv2.imshow("Processed Frame", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
	
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
