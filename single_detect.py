from ultralytics import YOLO
import cv2
import numpy as np
import pafy
import concurrent.futures

from collections import defaultdict

import argparse

# get input argument
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', action='store_true', help='use webcam')               # webcam usually is 0
parser.add_argument('--camera', type=int, default=None, help='camera port number')    # you can find it using find_port.py
parser.add_argument('--video-file', type=str, default=None, help='video filenames')   # example: "dataset_cam1.mp4"
parser.add_argument('--rtsp', type=str, default=None, help='rtsp link')               # example: "rtsp://192.168.1.136:8554/"
parser.add_argument('--youtube', type=str, default=None, help='youtube link')         # example: "http://www.youtube.com/watch?v=q0kPBRIPm6o"
opt = parser.parse_args()

# Define the source
WEBCAM = opt.webcam
CAMERA = opt.camera
VIDEO_FILE = opt.video_file
RTSP = opt.rtsp
YOUTUBE = opt.youtube # need ssl to be set


# load video source
if WEBCAM:
   cap = cv2.VideoCapture(0) # usually webcam is 0
elif CAMERA is not None: 
   cap = cv2.VideoCapture(CAMERA)
elif VIDEO_FILE:
   cap = cv2.VideoCapture(VIDEO_FILE)
elif RTSP:
   cap = cv2.VideoCapture(RTSP)
elif YOUTUBE:
   video = pafy.new(YOUTUBE)
   best = video.getbest(preftype="mp4")
   cap = cv2.VideoCapture(best.url)   
else:
   assert False, "You do not specificy input video source!"


# resize your input video frame size (smaller -> faster, but less accurate)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 1280   # Adjust based on your needs
resize_height = 720  # Adjust based on your needs
if frame_width > 0:
   resize_height = int((resize_width / frame_width) * frame_height)


# Load the YOLO model
chosen_model = YOLO("yolov8n_face.pt")  # Adjust model version as needed


# predict
def predict(chosen_model, img, classes=[], conf=0.5):
   #resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   if classes:
       results = chosen_model.track(img, classes=classes, conf=conf, save_txt=False, persist=True)
   else:
       results = chosen_model.track(img, conf=conf, save_txt=False, persist=True)

   return results


# predict and detect
def predict_and_detect(chosen_model, track_history, img, classes=[], conf=0.5):
   # resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   results = predict(chosen_model, img, classes, conf=conf)

   # Get the boxes and track IDs
   boxes = results[0].boxes.xywh.cpu()
   try:
      track_ids = results[0].boxes.id.int().cpu().tolist()
   except:
      return img, results

   # visualize
   annotated_frame = results[0].plot()
   for box, track_id in zip(boxes, track_ids):
      x, y, w, h = box
      track = track_history[track_id]
      track.append((float(x), float(y)))  # x, y center point
      if len(track) > 30:  # retain 90 tracks for 90 frames
         track.pop(0)

      # Draw the tracking lines
      # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
      # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)


   return annotated_frame, results


# process frame
def process_frame(track_history, frame):
   result_frame, _ = predict_and_detect(chosen_model, track_history, frame)
   return result_frame


# main
def main():
   skip_frames = 2  # Number of frames to skip before processing the next one
   frame_count = 0

   # Store the track history
   track_history = defaultdict(lambda: [])

   with concurrent.futures.ThreadPoolExecutor() as executor:
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           frame_count = 1+frame_count
           if frame_count % skip_frames != 0:
               continue  # Skip this frame

           # Submit the frame for processing
           future = executor.submit(process_frame, track_history, frame)
           result_frame = future.result()

           # Display the processed frame
           cv2.imshow("Processed Frame", result_frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()
