from ultralytics import YOLO
import cv2
import pafy
import concurrent.futures

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
       results = chosen_model.predict(img, classes=classes, conf=conf, save_txt=False, verbose=False)
   else:
       results = chosen_model.predict(img, conf=conf, save_txt=False, verbose=False)

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
def process_frame(frame):
   result_frame, _ = predict_and_detect(chosen_model, frame)
   return result_frame


# main
def main():
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
           future = executor.submit(process_frame, frame)
           result_frame = future.result()

           # Display the processed frame
           cv2.imshow("Processed Frame", result_frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()
