from ultralytics import YOLO
import cv2
import pafy
import concurrent.futures
# Define the source
WEBCAM = 0  # Use "samples/v1.mp4" for a video file
VIDEO_FILE = "dataset_cam1.mp4"
YOUTUBE = "http://www.youtube.com/watch?v=q0kPBRIPm6o" # need ssl to be set

# Initialize video capture
# cap = cv2.VideoCapture(WEBCAM)

# video = pafy.new(YOUTUBE)
# best = video.getbest(preftype="mp4")
# cap = cv2.VideoCapture(best.url)

cap = cv2.VideoCapture(VIDEO_FILE)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 1280   # Adjust based on your needs
resize_height = 720  # Adjust based on your needs
if frame_width > 0:
   resize_height = int((resize_width / frame_width) * frame_height)


# Load the YOLO model
chosen_model = YOLO("yolov8n.pt")  # Adjust model version as needed

def predict(chosen_model, img, classes=[], conf=0.5):
   #resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   if classes:
       results = chosen_model.predict(img, classes=classes, conf=conf, save_txt=False)
   else:
       results = chosen_model.predict(img, conf=conf, save_txt=False)

   return results
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
def process_frame(frame):
   result_frame, _ = predict_and_detect(chosen_model, frame)
   return result_frame

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
