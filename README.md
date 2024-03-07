# Efficient YOLOv8 Inference

## Setup
Conda environment
```
conda create --name yolo python=3.8.10 -y
conda activate yolo
```

Install dependencies
```
pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.1.24
pip install pip install pafy==0.5.5
pip install youtube-dl==2021.12.17
```

## Find port number connected to camera
```
python3 find_port.py
```

## Single stream detection
```
python3 single_detect.py --webcam
python3 single_detect.py --camera 0
python3 single_detect.py --video-file dataset_cam1.mp4
python3 single_detect.py --rtsp "rtsp://192.168.1.136:8554/"
python3 single_detect.py --youtube "http://www.youtube.com/watch?v=q0kPBRIPm6o"
```

## BUGS
- cannot play youtube yet


## Acknowledgement
1. ultralytics official repo
2. tips for effecient single-stream detection (multithread, resize frame, skipping frame) [[ref]](https://blog.stackademic.com/step-by-step-to-surveillance-innovation-pedestrian-detection-with-yolov8-and-python-opencv-dbada14ca4e9)
3. multi-thread for multi-stream detection [[ref]](https://ultralytics.medium.com/object-tracking-across-multiple-streams-using-ultralytics-yolov8-7934618ddd2)
