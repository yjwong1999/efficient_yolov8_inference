# Efficient YOLOv8 Inference

## Setup
Conda environment
```
conda create --name yolo python=3.8.10 -y
conda activate yolo

git clone https://github.com/yjwong1999/efficient_yolov8_inference.git
cd efficient_yolov8_inference
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

## Install VLC player to simulate rtsp streaming
```
sudo snap install vlc
```

## Detection
Single stream detection
```
python3 single_detect.py --webcam
python3 single_detect.py --camera 0
python3 single_detect.py --video-file sample_video.mp4
python3 single_detect.py --rtsp "rtsp://192.168.1.136:8554/"
python3 single_detect.py --youtube "http://www.youtube.com/watch?v=q0kPBRIPm6o"
```

Multi stream detection
```
python3 multi_detect.py
```

## Tracking
Single stream tracking
```
python3 single_track.py --webcam
python3 single_track.py --camera 0
python3 single_track.py --video-file sample_video.mp4
python3 single_track.py --rtsp "rtsp://192.168.1.136:8554/"
python3 single_track.py --youtube "http://www.youtube.com/watch?v=q0kPBRIPm6o"
```

Multi stream tracking
```
python3 multi_track.py
```

## BUGS
- cannot play youtube yet


## Acknowledgement
1. ultralytics official repo
2. tips for effecient single-stream detection (multithread, resize frame, skipping frame) [[ref]](https://blog.stackademic.com/step-by-step-to-surveillance-innovation-pedestrian-detection-with-yolov8-and-python-opencv-dbada14ca4e9)
3. multi-thread for multi-stream detection [[ref]](https://ultralytics.medium.com/object-tracking-across-multiple-streams-using-ultralytics-yolov8-7934618ddd2)
