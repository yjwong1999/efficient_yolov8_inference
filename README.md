# Efficient YOLOv8 Inferencing using Multithreading

Efficient YOLOv8 inference depends not only on GPU specifications but also on CPU processing. However, the significance of fully utilizing the CPU is often overlooked. In fact, leveraging the CPU is crucial because it plays an essential role in the I/O aspect of model deployment (specifically, reading input frames and plotting the outputs). In this repository, we explore how to utilize CPU multi-threading to enhance inference speed.

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

## If you are doing multi-stream
```
1. List all the sources in source.streams
2. If you are doing tracking + geofencing, list the geofencing roi xyxy in geofencing.streams
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
# Example (without geofencing)
python3 single_track.py --webcam
python3 single_track.py --camera 0
python3 single_track.py --video-file sample_video.mp4
python3 single_track.py --rtsp "rtsp://192.168.1.136:8554/"
python3 single_track.py --youtube "http://www.youtube.com/watch?v=q0kPBRIPm6o"

# Example (with geofencing)
python3 single_track.py -video-file sample_video.mp4 --roi-xyxy 0.6,0.4,0.9,0.8
```

Multi stream tracking
```
# without geofencing
python3 multi_track.py

# with geofencing
python3 multi_track.py --geofencing
```

## TODO
- [ ] cannot play youtube yet
- [ ] drive handling fails for multiple source
- [ ] no error warning when the video source is not available, not sure this will happen for other source types onot
- [ ] the dummy handler in multi_track.py will post() today, should post tmr only

## Citation
```
@software{Wong_Efficient_YOLOv8_Inferencing_2024,
  author = {Wong, Yi Jie},
  doi = {10.5281/zenodo.10792741},
  month = mar,
  title = {{Efficient YOLOv8 Inferencing using Multithreading}},
  url = {https://github.com/yjwong1999/efficient_yolov8_inference},
  version = {1.0.0},
  year = {2024}}
```


## Acknowledgement
1. ultralytics official repo [[ref]](https://github.com/ultralytics/ultralytics)
2. tips for effecient single-stream detection (multithread, resize frame, skipping frame) [[ref]](https://blog.stackademic.com/step-by-step-to-surveillance-innovation-pedestrian-detection-with-yolov8-and-python-opencv-dbada14ca4e9)
3. multi-thread for multi-stream detection [[ref]](https://ultralytics.medium.com/object-tracking-across-multiple-streams-using-ultralytics-yolov8-7934618ddd2)
4. Tracking with Ultralytics YOLO (how to handle the results) [[ref]](https://docs.ultralytics.com/modes/track/#plotting-tracks-over-time)
