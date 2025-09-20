# CCTV Real-time Face Recognition Setup Guide

## Overview

This guide explains how to connect your face recognition system to various CCTV camera sources including webcams, IP cameras, RTSP streams, and mobile phone cameras.

## Quick Start

### 1. Basic Webcam
```bash
# Use default webcam (camera 0)
python cctv_realtime.py --source 0

# Use second webcam (camera 1)
python cctv_realtime.py --source 1
```

### 2. IP Camera (HTTP/MJPEG)
```bash
python cctv_realtime.py --source "http://192.168.1.100:8080/video"
```

### 3. RTSP Stream
```bash
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/stream1"
```

## Supported Camera Types

### 1. USB/Built-in Webcams
- **Source**: Integer (0, 1, 2, etc.)
- **Example**: `--source 0`

### 2. IP Cameras (HTTP/MJPEG)
- **Source**: HTTP URL
- **Example**: `--source "http://192.168.1.100:8080/video"`

### 3. RTSP Cameras
- **Source**: RTSP URL with credentials
- **Example**: `--source "rtsp://admin:password@192.168.1.100:554/stream1"`

### 4. Mobile Phone Cameras
- **Apps**: DroidCam, IP Webcam, iVCam
- **Example**: `--source "http://192.168.1.50:4747/video"`

## Common CCTV Camera Configurations

### Hikvision Cameras
```bash
# Main stream (high quality)
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"

# Sub stream (lower quality, better performance)
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/102"
```

### Dahua Cameras
```bash
# Main stream
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"

# Sub stream
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1"
```

### Axis Cameras
```bash
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/axis-media/media.amp"
```

### Foscam Cameras
```bash
python cctv_realtime.py --source "rtsp://admin:password@192.168.1.100:554/videoMain"
```

## Mobile Phone as CCTV Camera

### Option 1: DroidCam (Android/iOS)
1. Install DroidCam app on your phone
2. Connect phone and computer to same WiFi
3. Note the IP address shown in the app
4. Use: `python cctv_realtime.py --source "http://PHONE_IP:4747/video"`

### Option 2: IP Webcam (Android)
1. Install "IP Webcam" app
2. Start the server in the app
3. Use: `python cctv_realtime.py --source "http://PHONE_IP:8080/video"`

### Option 3: iVCam (iOS/Android)
1. Install iVCam on phone and computer
2. Connect both devices
3. Use: `python cctv_realtime.py --source 0` (appears as virtual webcam)

## Command Line Options

### Basic Usage
```bash
python cctv_realtime.py [OPTIONS]
```

### Essential Options
- `--source`: Camera source (webcam index, IP camera URL, or RTSP stream)
- `--confidence-thresh`: Face detection confidence (default: 0.5)
- `--similarity-thresh`: Face recognition similarity (default: 0.4)
- `--max-faces`: Maximum faces to detect per frame (default: 0 = unlimited)
- `--frame-skip`: Skip N frames between processing for performance (default: 0)

### Output Options
- `--save-video`: Save processed video to file
- `--output`: Output video file path (default: cctv_output.mp4)
- `--save-detections`: Save detection results to JSON file
- `--detection-log`: Path to detection log file (default: detections.json)
- `--no-display`: Run without displaying video window (headless mode)

## Performance Optimization

### For High-Resolution Cameras
```bash
python cctv_realtime.py \
  --source "rtsp://admin:password@192.168.1.100:554/stream1" \
  --frame-skip 3 \
  --max-faces 10 \
  --confidence-thresh 0.6
```

### For Multiple Camera Monitoring
```bash
# Use sub-streams and higher frame skip
python cctv_realtime.py \
  --source "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/102" \
  --frame-skip 5 \
  --max-faces 5
```

### For Security Checkpoints (High Accuracy)
```bash
python cctv_realtime.py \
  --source 0 \
  --frame-skip 0 \
  --confidence-thresh 0.3 \
  --similarity-thresh 0.6 \
  --max-faces 3 \
  --save-detections
```

## Troubleshooting

### Connection Issues

1. **"Could not open camera source"**
   - Check if camera is accessible from browser: `http://CAMERA_IP:PORT/`
   - Verify credentials and IP address
   - Try different stream URLs (main vs sub stream)

2. **RTSP Authentication Failed**
   - Verify username and password
   - Check if camera requires specific RTSP port
   - Try without credentials: `rtsp://CAMERA_IP:554/stream1`

3. **Poor Performance**
   - Increase `--frame-skip` value
   - Use sub-stream instead of main stream
   - Reduce `--max-faces` limit
   - Lower camera resolution in camera settings

### Network Configuration

1. **Find Camera IP Address**
   ```bash
   # Scan network for cameras
   nmap -sn 192.168.1.0/24
   
   # Check specific IP
   ping 192.168.1.100
   ```

2. **Test RTSP Stream**
   ```bash
   # Using FFmpeg
   ffplay rtsp://admin:password@192.168.1.100:554/stream1
   
   # Using VLC
   vlc rtsp://admin:password@192.168.1.100:554/stream1
   ```

## Example Scenarios

### 1. Office Entrance Monitoring
```bash
python cctv_realtime.py \
  --source "rtsp://admin:office123@192.168.1.10:554/Streaming/Channels/101" \
  --similarity-thresh 0.5 \
  --save-detections \
  --save-video \
  --output office_entrance.mp4
```

### 2. Home Security with Phone
```bash
python cctv_realtime.py \
  --source "http://192.168.1.50:4747/video" \
  --confidence-thresh 0.4 \
  --frame-skip 2 \
  --save-detections
```

### 3. Multiple Camera Setup (Run multiple instances)
```bash
# Terminal 1 - Front door
python cctv_realtime.py --source "rtsp://admin:pass@192.168.1.10:554/stream1" --detection-log front_door.json

# Terminal 2 - Back door  
python cctv_realtime.py --source "rtsp://admin:pass@192.168.1.11:554/stream1" --detection-log back_door.json

# Terminal 3 - Office
python cctv_realtime.py --source 0 --detection-log office_webcam.json
```

## Integration with Existing Systems

### 1. Home Assistant
Create automation to trigger based on detection log files

### 2. Security Systems
Parse JSON detection logs for alerts and notifications

### 3. Access Control
Use detection results to trigger door locks or access systems

## Advanced Features

### Detection Logging
The system can log all detections to a JSON file:
```json
[
  {
    "timestamp": "2025-09-20T22:45:30.123456",
    "person_name": "ram",
    "similarity_score": 0.756,
    "confidence_score": 0.89,
    "bounding_box": {
      "x1": 100,
      "y1": 150,
      "x2": 200,
      "y2": 250
    }
  }
]
```

### Real-time Alerts
Modify the code to send notifications when specific persons are detected:
- Email alerts
- Webhook notifications  
- Database logging
- SMS notifications

### Multi-threading Architecture
The system uses separate threads for:
- Frame capture (from camera)
- Frame processing (face recognition)
- Display/output (video display and saving)

This ensures smooth real-time processing even with slower cameras or complex processing.

## Security Considerations

1. **Network Security**
   - Use strong passwords for camera access
   - Consider VPN for remote access
   - Regularly update camera firmware

2. **Privacy**
   - Inform people about face recognition in use
   - Consider data retention policies
   - Secure storage of detection logs

3. **Performance Monitoring**
   - Monitor CPU and memory usage
   - Check for dropped frames
   - Regular system maintenance
