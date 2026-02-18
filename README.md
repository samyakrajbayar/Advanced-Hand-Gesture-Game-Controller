# 🎮 Advanced Hand Gesture Game Controller

A sophisticated computer vision system that transforms your hand gestures into game controls using MediaPipe, OpenCV, and PyAutoGUI. Turn your webcam into a magical game controller!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **Advanced Gesture Recognition**: Detects 12+ different hand gestures including fist, open hand, pointing (all directions), peace sign, thumbs up/down, and more
- 🎨 **Real-time Visual HUD**: Live feedback with gesture panels, FPS counter, confidence meters, and control guides
- 🎮 **Multiple Game Profiles**: Pre-configured profiles for Temple Run, Subway Surfers, Racing games, Platformers, and FPS games
- ⚙️ **Customizable Configuration**: JSON-based configuration system for easy customization
- 🔧 **Calibration System**: Built-in calibration for accurate hand size detection
- 🛡️ **Gesture Smoothing**: Intelligent debouncing prevents accidental inputs
- ⏸️ **Pause/Resume**: Quick pause functionality without closing the app
- 📊 **Performance Stats**: Real-time FPS monitoring and detection statistics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/hand-gesture-controller.git
cd hand-gesture-controller
```

2. **Run the setup script**:
```bash
python setup.py
```

Or manually install dependencies:
```bash
pip install -r requirements.txt
```

3. **Launch the controller**:
```bash
# Default profile (Temple Run)
python hand_controller.py

# With specific profile
python hand_controller.py --profile subway_surfers

# List all profiles
python hand_controller.py --list
```

## 🎮 Supported Gestures

| Gesture | Description | Icon |
|---------|-------------|------|
| ✊ Fist | Closed hand, all fingers curled | ✊ |
| ✋ Open Hand | All fingers extended | ✋ |
| ☝️ Point Up | Index finger up | ☝️ |
| 👇 Point Down | Index finger down | 👇 |
| 👈 Point Left | Index finger left | 👈 |
| 👉 Point Right | Index finger right | 👉 |
| 👍 Thumb Up | Thumb extended upward | 👍 |
| 👎 Thumb Down | Thumb extended downward | 👎 |
| ✌️ Peace Sign | Index and middle fingers | ✌️ |
| 🤘 Rock On | Index and pinky fingers | 🤘 |
| 👌 OK Sign | Thumb and index touching | 👌 |
| 🖕 Flip Off | All except thumb | 🖕 |

## 📋 Available Profiles

### Temple Run
```bash
python hand_controller.py --profile temple_run
```
- ✊ **Fist** → Slide (Down arrow)
- ✋ **Open Hand** → Jump (Up arrow)
- 👈 **Point Left** → Turn Left (Left arrow)
- 👉 **Point Right** → Turn Right (Right arrow)
- 👍 **Thumb Up** → Pause (P key)

### Subway Surfers
```bash
python hand_controller.py --profile subway_surfers
```
- ✊ **Fist** → Roll (Down arrow)
- ✋ **Open Hand** → Jump (Up arrow)
- 👈 **Point Left** → Move Left (Left arrow)
- 👉 **Point Right** → Move Right (Right arrow)
- ✌️ **Peace Sign** → Hoverboard (Space)

### Racing
```bash
python hand_controller.py --profile racing
```
- 👈 **Point Left** → Steer Left (Left arrow)
- 👉 **Point Right** → Steer Right (Right arrow)
- ✊ **Fist** → Brake (Down arrow)
- ✋ **Open Hand** → Accelerate (Up arrow)
- 👍 **Thumb Up** → Nitro (Space)

### Platformer
```bash
python hand_controller.py --profile platformer
```
- ✋ **Open Hand** → Jump (Space)
- 👈 **Point Left** → Move Left (A)
- 👉 **Point Right** → Move Right (D)
- ✊ **Fist** → Crouch (S)
- ✌️ **Peace Sign** → Attack (J)

### FPS Games
```bash
python hand_controller.py --profile fps
```
- ☝️ **Point Up** → Forward/Up (W)
- 👇 **Point Down** → Backward/Down (S)
- 👈 **Point Left** → Strafe Left (A)
- 👉 **Point Right** → Strafe Right (D)
- ✊ **Fist** → Crouch (Ctrl)
- ✋ **Open Hand** → Reload (R)
- ✌️ **Peace Sign** → Fire (Space)

## 🎯 How It Works

### Architecture

```
┌─────────────────┐
│   Webcam Feed   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  MediaPipe Hands│  ← 21 hand landmarks detection
└────────┬────────┘
         ▼
┌─────────────────┐
│ Gesture Detector│  ← Pattern matching & orientation
└────────┬────────┘
         ▼
┌─────────────────┐
│ Gesture Buffer  │  ← Smoothing & debouncing
└────────┬────────┘
         ▼
┌─────────────────┐
│ Input Simulator │  ← PyAutoGUI key simulation
└────────┬────────┘
         ▼
┌─────────────────┐
│   Game Input    │
└─────────────────┘
```

### Gesture Detection Pipeline

1. **Hand Detection**: MediaPipe Hands detects 21 landmarks per hand
2. **Finger State Calculation**: Determines which fingers are extended
3. **Orientation Analysis**: Calculates pitch, yaw, and dominant direction
4. **Gesture Classification**: Matches patterns to known gestures
5. **Temporal Smoothing**: Applies majority vote over buffer window
6. **Input Execution**: Triggers keyboard events via PyAutoGUI

## ⚙️ Configuration

### Customizing Profiles

Edit `config.json` to customize gestures:

```json
{
  "gestures": {
    "fist": {
      "name": "Slide",
      "key": "down",
      "cooldown": 0.4,
      "hold_time": 0.1,
      "description": "Slide under obstacles"
    }
  }
}
```

**Parameters**:
- `key`: Keyboard key to simulate
- `cooldown`: Minimum time between activations (seconds)
- `hold_time`: Minimum time to hold gesture before triggering (seconds)
- `sensitivity`: Gesture detection sensitivity multiplier
- `smoothing`: Number of frames for majority voting (higher = more stable)

### Creating Custom Profiles

1. Add new profile to `config.json`:
```json
{
  "profiles": {
    "my_game": {
      "name": "My Game",
      "description": "Custom controls",
      "sensitivity": 1.0,
      "smoothing": 5,
      "gestures": {
        "fist": {
          "name": "Action",
          "key": "space",
          "cooldown": 0.5,
          "hold_time": 0.2,
          "description": "Perform action"
        }
      }
    }
  }
}
```

## 🎨 Visual Features

### Heads-Up Display (HUD)

The controller displays:
- **Current Gesture**: Real-time gesture detection
- **Confidence Meter**: Visual bar showing detection confidence
- **FPS Counter**: Frames per second performance metric
- **Control Guide**: List of all gestures and their functions
- **Status Indicator**: Green = Active, Red = Paused

### Debug Information

Press keys during operation:
- `P` - Pause/Resume controller
- `Q` - Quit application
- `C` - Start calibration mode

## 🔧 Calibration

For optimal accuracy:

1. Press `C` to enter calibration mode
2. Show your **OPEN HAND** to the camera
3. Hold for a few seconds to collect samples
4. Press `SPACE` to confirm

This calibrates hand size for better gesture detection across different distances.

## 💡 Tips for Best Performance

### Lighting
- Use consistent, bright lighting
- Avoid backlighting (don't sit with window behind you)
- Minimize shadows on your hand

### Positioning
- Sit 2-3 feet from the camera
- Keep hand within the green detection zone
- Use neutral background for better contrast

### Performance
- Close unnecessary applications
- Lower camera resolution if FPS is low
- Ensure your computer meets minimum requirements

### Gesture Execution
- Make clear, deliberate gestures
- Hold gestures for at least 0.1 seconds
- Avoid rapid successive gestures

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# List available cameras
python setup.py --check

# Try different camera ID
python hand_controller.py --camera 1
```

### Gestures Not Recognized
- Run calibration: Press `C` while running
- Check lighting conditions
- Ensure hand is fully visible
- Try adjusting `sensitivity` in config

### Input Lag
- Reduce `smoothing` value in config
- Close other applications
- Lower camera resolution
- Check CPU usage

### False Positives
- Increase `smoothing` value
- Increase `hold_time` for gestures
- Improve lighting consistency
- Check for background motion

## 🔬 Advanced Features

### Gesture Buffer System
The controller uses a circular buffer for temporal smoothing:
- Collects gestures over N frames (configurable)
- Applies majority voting for stable detection
- Prevents accidental triggers from brief movements

### Cooldown System
Each gesture has independent cooldown:
- Prevents rapid-fire inputs
- Configurable per gesture
- Visual feedback when on cooldown

### Orientation Detection
Advanced hand orientation tracking:
- Pitch and yaw calculation
- Directional pointing detection
- Adaptive to hand rotation

## 📊 Performance Metrics

Typical performance on modern hardware:
- **Detection Rate**: 30 FPS @ 720p
- **Latency**: < 50ms end-to-end
- **CPU Usage**: 15-25% on quad-core processor
- **Memory**: ~200 MB

## 🛠️ Development

### Project Structure
```
hand-gesture-controller/
├── hand_controller.py    # Main controller application
├── setup.py              # Setup and installation script
├── config.json           # Configuration profiles
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── profiles/            # Custom profile storage
```

### Adding New Gestures

To add a new gesture:

1. Add to `GestureType` enum in `hand_controller.py`
2. Implement detection logic in `detect_gesture()` method
3. Add to configuration in `config.json`
4. Update documentation

Example:
```python
def detect_gesture(self, landmarks) -> GestureType:
    # ... existing code ...
    
    elif extended_count == 3:
        if fingers[1] and fingers[2] and fingers[3]:
            return GestureType.THREE_FINGERS
    
    # ... rest of code ...
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional gesture types
- More game profiles
- Performance optimizations
- UI/UX enhancements
- Documentation improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for the hand tracking model
- **OpenCV** for computer vision capabilities
- **PyAutoGUI** for cross-platform input simulation
- Inspired by the original Temple Run gesture controller concept

## 📧 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

**Enjoy gaming with gestures!** 🎮✨

*Computer Vision meets Gaming - Experience the magic of hands-free control!*

**gl**
