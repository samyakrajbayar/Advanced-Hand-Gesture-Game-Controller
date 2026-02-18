"""
Advanced Hand Gesture Game Controller
====================================
A sophisticated computer vision system that transforms hand gestures into game controls.
Supports multiple games, customizable gestures, calibration, and real-time visual feedback.

Author: AI Assistant
Version: 2.0.0
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import argparse


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    FIST = "fist"
    OPEN_HAND = "open_hand"
    POINT_UP = "point_up"
    POINT_DOWN = "point_down"
    POINT_LEFT = "point_left"
    POINT_RIGHT = "point_right"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    PEACE_SIGN = "peace_sign"
    ROCK_ON = "rock_on"
    OK_SIGN = "ok_sign"
    FLIP_OFF = "flip_off"
    NONE = "none"


@dataclass
class GestureConfig:
    """Configuration for a single gesture."""
    name: str
    key: str
    cooldown: float
    hold_time: float
    description: str


@dataclass
class ControlProfile:
    """Game-specific control profile."""
    name: str
    description: str
    gestures: Dict[GestureType, GestureConfig]
    sensitivity: float
    smoothing: int


class GestureBuffer:
    """Circular buffer for gesture smoothing and debouncing."""
    
    def __init__(self, size: int = 5):
        self.buffer = deque(maxlen=size)
        self.size = size
    
    def add(self, gesture: GestureType):
        """Add gesture to buffer."""
        self.buffer.append(gesture)
    
    def get_stable_gesture(self) -> GestureType:
        """Get most common gesture in buffer (majority vote)."""
        if len(self.buffer) < self.size // 2:
            return GestureType.NONE
        
        counts = {}
        for g in self.buffer:
            counts[g] = counts.get(g, 0) + 1
        
        max_count = max(counts.values())
        if max_count >= self.size // 2:
            return max(counts, key=counts.get)
        return GestureType.NONE
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class GestureDetector:
    """Advanced hand gesture detection using MediaPipe."""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Finger tip IDs
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.pip_ids = [2, 6, 10, 14, 18]  # PIP joints for comparison
        
        # Calibration data
        self.calibration_data = {
            'hand_size': 1.0,
            'palm_center': None,
            'finger_thresholds': {}
        }
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """Detect hands in frame and return landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0], results.multi_hand_landmarks
        return None, None
    
    def calculate_finger_states(self, landmarks) -> List[bool]:
        """Calculate which fingers are extended."""
        fingers = []
        
        # Thumb (check x position relative to IP joint)
        if landmarks.landmark[self.tip_ids[0]].x < landmarks.landmark[self.pip_ids[0]].x:
            fingers.append(landmarks.landmark[self.tip_ids[0]].x < landmarks.landmark[self.pip_ids[0] - 1].x)
        else:
            fingers.append(landmarks.landmark[self.tip_ids[0]].x > landmarks.landmark[self.pip_ids[0] - 1].x)
        
        # Other 4 fingers (check y position - tip above PIP)
        for id in range(1, 5):
            fingers.append(landmarks.landmark[self.tip_ids[id]].y < landmarks.landmark[self.pip_ids[id]].y)
        
        return fingers
    
    def calculate_hand_orientation(self, landmarks) -> Tuple[float, float, str]:
        """Calculate hand orientation (pitch, yaw, dominant direction)."""
        wrist = landmarks.landmark[0]
        middle_finger_base = landmarks.landmark[9]
        index_finger_base = landmarks.landmark[5]
        
        # Calculate angles
        dx = middle_finger_base.x - wrist.x
        dy = middle_finger_base.y - wrist.y
        
        pitch = np.arctan2(dy, np.sqrt(dx**2 + 0.01)) * 180 / np.pi
        yaw = np.arctan2(dx, np.sqrt(dy**2 + 0.01)) * 180 / np.pi
        
        # Determine dominant direction
        if abs(pitch) > abs(yaw):
            direction = "up" if pitch < 0 else "down"
        else:
            direction = "right" if dx > 0 else "left"
        
        return pitch, yaw, direction
    
    def detect_gesture(self, landmarks) -> GestureType:
        """Detect gesture from hand landmarks."""
        fingers = self.calculate_finger_states(landmarks)
        pitch, yaw, direction = self.calculate_hand_orientation(landmarks)
        
        # Count extended fingers
        extended_count = sum(fingers)
        
        # Gesture detection logic
        if extended_count == 0:
            return GestureType.FIST
        
        elif extended_count == 5:
            return GestureType.OPEN_HAND
        
        elif extended_count == 1:
            if fingers[0]:  # Only thumb
                if pitch < -30:
                    return GestureType.THUMB_UP
                elif pitch > 30:
                    return GestureType.THUMB_DOWN
            elif fingers[1]:  # Only index
                if direction == "up":
                    return GestureType.POINT_UP
                elif direction == "down":
                    return GestureType.POINT_DOWN
                elif direction == "left":
                    return GestureType.POINT_LEFT
                elif direction == "right":
                    return GestureType.POINT_RIGHT
        
        elif extended_count == 2:
            if fingers[1] and fingers[2]:  # Index and middle
                return GestureType.PEACE_SIGN
            elif fingers[1] and fingers[4]:  # Index and pinky
                return GestureType.ROCK_ON
        
        elif extended_count == 3:
            if not fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                return GestureType.OK_SIGN
        
        elif extended_count == 4:
            if not fingers[0]:  # All except thumb
                return GestureType.FLIP_OFF
        
        return GestureType.NONE
    
    def draw_landmarks(self, frame: np.ndarray, landmarks):
        """Draw hand landmarks on frame."""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )


class InputSimulator:
    """Simulates keyboard inputs with cooldown and hold detection."""
    
    def __init__(self):
        self.last_press_time: Dict[str, float] = {}
        self.key_hold_start: Dict[str, float] = {}
        self.active_keys: set = set()
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
    
    def press_key(self, key: str, cooldown: float = 0.3) -> bool:
        """Press a key if cooldown has elapsed."""
        current_time = time.time()
        
        if key in self.last_press_time:
            if current_time - self.last_press_time[key] < cooldown:
                return False
        
        pyautogui.press(key)
        self.last_press_time[key] = current_time
        return True
    
    def hold_key(self, key: str, hold_duration: float = 0.5) -> bool:
        """Hold a key for specified duration."""
        current_time = time.time()
        
        if key not in self.key_hold_start:
            self.key_hold_start[key] = current_time
            pyautogui.keyDown(key)
            return True
        
        elif current_time - self.key_hold_start[key] >= hold_duration:
            pyautogui.keyUp(key)
            del self.key_hold_start[key]
            return False
        
        return True
    
    def release_all(self):
        """Release all held keys."""
        for key in list(self.key_hold_start.keys()):
            pyautogui.keyUp(key)
        self.key_hold_start.clear()


class VisualHUD:
    """Visual overlay system for real-time feedback."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.colors = {
            'primary': (0, 255, 0),
            'secondary': (255, 0, 0),
            'accent': (0, 255, 255),
            'warning': (0, 165, 255),
            'error': (0, 0, 255),
            'text': (255, 255, 255),
            'background': (0, 0, 0)
        }
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
    
    def update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get average FPS."""
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def draw_gesture_panel(self, frame: np.ndarray, current_gesture: GestureType, 
                          gesture_config: Optional[GestureConfig], confidence: float):
        """Draw gesture information panel."""
        # Draw panel background
        panel_x = 10
        panel_y = 10
        panel_width = 300
        panel_height = 120
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['background'], -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['primary'], 2)
        
        # Draw gesture name
        gesture_text = f"Gesture: {current_gesture.value}"
        cv2.putText(frame, gesture_text, (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Draw mapped key
        if gesture_config:
            key_text = f"Mapped Key: {gesture_config.key}"
            cv2.putText(frame, key_text, (panel_x + 10, panel_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = panel_x + 10
        bar_y = panel_y + 80
        
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     self.colors['background'], -1)
        filled_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + filled_width, bar_y + bar_height),
                     self.colors['primary'], -1)
        
        # Draw confidence text
        conf_text = f"{confidence*100:.1f}%"
        cv2.putText(frame, conf_text, (bar_x + bar_width + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def draw_fps_counter(self, frame: np.ndarray):
        """Draw FPS counter."""
        fps = self.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        
        # Position in top right
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x = self.width - text_size[0] - 10
        y = 30
        
        cv2.putText(frame, fps_text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['accent'], 2)
    
    def draw_control_guide(self, frame: np.ndarray, profile: ControlProfile):
        """Draw control guide overlay."""
        guide_x = self.width - 320
        guide_y = 60
        line_height = 25
        
        # Draw background
        cv2.rectangle(frame, (guide_x - 10, guide_y - 30),
                     (self.width - 10, guide_y + len(profile.gestures) * line_height + 20),
                     self.colors['background'], -1)
        cv2.rectangle(frame, (guide_x - 10, guide_y - 30),
                     (self.width - 10, guide_y + len(profile.gestures) * line_height + 20),
                     self.colors['secondary'], 2)
        
        # Title
        cv2.putText(frame, f"Controls: {profile.name}", (guide_x, guide_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # List gestures
        y_offset = guide_y + 15
        for gesture_type, config in profile.gestures.items():
            text = f"{gesture_type.value}: {config.key} ({config.description})"
            cv2.putText(frame, text, (guide_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += line_height
    
    def draw_status_indicator(self, frame: np.ndarray, is_active: bool):
        """Draw system status indicator."""
        center = (self.width // 2, 30)
        radius = 10
        color = self.colors['primary'] if is_active else self.colors['error']
        
        cv2.circle(frame, center, radius, color, -1)
        cv2.circle(frame, center, radius + 2, self.colors['text'], 2)
        
        status_text = "ACTIVE" if is_active else "PAUSED"
        cv2.putText(frame, status_text, (center[0] - 30, center[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class GameController:
    """Main controller class that orchestrates gesture detection and input simulation."""
    
    def __init__(self, profile: ControlProfile, camera_id: int = 0):
        self.profile = profile
        self.camera_id = camera_id
        
        # Initialize components
        self.detector = GestureDetector()
        self.input_sim = InputSimulator()
        self.hud = VisualHUD()
        self.gesture_buffer = GestureBuffer(size=self.profile.smoothing)
        
        # State tracking
        self.current_gesture = GestureType.NONE
        self.previous_gesture = GestureType.NONE
        self.gesture_start_time = 0
        self.is_running = False
        self.is_paused = False
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        
        # Camera setup
        self.cap = None
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera initialized successfully")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        self.frame_count += 1
        self.hud.update_fps()
        
        # Detect hands
        landmarks, all_landmarks = self.detector.detect(frame)
        
        if landmarks:
            self.detection_count += 1
            
            # Draw landmarks
            self.detector.draw_landmarks(frame, landmarks)
            
            # Detect gesture
            raw_gesture = self.detector.detect_gesture(landmarks)
            self.gesture_buffer.add(raw_gesture)
            
            # Get stable gesture
            stable_gesture = self.gesture_buffer.get_stable_gesture()
            
            # Handle gesture change
            if stable_gesture != self.previous_gesture:
                self.gesture_start_time = time.time()
                self.previous_gesture = stable_gesture
            
            self.current_gesture = stable_gesture
            
            # Execute action if gesture is configured and not paused
            if not self.is_paused and stable_gesture in self.profile.gestures:
                config = self.profile.gestures[stable_gesture]
                hold_duration = time.time() - self.gesture_start_time
                
                if hold_duration >= config.hold_time:
                    success = self.input_sim.press_key(config.key, config.cooldown)
                    if success:
                        # Visual feedback for key press
                        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
            
            # Draw HUD
            gesture_config = self.profile.gestures.get(self.current_gesture)
            confidence = min(1.0, self.detection_count / max(1, self.frame_count))
            self.hud.draw_gesture_panel(frame, self.current_gesture, gesture_config, confidence)
        else:
            # No hand detected
            self.gesture_buffer.clear()
            self.current_gesture = GestureType.NONE
            
            # Draw "No Hand Detected" message
            cv2.putText(frame, "No Hand Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw common HUD elements
        self.hud.draw_fps_counter(frame)
        self.hud.draw_control_guide(frame, self.profile)
        self.hud.draw_status_indicator(frame, not self.is_paused)
        
        # Draw instructions
        cv2.putText(frame, "Press 'P' to pause/resume | 'Q' to quit | 'C' to calibrate",
                   (10, self.hud.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main execution loop."""
        print("\n" + "="*60)
        print("🎮 Advanced Hand Gesture Game Controller")
        print("="*60)
        print(f"Profile: {self.profile.name}")
        print(f"Description: {self.profile.description}")
        print("\nControls:")
        print("  P - Pause/Resume")
        print("  Q - Quit")
        print("  C - Calibrate")
        print("="*60 + "\n")
        
        if not self.initialize_camera():
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error: Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Hand Gesture Controller', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Shutting down...")
                    break
                
                elif key == ord('p') or key == ord('P'):
                    self.is_paused = not self.is_paused
                    status = "PAUSED" if self.is_paused else "RESUMED"
                    print(f"⏯️  Controller {status}")
                    
                    if self.is_paused:
                        self.input_sim.release_all()
                
                elif key == ord('c') or key == ord('C'):
                    print("🔧 Starting calibration...")
                    self.calibrate()
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        finally:
            self.shutdown()
    
    def calibrate(self):
        """Run calibration routine."""
        print("\n" + "-"*40)
        print("🔧 CALIBRATION MODE")
        print("-"*40)
        print("Show your OPEN HAND to the camera")
        print("Press SPACE when ready (or Q to skip)")
        print("-"*40 + "\n")
        
        calibration_samples = []
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks, _ = self.detector.detect(frame)
            
            if landmarks:
                self.detector.draw_landmarks(frame, landmarks)
                
                # Calculate hand size (distance from wrist to middle finger tip)
                wrist = landmarks.landmark[0]
                middle_tip = landmarks.landmark[12]
                hand_size = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
                calibration_samples.append(hand_size)
                
                # Show calibration progress
                cv2.putText(frame, f"Samples: {len(calibration_samples)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to confirm", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Show your hand!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Hand Gesture Controller', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(calibration_samples) > 10:
                # Calculate average hand size
                avg_size = np.mean(calibration_samples)
                self.detector.calibration_data['hand_size'] = avg_size
                print(f"✅ Calibration complete! Hand size: {avg_size:.4f}")
                break
            
            elif key == ord('q') or key == ord('Q'):
                print("⏭️  Calibration skipped")
                break
    
    def shutdown(self):
        """Clean shutdown."""
        self.is_running = False
        self.input_sim.release_all()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Successful detections: {self.detection_count}")
        print(f"Detection rate: {100*self.detection_count/max(1,self.frame_count):.1f}%")
        print("="*60)


def load_profiles() -> Dict[str, ControlProfile]:
    """Load predefined control profiles."""
    
    # Temple Run Profile
    temple_run = ControlProfile(
        name="Temple Run",
        description="Control Temple Run with hand gestures",
        gestures={
            GestureType.FIST: GestureConfig(
                name="Slide",
                key="down",
                cooldown=0.4,
                hold_time=0.1,
                description="Slide under obstacles"
            ),
            GestureType.OPEN_HAND: GestureConfig(
                name="Jump",
                key="up",
                cooldown=0.4,
                hold_time=0.1,
                description="Jump over obstacles"
            ),
            GestureType.POINT_LEFT: GestureConfig(
                name="Turn Left",
                key="left",
                cooldown=0.5,
                hold_time=0.2,
                description="Turn left at corners"
            ),
            GestureType.POINT_RIGHT: GestureConfig(
                name="Turn Right",
                key="right",
                cooldown=0.5,
                hold_time=0.2,
                description="Turn right at corners"
            ),
            GestureType.THUMB_UP: GestureConfig(
                name="Pause",
                key="p",
                cooldown=1.0,
                hold_time=0.5,
                description="Pause game"
            )
        },
        sensitivity=1.0,
        smoothing=5
    )
    
    # Subway Surfers Profile
    subway_surfers = ControlProfile(
        name="Subway Surfers",
        description="Control Subway Surfers with hand gestures",
        gestures={
            GestureType.FIST: GestureConfig(
                name="Roll",
                key="down",
                cooldown=0.4,
                hold_time=0.1,
                description="Roll under barriers"
            ),
            GestureType.OPEN_HAND: GestureConfig(
                name="Jump",
                key="up",
                cooldown=0.4,
                hold_time=0.1,
                description="Jump over trains"
            ),
            GestureType.POINT_LEFT: GestureConfig(
                name="Move Left",
                key="left",
                cooldown=0.3,
                hold_time=0.1,
                description="Move to left lane"
            ),
            GestureType.POINT_RIGHT: GestureConfig(
                name="Move Right",
                key="right",
                cooldown=0.3,
                hold_time=0.1,
                description="Move to right lane"
            ),
            GestureType.PEACE_SIGN: GestureConfig(
                name="Hoverboard",
                key="space",
                cooldown=2.0,
                hold_time=0.3,
                description="Activate hoverboard"
            )
        },
        sensitivity=1.2,
        smoothing=4
    )
    
    # Racing Game Profile
    racing = ControlProfile(
        name="Racing",
        description="Control racing games with hand steering",
        gestures={
            GestureType.POINT_LEFT: GestureConfig(
                name="Steer Left",
                key="left",
                cooldown=0.1,
                hold_time=0.0,
                description="Steer left"
            ),
            GestureType.POINT_RIGHT: GestureConfig(
                name="Steer Right",
                key="right",
                cooldown=0.1,
                hold_time=0.0,
                description="Steer right"
            ),
            GestureType.FIST: GestureConfig(
                name="Brake",
                key="down",
                cooldown=0.1,
                hold_time=0.0,
                description="Apply brakes"
            ),
            GestureType.OPEN_HAND: GestureConfig(
                name="Accelerate",
                key="up",
                cooldown=0.1,
                hold_time=0.0,
                description="Accelerate"
            ),
            GestureType.THUMB_UP: GestureConfig(
                name="Nitro",
                key="space",
                cooldown=5.0,
                hold_time=0.5,
                description="Use nitro boost"
            )
        },
        sensitivity=1.5,
        smoothing=3
    )
    
    # Platformer Profile
    platformer = ControlProfile(
        name="Platformer",
        description="Control platformer games",
        gestures={
            GestureType.OPEN_HAND: GestureConfig(
                name="Jump",
                key="space",
                cooldown=0.3,
                hold_time=0.1,
                description="Jump"
            ),
            GestureType.POINT_LEFT: GestureConfig(
                name="Move Left",
                key="a",
                cooldown=0.1,
                hold_time=0.0,
                description="Move left"
            ),
            GestureType.POINT_RIGHT: GestureConfig(
                name="Move Right",
                key="d",
                cooldown=0.1,
                hold_time=0.0,
                description="Move right"
            ),
            GestureType.FIST: GestureConfig(
                name="Crouch",
                key="s",
                cooldown=0.1,
                hold_time=0.0,
                description="Crouch/duck"
            ),
            GestureType.PEACE_SIGN: GestureConfig(
                name="Attack",
                key="j",
                cooldown=0.5,
                hold_time=0.1,
                description="Attack/action"
            )
        },
        sensitivity=1.0,
        smoothing=5
    )
    
    return {
        "temple_run": temple_run,
        "subway_surfers": subway_surfers,
        "racing": racing,
        "platformer": platformer
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Hand Gesture Game Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Profiles:
  temple_run    - Optimized for Temple Run (Jump, Slide, Turn)
  subway_surfers - Optimized for Subway Surfers
  racing        - Steering wheel style controls for racing games
  platformer    - Platformer game controls

Examples:
  python hand_controller.py --profile temple_run
  python hand_controller.py --profile racing --camera 1
  python hand_controller.py --list
        """
    )
    
    parser.add_argument(
        '--profile', '-p',
        type=str,
        default='temple_run',
        help='Game profile to use (default: temple_run)'
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available profiles'
    )
    
    args = parser.parse_args()
    
    # Load profiles
    profiles = load_profiles()
    
    if args.list:
        print("\nAvailable Profiles:")
        print("-" * 60)
        for key, profile in profiles.items():
            print(f"\n{key}:")
            print(f"  Name: {profile.name}")
            print(f"  Description: {profile.description}")
            print(f"  Gestures: {len(profile.gestures)}")
        print("\n" + "-" * 60)
        return
    
    if args.profile not in profiles:
        print(f"❌ Error: Unknown profile '{args.profile}'")
        print(f"Use --list to see available profiles")
        return
    
    # Create and run controller
    profile = profiles[args.profile]
    controller = GameController(profile, camera_id=args.camera)
    controller.run()


if __name__ == "__main__":
    main()
