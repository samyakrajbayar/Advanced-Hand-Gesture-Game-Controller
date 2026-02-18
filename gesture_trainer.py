"""
Gesture Trainer Utility
======================
Interactive tool to practice and learn hand gestures.

Usage:
    python gesture_trainer.py [--gesture GESTURE]
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque


class GestureTrainer:
    """Interactive gesture training system."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.tip_ids = [4, 8, 12, 16, 20]
        self.pip_ids = [2, 6, 10, 14, 18]
        
        self.target_gesture = None
        self.attempts = 0
        self.successes = 0
        self.start_time = None
        self.gesture_start_time = None
        
        # Performance tracking
        self.detection_history = deque(maxlen=100)
    
    def calculate_finger_states(self, landmarks) -> list:
        """Calculate which fingers are extended."""
        fingers = []
        
        # Thumb
        if landmarks.landmark[self.tip_ids[0]].x < landmarks.landmark[self.pip_ids[0]].x:
            fingers.append(landmarks.landmark[self.tip_ids[0]].x < landmarks.landmark[self.pip_ids[0] - 1].x)
        else:
            fingers.append(landmarks.landmark[self.tip_ids[0]].x > landmarks.landmark[self.pip_ids[0] - 1].x)
        
        # Other fingers
        for id in range(1, 5):
            fingers.append(landmarks.landmark[self.tip_ids[id]].y < landmarks.landmark[self.pip_ids[id]].y)
        
        return fingers
    
    def detect_gesture(self, landmarks) -> str:
        """Detect gesture from landmarks."""
        fingers = self.calculate_finger_states(landmarks)
        extended_count = sum(fingers)
        
        # Calculate orientation
        wrist = landmarks.landmark[0]
        middle_base = landmarks.landmark[9]
        dx = middle_base.x - wrist.x
        dy = middle_base.y - wrist.y
        pitch = np.arctan2(dy, np.sqrt(dx**2 + 0.01)) * 180 / np.pi
        yaw = np.arctan2(dx, np.sqrt(dy**2 + 0.01)) * 180 / np.pi
        
        if abs(pitch) > abs(yaw):
            direction = "up" if pitch < 0 else "down"
        else:
            direction = "right" if dx > 0 else "left"
        
        # Gesture detection
        if extended_count == 0:
            return "fist"
        elif extended_count == 5:
            return "open_hand"
        elif extended_count == 1:
            if fingers[0]:
                return "thumb_up" if pitch < -30 else "thumb_down" if pitch > 30 else "thumb"
            elif fingers[1]:
                return f"point_{direction}"
        elif extended_count == 2:
            if fingers[1] and fingers[2]:
                return "peace_sign"
            elif fingers[1] and fingers[4]:
                return "rock_on"
        elif extended_count == 3:
            if not fingers[1]:
                return "ok_sign"
        elif extended_count == 4:
            if not fingers[0]:
                return "flip_off"
        
        return "unknown"
    
    def draw_training_ui(self, frame, detected_gesture, target_gesture, is_match):
        """Draw training interface."""
        height, width = frame.shape[:2]
        
        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Target gesture
        target_text = f"Target: {target_gesture.replace('_', ' ').title() if target_gesture else 'Free Practice'}"
        cv2.putText(frame, target_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Detected gesture
        detected_text = f"Detected: {detected_gesture.replace('_', ' ').title() if detected_gesture else 'None'}"
        color = (0, 255, 0) if is_match else (0, 165, 255) if detected_gesture else (128, 128, 128)
        cv2.putText(frame, detected_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Statistics
        accuracy = (self.successes / max(1, self.attempts)) * 100
        stats_text = f"Attempts: {self.attempts} | Success: {self.successes} | Accuracy: {accuracy:.1f}%"
        cv2.putText(frame, stats_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Success indicator
        if is_match:
            cv2.circle(frame, (width - 50, 50), 30, (0, 255, 0), -1)
            cv2.circle(frame, (width - 50, 50), 30, (255, 255, 255), 2)
        
        # Progress bar for gesture hold
        if self.gesture_start_time and detected_gesture:
            elapsed = time.time() - self.gesture_start_time
            progress = min(1.0, elapsed / 1.0)
            bar_width = 400
            filled = int(bar_width * progress)
            
            cv2.rectangle(frame, (width//2 - bar_width//2, 140), 
                         (width//2 + bar_width//2, 160), (128, 128, 128), 2)
            cv2.rectangle(frame, (width//2 - bar_width//2, 140), 
                         (width//2 - bar_width//2 + filled, 160), (0, 255, 0), -1)
        
        # Instructions
        instructions = [
            "SPACE: Change target | R: Reset stats | Q: Quit",
            "Hold gesture for 1 second to register success"
        ]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (20, height - 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self, initial_gesture=None):
        """Main training loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.target_gesture = initial_gesture
        self.start_time = time.time()
        
        gestures = ["fist", "open_hand", "point_up", "point_down", 
                   "point_left", "point_right", "thumb_up", "thumb_down",
                   "peace_sign", "rock_on"]
        gesture_idx = gestures.index(initial_gesture) if initial_gesture else 0
        
        last_detected = None
        detected_gesture = None
        is_match = False
        
        print("\n" + "="*60)
        print("🎓 GESTURE TRAINER")
        print("="*60)
        print("\nControls:")
        print("  SPACE - Change target gesture")
        print("  R - Reset statistics")
        print("  Q - Quit trainer")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gesture
                    detected_gesture = self.detect_gesture(landmarks)
                    self.detection_history.append(detected_gesture)
                    
                    # Check if target achieved
                    if self.target_gesture:
                        is_match = detected_gesture == self.target_gesture
                        
                        if is_match:
                            if last_detected != detected_gesture:
                                self.gesture_start_time = time.time()
                            
                            elif time.time() - self.gesture_start_time >= 1.0:
                                self.successes += 1
                                print(f"✅ Success! {self.target_gesture} held for 1 second")
                                self.gesture_start_time = None
                        else:
                            self.gesture_start_time = None
                    
                    last_detected = detected_gesture
                else:
                    detected_gesture = None
                    self.gesture_start_time = None
                
                # Draw UI
                frame = self.draw_training_ui(frame, detected_gesture, self.target_gesture, is_match)
                
                cv2.imshow('Gesture Trainer', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                
                elif key == ord(' '):
                    gesture_idx = (gesture_idx + 1) % len(gestures)
                    self.target_gesture = gestures[gesture_idx]
                    self.attempts += 1
                    self.gesture_start_time = None
                    print(f"🎯 New target: {self.target_gesture}")
                
                elif key == ord('r') or key == ord('R'):
                    self.attempts = 0
                    self.successes = 0
                    print("🔄 Statistics reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            duration = time.time() - self.start_time
            accuracy = (self.successes / max(1, self.attempts)) * 100
            
            print("\n" + "="*60)
            print("📊 TRAINING SUMMARY")
            print("="*60)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Attempts: {self.attempts}")
            print(f"Successes: {self.successes}")
            print(f"Accuracy: {accuracy:.1f}%")
            print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Gesture Training Utility")
    parser.add_argument('--gesture', '-g', type=str, 
                       choices=['fist', 'open_hand', 'point_up', 'point_down',
                               'point_left', 'point_right', 'thumb_up', 'thumb_down',
                               'peace_sign', 'rock_on'],
                       help='Initial gesture to practice')
    
    args = parser.parse_args()
    
    trainer = GestureTrainer()
    trainer.run(initial_gesture=args.gesture)


if __name__ == "__main__":
    main()
