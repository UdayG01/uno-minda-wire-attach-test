import cv2
import numpy as np
import time
from typing import Optional, Tuple

class VideoHandler:
    """
    Handles video input from both recorded files and webcam streams
    Designed for real-time processing with frame rate monitoring
    """
    
    def __init__(self, source: Optional[str] = None, target_fps: int = 30):
        """
        Initialize video handler
        
        Args:
            source: Path to video file (None for webcam)
            target_fps: Target FPS for processing
        """
        self.source = source
        self.target_fps = target_fps
        self.cap = None
        self.is_webcam = source is None
        self.frame_time = 1.0 / target_fps
        
    def initialize(self) -> bool:
        """Initialize video capture"""
        if self.is_webcam:
            self.cap = cv2.VideoCapture(1)  # Default webcam
            print("Initializing webcam...")
        else:
            self.cap = cv2.VideoCapture(self.source)
            print(f"Loading video file: {self.source}")
            
        if not self.cap.isOpened():
            print("Error: Could not open video source")
            return False
            
        # Set webcam properties for better performance
        if self.is_webcam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
        return True
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get next frame from video source"""
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_fps(self) -> float:
        """Get actual FPS of video source"""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self) -> int:
        """Get total frame count (for video files only)"""
        if self.cap is None or self.is_webcam:
            return -1
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def test_video_system():
    """
    Test function to verify video input system
    """
    print("=== Video Input System Test ===")
    
    # Test with webcam first
    print("\n1. Testing webcam input...")
    webcam_handler = VideoHandler()
    
    if webcam_handler.initialize():
        print("Webcam initialized successfully")
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 90:  # Test for ~3 seconds at 30fps
            ret, frame = webcam_handler.get_frame()
            
            if not ret:
                print("Failed to get frame from webcam")
                break
                
            # Display frame with info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Webcam Test - Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Webcam Test', frame)
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            expected_time = frame_count * (1.0 / 30)
            sleep_time = expected_time - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        # Calculate actual FPS
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time
        print(f"Actual FPS achieved: {actual_fps:.2f}")
        
        webcam_handler.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize webcam")
    
    # Test with video file (if available)
    print("\n2. Testing with video file...")
    print("To test with a video file, update VIDEO_PATH below and uncomment the test")
    
    # Uncomment and update path to test with video file
    VIDEO_PATH = "input videos/SampleClip1_2.mp4"
    file_handler = VideoHandler(VIDEO_PATH)
    
    if file_handler.initialize():
        print(f"Video file loaded: {file_handler.get_frame_count()} frames")
        print(f"Video FPS: {file_handler.get_fps()}")
        
        frame_count = 0
        while frame_count < 1600:  # Test first 100 frames
            ret, frame = file_handler.get_frame()
            if not ret:
                break
                
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video File Test', frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):  # ~30fps playback
                break
                
            frame_count += 1
        
        file_handler.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_video_system()