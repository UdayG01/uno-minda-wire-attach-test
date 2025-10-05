import cv2
import numpy as np
import time
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from video_handler import VideoHandler

class TerminalBoundaryDrawer:
    """
    Interactive terminal boundary drawing system with JSON configuration support
    """
    
    def __init__(self, pause_after_seconds: float = 2.0, config_file: str = "terminal_config.json"):
        """
        Initialize boundary drawer
        
        Args:
            pause_after_seconds: Time to wait before pausing for boundary drawing
            config_file: Path to JSON file for storing terminal configurations
        """
        self.pause_after_seconds = pause_after_seconds
        self.config_file = config_file
        self.terminals = []  # List of terminal polygons
        self.current_polygon = []  # Points for polygon being drawn
        self.drawing = False
        self.frame_for_drawing = None
        self.drawing_frame = None  # Copy of frame for drawing on
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for interactive boundary drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start new polygon or add point to current
            self.current_polygon.append((x, y))
            print(f"Point added: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current polygon
            if len(self.current_polygon) >= 3:
                self.terminals.append({
                    'points': self.current_polygon.copy(),
                    'color': (0, 165, 255),  # Orange in BGR
                    'status': 'waiting',  # waiting, success, failed
                    'push_count': 0
                })
                print(f"Terminal {len(self.terminals)} registered with {len(self.current_polygon)} points")
                self.current_polygon = []
            else:
                print("Need at least 3 points to create a terminal boundary")
                
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Clear all terminals
            self.terminals = []
            self.current_polygon = []
            print("All terminals cleared")
    
    def save_configuration(self) -> bool:
        """Save terminal configuration to JSON file"""
        try:
            config_data = {
                'terminals': [],
                'timestamp': time.time(),
                'total_terminals': len(self.terminals)
            }
            
            for i, terminal in enumerate(self.terminals):
                terminal_data = {
                    'id': i,
                    'points': terminal['points'],
                    'status': terminal['status'],
                    'push_count': terminal['push_count']
                }
                config_data['terminals'].append(terminal_data)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Configuration saved to {self.config_file}")
            print(f"Saved {len(self.terminals)} terminals")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def load_configuration(self) -> bool:
        """Load terminal configuration from JSON file"""
        try:
            if not os.path.exists(self.config_file):
                print(f"Configuration file {self.config_file} not found")
                return False
            
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.terminals = []
            for terminal_data in config_data['terminals']:
                terminal = {
                    'points': terminal_data['points'],
                    'color': (0, 165, 255),  # Reset to orange
                    'status': 'waiting',  # Reset status
                    'push_count': 0  # Reset push count
                }
                self.terminals.append(terminal)
            
            print(f"Configuration loaded from {self.config_file}")
            print(f"Loaded {len(self.terminals)} terminals")
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def draw_terminals(self, frame: np.ndarray) -> np.ndarray:
        """Draw terminal boundaries on frame"""
        display_frame = frame.copy()
        
        # Draw completed terminals
        for i, terminal in enumerate(self.terminals):
            points = np.array(terminal['points'], np.int32)
            color = terminal['color']
            
            # Draw filled polygon with transparency
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0, display_frame)
            
            # Draw boundary outline
            cv2.polylines(display_frame, [points], True, color, 2)
            
            # Add terminal label
            center_x = int(np.mean([p[0] for p in terminal['points']]))
            center_y = int(np.mean([p[1] for p in terminal['points']]))
            cv2.putText(display_frame, f"T{i+1}", (center_x-10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current polygon being created
        if len(self.current_polygon) > 0:
            for i, point in enumerate(self.current_polygon):
                cv2.circle(display_frame, point, 3, (255, 255, 0), -1)
                if i > 0:
                    cv2.line(display_frame, self.current_polygon[i-1], point, (255, 255, 0), 2)
            
            # Draw line to close polygon preview
            if len(self.current_polygon) > 2:
                cv2.line(display_frame, self.current_polygon[-1], self.current_polygon[0], (128, 128, 0), 1)
        
        return display_frame
    
    def draw_instructions(self, frame: np.ndarray, config_mode: bool = True) -> np.ndarray:
        """Draw instruction text on frame"""
        if config_mode:
            instructions = [
                "CONFIG MODE - Mark terminal boundaries:",
                "Left Click: Add point to terminal boundary",
                "Right Click: Finish current terminal",
                "Middle Click: Clear all terminals", 
                "Press 's' to SAVE config, 'c' to continue"
            ]
        else:
            instructions = [
                "TRACKING MODE - Using saved terminal config",
                f"Loaded {len(self.terminals)} terminals from {self.config_file}"
            ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
        # Show terminal count
        cv2.putText(frame, f"Terminals: {len(self.terminals)}", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def is_point_in_terminal(self, point: Tuple[int, int], terminal_idx: int) -> bool:
        """Check if a point is inside a terminal boundary"""
        if terminal_idx >= len(self.terminals):
            return False
            
        terminal_points = np.array(self.terminals[terminal_idx]['points'], np.int32)
        result = cv2.pointPolygonTest(terminal_points, point, False)
        return result >= 0
    
    def update_terminal_color(self, terminal_idx: int, status: str):
        """Update terminal color based on status"""
        if terminal_idx >= len(self.terminals):
            return
            
        color_map = {
            'waiting': (0, 165, 255),    # Orange
            'success': (0, 255, 0),      # Green  
            'failed': (0, 0, 255)        # Red
        }
        
        self.terminals[terminal_idx]['color'] = color_map.get(status, (0, 165, 255))
        self.terminals[terminal_idx]['status'] = status

def run_boundary_drawing_test(video_source: Optional[str] = None, 
                             config_mode: bool = True, 
                             pause_after: float = 2.0,
                             config_file: str = "terminal_config_test2.json"):
    """
    Test the boundary drawing system with configuration support
    
    Args:
        video_source: Path to video file (None for webcam)
        config_mode: If True, enter drawing mode. If False, load from config
        pause_after: Seconds to wait before pausing for drawing (only in config mode)
        config_file: Path to configuration JSON file
    """
    print("=== Terminal Boundary Drawing Test ===")
    print(f"Config mode: {'ON (drawing new terminals)' if config_mode else 'OFF (using saved config)'}")
    print(f"Config file: {config_file}")
    print("Video source:", "Webcam" if video_source is None else video_source)
    
    # Initialize video handler
    video_handler = VideoHandler(video_source, target_fps=30)
    if not video_handler.initialize():
        print("Failed to initialize video")
        return

        # Print actual resolution being used
    if video_handler.is_webcam:
        width = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📐 Using webcam resolution: {width}x{height}")
    
    # Initialize boundary drawer
    boundary_drawer = TerminalBoundaryDrawer(pause_after, config_file)
    
    # Load existing configuration if not in config mode
    if not config_mode:
        if not boundary_drawer.load_configuration():
            print("Failed to load configuration. Switching to config mode.")
            config_mode = True
    
    # Setup mouse callback only in config mode
    if config_mode:
        cv2.namedWindow('Terminal Boundary Drawing', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Terminal Boundary Drawing', boundary_drawer.mouse_callback)
    else:
        cv2.namedWindow('Terminal Boundary Drawing', cv2.WINDOW_NORMAL)

    start_time = time.time()
    drawing_mode = False
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0
    paused_frame = None
    video_playing = True  # Track video playback state
    
    if config_mode:
        print(f"\nVideo started. Drawing mode will activate after {pause_after} seconds...")
        print("Or press 'd' to enter drawing mode manually")
    else:
        print(f"\nVideo started with {len(boundary_drawer.terminals)} loaded terminals")
        print("Press 'q' to quit")
    
    while True:
        # Only get new frame if video is playing (not paused)
        if video_playing:
            ret, frame = video_handler.get_frame()
            if not ret:
                if video_source is not None:  # End of video file
                    if not drawing_mode:
                        print("End of video reached")
                        break
                    else:
                        # If in drawing mode, just use the paused frame
                        frame = paused_frame if paused_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    print("Failed to get webcam frame")
                    continue
                    
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Auto-pause after specified time (only in config mode)
            if config_mode and not drawing_mode and elapsed_time >= pause_after:
                drawing_mode = True
                video_playing = False  # PAUSE VIDEO
                paused_frame = frame.copy()
                boundary_drawer.frame_for_drawing = frame.copy()
                print(f"\n*** VIDEO PAUSED - DRAWING MODE ACTIVATED after {pause_after} seconds ***")
                print("Start marking terminal boundaries...")
                print("Video playback is PAUSED until you press 'c' to continue")
        else:
            # Video is paused - use stored frame
            frame = paused_frame if paused_frame is not None else boundary_drawer.frame_for_drawing
        
        # Create display frame
        if drawing_mode and config_mode:
            if boundary_drawer.frame_for_drawing is not None:
                display_frame = boundary_drawer.frame_for_drawing.copy()
            else:
                display_frame = frame.copy()
            
            # Draw terminals and instructions
            display_frame = boundary_drawer.draw_terminals(display_frame)
            display_frame = boundary_drawer.draw_instructions(display_frame, config_mode)
            
            # Add drawing mode indicator
            cv2.putText(display_frame, "VIDEO PAUSED - CONFIG MODE", 
                       (10, display_frame.shape[0] - 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 's' to SAVE config, 'c' to resume", 
                       (10, display_frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'r' to reset terminals", 
                       (10, display_frame.shape[0] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            display_frame = frame.copy()
            
            # Draw loaded terminals (even in non-config mode)
            if len(boundary_drawer.terminals) > 0:
                display_frame = boundary_drawer.draw_terminals(display_frame)
            
            display_frame = boundary_drawer.draw_instructions(display_frame, config_mode)
            
            if config_mode and not drawing_mode:
                # Show countdown
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_time = pause_after - elapsed_time
                
                if remaining_time > 0:
                    cv2.putText(display_frame, f"Config mode in: {remaining_time:.1f}s", 
                               (10, display_frame.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Press 'd' to enter config mode now", 
                               (10, display_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate and display FPS only when video is playing
            if video_playing:
                frame_count += 1
                if frame_count % 30 == 0:  # Update every 30 frames
                    fps_elapsed = time.time() - fps_start_time
                    current_fps = 30 / fps_elapsed
                    fps_start_time = time.time()
        
        # Display FPS
        if frame_count > 30:
            fps_color = (255, 255, 255) if video_playing else (128, 128, 128)
            status_text = "PLAYING" if video_playing else "PAUSED"
            cv2.putText(display_frame, f"FPS: {current_fps:.1f} [{status_text}]", 
                       (display_frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        cv2.imshow('Terminal Boundary Drawing', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') and config_mode:
            if not drawing_mode:
                drawing_mode = True
                video_playing = False  # PAUSE VIDEO
                paused_frame = frame.copy()
                boundary_drawer.frame_for_drawing = frame.copy()
                print("\n*** VIDEO PAUSED - MANUAL CONFIG MODE ACTIVATED ***")
                print("Video playback is PAUSED until you press 'c' to continue")
        elif key == ord('s') and drawing_mode and config_mode:
            # Save configuration
            if len(boundary_drawer.terminals) > 0:
                boundary_drawer.save_configuration()
            else:
                print("No terminals to save")
        elif key == ord('c') and drawing_mode and config_mode:
            if len(boundary_drawer.terminals) > 0:
                print(f"\nDrawing complete! {len(boundary_drawer.terminals)} terminals registered")
                print("*** VIDEO PLAYBACK RESUMED ***")
                drawing_mode = False
                video_playing = True  # RESUME VIDEO
                paused_frame = None
                # Don't reset timer to prevent immediate re-pause
            else:
                print("Please mark at least one terminal before continuing")
        elif key == ord('r') and drawing_mode and config_mode:
            boundary_drawer.terminals = []
            boundary_drawer.current_polygon = []
            print("Drawing reset - all terminals cleared")
    
    # Cleanup
    video_handler.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\n=== Session Summary ===")
    print(f"Config mode: {'ON' if config_mode else 'OFF'}")
    print(f"Total terminals: {len(boundary_drawer.terminals)}")
    if config_mode:
        for i, terminal in enumerate(boundary_drawer.terminals):
            print(f"Terminal {i+1}: {len(terminal['points'])} points")
    
    return boundary_drawer.terminals

if __name__ == "__main__":
    # Test with different configurations
    print("Choose test mode:")
    print("1. Config mode ON - Draw new terminals (webcam)")
    print("2. Config mode OFF - Use saved terminals (webcam)")
    print("3. Config mode ON - Draw new terminals (video file)")
    print("4. Config mode OFF - Use saved terminals (video file)")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        terminals = run_boundary_drawing_test(None, config_mode=True, pause_after=2.0)
    elif choice == "2":
        terminals = run_boundary_drawing_test(None, config_mode=False)
    elif choice == "3":
        VIDEO_PATH = "input videos/SampleClip1_2.mp4"  # Update this path
        terminals = run_boundary_drawing_test(VIDEO_PATH, config_mode=True, pause_after=2.0)
    elif choice == "4":
        VIDEO_PATH = "input videos/SampleClip1_2.mp4"  # Update this path
        terminals = run_boundary_drawing_test(VIDEO_PATH, config_mode=False)
    else:
        print("Running default: Config mode ON with webcam")
        terminals = run_boundary_drawing_test(None, config_mode=True, pause_after=2.0)