import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from ultralytics import YOLO
import torch

from video_handler import VideoHandler
from terminal_boundary_drawer import TerminalBoundaryDrawer

import intel_extension_for_pytorch as ipex

class ConnectorDetector:
    """
    YOLOv8 segmentation-based connector detection system
    Optimized for real-time performance on RTX 3050-class hardware
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 device: str = 'auto', target_classes: List[str] = None):
        """
        Initialize YOLOv8 connector detector
        
        Args:
            model_path: Path to trained YOLOv8n-seg model
            confidence_threshold: Minimum confidence for detections
            device: Device for inference ('cpu', 'cuda', 'xpu', 'auto')
            target_classes: List of class names to detect (e.g., ['connector'])
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ['connector']
        self.device = self._setup_device(device)
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        self.frame_count = 0
        
        # Model initialization
        self.model = None
        self.class_names = {}
        self.target_class_ids = []
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate compute device"""
        if device == 'auto':
            # Check for Intel XPU first, then CUDA, then CPU
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = 'xpu'
                print(f"🚀 Intel XPU available: {torch.xpu.get_device_name()}")
                print(f"   Device count: {torch.xpu.device_count()}")
            elif torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("💻 Using CPU (Neither XPU nor CUDA available)")
        elif device == 'xpu':
            if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
                print("❌ XPU requested but not available. Falling back to CPU.")
                device = 'cpu'
        
        print(f"📱 Inference device: {device}")
        return device
        
    def initialize_model(self) -> bool:
        """
        Initialize YOLOv8 model and validate target classes
        
        Returns:
            bool: True if initialization successful
        """
        try:
            print(f"🔄 Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to appropriate device
            if self.device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
                # For Intel XPU
                self.model.to('xpu')
                print("✅ Model moved to Intel XPU")
            elif self.device == 'cuda' and torch.cuda.is_available():
                # For CUDA
                self.model.to('cuda')
                print("✅ Model moved to CUDA")
            else:
                # Keep on CPU
                print("✅ Model running on CPU")
            
            # Get class names from model
            self.class_names = self.model.names
            print(f"📋 Model classes: {list(self.class_names.values())}")
            
            # Find target class IDs
            self.target_class_ids = []
            for class_id, class_name in self.class_names.items():
                if class_name.lower() in [tc.lower() for tc in self.target_classes]:
                    self.target_class_ids.append(class_id)
            
            if not self.target_class_ids:
                print(f"❌ Target classes {self.target_classes} not found in model")
                return False
            
            print(f"🎯 Target class IDs: {self.target_class_ids}")
            print(f"🎯 Target classes: {[self.class_names[cid] for cid in self.target_class_ids]}")
            
            # Warm up model with dummy inference
            print("🔥 Warming up model...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Move dummy image to device if needed
            if self.device == 'xpu':
                # The YOLO model will handle device placement internally
                pass
            
            _ = self.model(dummy_img, verbose=False)
            print("✅ Model initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
    
    def detect_connectors(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect connector masks in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of connector detections with masks and metadata
        """
        if self.model is None:
            return []
        
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            connectors = []
            if results and len(results) > 0:
                result = results[0]  # Single image inference
                
                # Check if we have detections
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    # Get masks if available
                    masks = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                    
                    # Process each detection
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2, conf, class_id = box
                        class_id = int(class_id)
                        
                        # Filter by target classes
                        if class_id in self.target_class_ids:
                            connector = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'mask': None,
                                'centroid': None,
                                'area': 0
                            }
                            
                            # Process mask if available
                            if masks is not None and i < len(masks):
                                mask = masks[i]
                                
                                # Resize mask to frame dimensions
                                if mask.shape != frame.shape[:2]:
                                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                                
                                connector['mask'] = mask.astype(np.uint8)
                                
                                # Calculate centroid and area
                                moments = cv2.moments(connector['mask'])
                                if moments['m00'] > 0:
                                    cx = int(moments['m10'] / moments['m00'])
                                    cy = int(moments['m01'] / moments['m00'])
                                    connector['centroid'] = (cx, cy)
                                    connector['area'] = moments['m00']
                                else:
                                    # Fallback to bbox center
                                    cx = int((x1 + x2) / 2)
                                    cy = int((y1 + y2) / 2)
                                    connector['centroid'] = (cx, cy)
                                    connector['area'] = (x2 - x1) * (y2 - y1)
                            else:
                                # No mask available, use bbox center
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                connector['centroid'] = (cx, cy)
                                connector['area'] = (x2 - x1) * (y2 - y1)
                            
                            connectors.append(connector)
                            self.total_detections += 1
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 50:  # Keep last 50 measurements
                self.inference_times.pop(0)
            
            self.frame_count += 1
            
            return connectors
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        max_inference_time = max(self.inference_times)
        min_inference_time = min(self.inference_times)
        
        return {
            'avg_inference_fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'max_inference_time_ms': max_inference_time * 1000,
            'min_inference_time_ms': min_inference_time * 1000,
            'total_detections': self.total_detections,
            'frames_processed': self.frame_count,
            'detections_per_frame': self.total_detections / max(self.frame_count, 1)
        }
    
    def draw_detections(self, frame: np.ndarray, connectors: List[Dict[str, Any]], 
                       debug_mode: bool = True) -> np.ndarray:
        """
        Draw connector detections on frame
        
        Args:
            frame: Input frame
            connectors: List of connector detections
            debug_mode: Whether to show detailed debug info
            
        Returns:
            Frame with detections drawn
        """
        display_frame = frame.copy()
        
        for i, connector in enumerate(connectors):
            bbox = connector['bbox']
            centroid = connector['centroid']
            confidence = connector['confidence']
            class_name = connector['class_name']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw mask if available
            if connector['mask'] is not None and debug_mode:
                # Create colored mask overlay
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 1] = connector['mask'] * 255  # Green channel
                
                # Apply with transparency
                mask_overlay = cv2.addWeighted(display_frame, 0.8, mask_colored, 0.2, 0)
                display_frame = mask_overlay
            
            # Draw centroid
            if centroid:
                cv2.circle(display_frame, centroid, 3, (255, 0, 0), -1)
                cv2.circle(display_frame, centroid, 6, (255, 0, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            if debug_mode:
                label += f" | Area: {int(connector['area'])}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw connector ID
            if debug_mode:
                cv2.putText(display_frame, f"C{i+1}", 
                           (centroid[0] - 10, centroid[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame

def test_connector_detection(model_path: str, video_source: Optional[str] = None,
                           config_file: str = "terminal_config.json",
                           debug_mode: bool = True):
    """
    Test YOLOv8 connector detection system
    
    Args:
        model_path: Path to trained YOLOv8 model
        video_source: Video file path (None for webcam)
        config_file: Terminal configuration file
        debug_mode: Show debug visualizations
    """
    print("🔍 YOLOv8 Connector Detection Test")
    print("="*50)
    
    # Initialize components
    print("🚀 Initializing components...")
    
    # Video handler
    video_handler = VideoHandler(video_source, target_fps=30)
    if not video_handler.initialize():
        print("❌ Failed to initialize video")
        return
    
    # Terminal boundary drawer (load existing config)
    boundary_drawer = TerminalBoundaryDrawer(config_file=config_file)
    terminals_loaded = boundary_drawer.load_configuration()
    if not terminals_loaded:
        print("⚠️  No terminal configuration found - running without terminals")
    
    # Connector detector
    detector = ConnectorDetector(model_path, confidence_threshold=0.5, target_classes=['connector'])
    if not detector.initialize_model():
        print("❌ Failed to initialize YOLO model")
        return
    
    print("✅ All components initialized successfully")
    
    # Setup display
    cv2.namedWindow('Connector Detection Test')
    
    # Performance tracking
    frame_count = 0
    fps_start_time = time.time()
    display_fps = 0
    
    print("\n🎥 Starting detection test...")
    print("Controls:")
    print("  'd': Toggle debug mode")
    print("  'q': Quit test")
    
    try:
        while True:
            frame_start_time = time.time()
            
            # Get frame
            ret, frame = video_handler.get_frame()
            if not ret:
                if video_source is not None:
                    print("📹 End of video reached")
                    break
                else:
                    print("❌ Failed to get webcam frame")
                    continue
            
            # Detect connectors
            connectors = detector.detect_connectors(frame)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw terminals if loaded
            if terminals_loaded:
                display_frame = boundary_drawer.draw_terminals(display_frame)
            
            # Draw connector detections
            display_frame = detector.draw_detections(display_frame, connectors, debug_mode)
            
            # Add status information
            cv2.putText(display_frame, "YOLOv8 Connector Detection Test", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Detection stats
            cv2.putText(display_frame, f"Connectors: {len(connectors)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Performance stats
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                display_fps = 30 / elapsed
                fps_start_time = time.time()
            
            if frame_count > 30:
                perf_stats = detector.get_performance_stats()
                cv2.putText(display_frame, f"Display FPS: {display_fps:.1f}", 
                           (10, display_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Inference FPS: {perf_stats.get('avg_inference_fps', 0):.1f}", 
                           (10, display_frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Inference: {perf_stats.get('avg_inference_time_ms', 0):.1f}ms", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Debug mode indicator
            if debug_mode:
                cv2.putText(display_frame, "DEBUG MODE", 
                           (display_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Connector Detection Test', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            # Frame rate limiting
            frame_time = time.time() - frame_start_time
            target_frame_time = 1.0 / 30  # 30 FPS target
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    # Cleanup and final stats
    video_handler.release()
    cv2.destroyAllWindows()
    
    # Print final performance report
    print("\n📊 Final Performance Report")
    print("="*40)
    perf_stats = detector.get_performance_stats()
    for metric, value in perf_stats.items():
        print(f"{metric}: {value:.2f}")
    
    print(f"\nTerminals loaded: {len(boundary_drawer.terminals) if terminals_loaded else 0}")
    print("✅ Detection test completed")

if __name__ == "__main__":
    # Interactive test runner
    print("🔍 YOLOv8 Connector Detection Tester")
    print("="*45)
    
    # Get model path
    # model_path = input("Enter path to your YOLOv8n-seg model (.pt file): ").strip()
    model_path = "model/seg-con2.pt"
    if not model_path:
        print("❌ Model path required")
        exit(1)
    
    # Choose video source
    print("\nChoose video source:")
    print("1. Webcam")
    print("2. Video file")
    
    choice = input("Enter choice (1-2): ").strip()
    
    video_source = None
    if choice == "2":
        video_source = "input videos/SampleClip1_2.mp4"
        if not video_source:
            print("❌ Video file path required")
            exit(1)
    
    # Run test
    test_connector_detection(model_path, video_source)