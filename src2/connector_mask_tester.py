import cv2
import numpy as np
import time
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from video_handler import VideoHandler
from terminal_boundary_drawer import TerminalBoundaryDrawer
from connector_detector import ConnectorDetector

@dataclass
class DetectionMetrics:
    """Metrics for evaluating detection quality"""
    total_frames: int = 0
    frames_with_detections: int = 0
    total_detections: int = 0
    avg_confidence: float = 0.0
    avg_detections_per_frame: float = 0.0
    detection_consistency: float = 0.0  # Consistency across frames
    avg_mask_area: float = 0.0
    centroid_stability: float = 0.0

class ConnectorMaskTester:
    """
    Comprehensive testing system for connector mask detection
    Validates detection quality, consistency, and performance
    """
    
    def __init__(self, model_path: str, config_file: str = "terminal_config.json"):
        """
        Initialize the connector mask tester
        
        Args:
            model_path: Path to YOLOv8n-seg model
            config_file: Terminal configuration file
        """
        self.model_path = model_path
        self.config_file = config_file
        
        # Components
        self.detector = None
        self.boundary_drawer = None
        self.video_handler = None
        
        # Testing metrics
        self.metrics = DetectionMetrics()
        self.detection_history = []  # Store detection data over time
        self.confidence_history = []
        self.centroid_history = defaultdict(list)  # Track centroid movement
        
        # Test results
        self.test_results = {
            'webcam_test': {'passed': False, 'metrics': {}},
            'video_test': {'passed': False, 'metrics': {}},
            'consistency_test': {'passed': False, 'metrics': {}},
            'performance_test': {'passed': False, 'metrics': {}}
        }
    
    def initialize_components(self) -> bool:
        """Initialize all required components"""
        try:
            print("🔄 Initializing components...")
            
            # Initialize detector
            self.detector = ConnectorDetector(
                model_path=self.model_path,
                confidence_threshold=0.4,  # Lower threshold for testing
                target_classes=['connector']
            )
            
            if not self.detector.initialize_model():
                print("❌ Failed to initialize detector")
                return False
            
            # Initialize boundary drawer
            self.boundary_drawer = TerminalBoundaryDrawer(config_file=self.config_file)
            
            print("✅ Components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all metrics for new test"""
        self.metrics = DetectionMetrics()
        self.detection_history = []
        self.confidence_history = []
        self.centroid_history = defaultdict(list)
    
    def update_metrics(self, connectors: List[Dict[str, Any]], frame_idx: int):
        """Update metrics based on current frame detections"""
        self.metrics.total_frames += 1
        
        if connectors:
            self.metrics.frames_with_detections += 1
            self.metrics.total_detections += len(connectors)
            
            # Confidence tracking
            frame_confidences = [c['confidence'] for c in connectors]
            self.confidence_history.extend(frame_confidences)
            
            # Centroid tracking for stability analysis
            for i, connector in enumerate(connectors):
                if connector['centroid']:
                    self.centroid_history[i].append(connector['centroid'])
            
            # Mask area tracking
            mask_areas = [c['area'] for c in connectors if c['area'] > 0]
            if mask_areas:
                avg_area = sum(mask_areas) / len(mask_areas)
                if self.metrics.avg_mask_area == 0:
                    self.metrics.avg_mask_area = avg_area
                else:
                    self.metrics.avg_mask_area = (self.metrics.avg_mask_area + avg_area) / 2
        
        # Store detection count for this frame
        self.detection_history.append(len(connectors))
    
    def calculate_final_metrics(self):
        """Calculate final metrics after test completion"""
        if self.metrics.total_frames == 0:
            return
        
        # Detection rate
        self.metrics.avg_detections_per_frame = (
            self.metrics.total_detections / self.metrics.total_frames
        )
        
        # Average confidence
        if self.confidence_history:
            self.metrics.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Detection consistency (lower std dev = more consistent)
        if len(self.detection_history) > 1:
            mean_detections = sum(self.detection_history) / len(self.detection_history)
            variance = sum((x - mean_detections) ** 2 for x in self.detection_history) / len(self.detection_history)
            std_dev = variance ** 0.5
            # Normalize consistency score (0-100, higher = more consistent)
            self.metrics.detection_consistency = max(0, 100 - (std_dev * 20))
        
        # Centroid stability
        stability_scores = []
        for connector_id, centroids in self.centroid_history.items():
            if len(centroids) > 1:
                # Calculate movement variance
                x_coords = [c[0] for c in centroids]
                y_coords = [c[1] for c in centroids]
                x_var = np.var(x_coords)
                y_var = np.var(y_coords)
                movement_var = (x_var + y_var) / 2
                # Lower variance = higher stability
                stability_scores.append(max(0, 100 - movement_var / 10))
        
        if stability_scores:
            self.metrics.centroid_stability = sum(stability_scores) / len(stability_scores)
    
    def draw_test_overlay(self, frame: np.ndarray, connectors: List[Dict[str, Any]], 
                         test_name: str, frame_idx: int) -> np.ndarray:
        """Draw comprehensive test overlay on frame"""
        display_frame = frame.copy()
        
        # Draw terminals if available
        if self.boundary_drawer and len(self.boundary_drawer.terminals) > 0:
            display_frame = self.boundary_drawer.draw_terminals(display_frame)
        
        # Draw detections
        display_frame = self.detector.draw_detections(display_frame, connectors, debug_mode=True)
        
        # Test information overlay
        overlay_height = 200
        overlay = np.zeros((overlay_height, display_frame.shape[1], 3), dtype=np.uint8)
        overlay[:, :] = (0, 0, 0)  # Black background
        
        # Title
        cv2.putText(overlay, f"STEP 5: {test_name}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current frame stats
        cv2.putText(overlay, f"Frame: {frame_idx}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Connectors detected: {len(connectors)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Confidence scores
        if connectors:
            confidences = [f"{c['confidence']:.2f}" for c in connectors]
            cv2.putText(overlay, f"Confidences: {', '.join(confidences)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Running metrics
        if self.metrics.total_frames > 0:
            detection_rate = (self.metrics.frames_with_detections / self.metrics.total_frames) * 100
            avg_detections = self.metrics.total_detections / self.metrics.total_frames
            
            cv2.putText(overlay, f"Detection rate: {detection_rate:.1f}%", (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay, f"Avg detections/frame: {avg_detections:.1f}", (10, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Performance stats
        perf_stats = self.detector.get_performance_stats()
        if perf_stats:
            inference_fps = perf_stats.get('avg_inference_fps', 0)
            cv2.putText(overlay, f"Inference FPS: {inference_fps:.1f}", (10, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(overlay, "Controls: 'q'=quit, 'p'=pause, 's'=save results", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Combine overlay with frame
        combined = np.vstack([overlay, display_frame])
        
        return combined
    
    def test_webcam_detection(self, duration_seconds: int = 30) -> bool:
        """
        Test connector detection with webcam for specified duration
        
        Args:
            duration_seconds: How long to run the test
            
        Returns:
            bool: True if test passed
        """
        print(f"\n🎥 WEBCAM DETECTION TEST ({duration_seconds}s)")
        print("="*50)
        
        # Initialize video
        self.video_handler = VideoHandler(source=None, target_fps=30)
        if not self.video_handler.initialize():
            print("❌ Failed to initialize webcam")
            return False
        
        # Load terminals if available
        terminals_loaded = self.boundary_drawer.load_configuration()
        if terminals_loaded:
            print(f"✅ Loaded {len(self.boundary_drawer.terminals)} terminals")
        else:
            print("⚠️  No terminals loaded - testing detection only")
        
        # Reset metrics
        self.reset_metrics()
        
        cv2.namedWindow('Webcam Detection Test', cv2.WINDOW_NORMAL)
        
        start_time = time.time()
        frame_idx = 0
        paused = False
        
        print(f"🚀 Starting {duration_seconds}s webcam test...")
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if test duration completed
                if elapsed >= duration_seconds:
                    print(f"⏱️  Test duration ({duration_seconds}s) completed")
                    break
                
                if not paused:
                    ret, frame = self.video_handler.get_frame()
                    if not ret:
                        print("❌ Failed to get webcam frame")
                        break
                    
                    # Detect connectors
                    connectors = self.detector.detect_connectors(frame)
                    
                    # Update metrics
                    self.update_metrics(connectors, frame_idx)
                    frame_idx += 1
                
                # Create display
                display_frame = self.draw_test_overlay(frame, connectors, 
                                                     "Webcam Detection Test", frame_idx)
                
                # Add countdown
                remaining = duration_seconds - elapsed
                cv2.putText(display_frame, f"Time remaining: {remaining:.1f}s", 
                           (display_frame.shape[1] - 200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Pause indicator
                if paused:
                    cv2.putText(display_frame, "PAUSED", 
                               (display_frame.shape[1]//2 - 50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.imshow('Webcam Detection Test', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    self.save_test_results("webcam_test_results.json")
        
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        
        finally:
            self.video_handler.release()
            cv2.destroyAllWindows()
        
        # Calculate final metrics
        self.calculate_final_metrics()
        
        # Evaluate test success
        success = self._evaluate_webcam_test()
        self.test_results['webcam_test'] = {
            'passed': success,
            'metrics': self._get_metrics_dict()
        }
        
        print(f"\n📊 Webcam test {'PASSED' if success else 'FAILED'}")
        self._print_metrics()
        
        return success
    
    def test_video_detection(self, video_path: str, max_frames: int = 300) -> bool:
        """
        Test connector detection with video file
        
        Args:
            video_path: Path to test video file
            max_frames: Maximum frames to process
            
        Returns:
            bool: True if test passed
        """
        print(f"\n📹 VIDEO DETECTION TEST")
        print(f"Video: {video_path}")
        print(f"Max frames: {max_frames}")
        print("="*50)
        
        # Initialize video
        self.video_handler = VideoHandler(source=video_path, target_fps=30)
        if not self.video_handler.initialize():
            print("❌ Failed to initialize video")
            return False
        
        # Load terminals
        terminals_loaded = self.boundary_drawer.load_configuration()
        if terminals_loaded:
            print(f"✅ Loaded {len(self.boundary_drawer.terminals)} terminals")
        
        # Reset metrics
        self.reset_metrics()
        
        cv2.namedWindow('Video Detection Test', cv2.WINDOW_NORMAL)
        
        frame_idx = 0
        paused = False
        
        print(f"🚀 Starting video detection test...")
        
        try:
            while frame_idx < max_frames:
                if not paused:
                    ret, frame = self.video_handler.get_frame()
                    if not ret:
                        print("📹 End of video reached")
                        break
                    
                    # Detect connectors
                    connectors = self.detector.detect_connectors(frame)
                    
                    # Update metrics
                    self.update_metrics(connectors, frame_idx)
                    frame_idx += 1
                
                # Create display
                display_frame = self.draw_test_overlay(frame, connectors, 
                                                     "Video Detection Test", frame_idx)
                
                # Add progress
                progress = (frame_idx / max_frames) * 100
                cv2.putText(display_frame, f"Progress: {progress:.1f}% ({frame_idx}/{max_frames})", 
                           (display_frame.shape[1] - 300, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Pause indicator
                if paused:
                    cv2.putText(display_frame, "PAUSED", 
                               (display_frame.shape[1]//2 - 50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.imshow('Video Detection Test', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    self.save_test_results("video_test_results.json")
        
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        
        finally:
            self.video_handler.release()
            cv2.destroyAllWindows()
        
        # Calculate final metrics
        self.calculate_final_metrics()
        
        # Evaluate test success
        success = self._evaluate_video_test()
        self.test_results['video_test'] = {
            'passed': success,
            'metrics': self._get_metrics_dict()
        }
        
        print(f"\n📊 Video test {'PASSED' if success else 'FAILED'}")
        self._print_metrics()
        
        return success
    
    def _evaluate_webcam_test(self) -> bool:
        """Evaluate if webcam test passed based on metrics"""
        # Success criteria for webcam test
        criteria = {
            'min_detection_rate': 60,  # At least 60% of frames should have detections
            'min_avg_confidence': 0.5,  # Average confidence should be > 0.5
            'min_consistency': 40,     # Detection consistency > 40%
            'min_inference_fps': 20    # Inference should be > 20 FPS
        }
        
        detection_rate = (self.metrics.frames_with_detections / max(self.metrics.total_frames, 1)) * 100
        perf_stats = self.detector.get_performance_stats()
        inference_fps = perf_stats.get('avg_inference_fps', 0)
        
        checks = [
            detection_rate >= criteria['min_detection_rate'],
            self.metrics.avg_confidence >= criteria['min_avg_confidence'],
            self.metrics.detection_consistency >= criteria['min_consistency'],
            inference_fps >= criteria['min_inference_fps']
        ]
        
        return all(checks)
    
    def _evaluate_video_test(self) -> bool:
        """Evaluate if video test passed based on metrics"""
        # Similar criteria to webcam but potentially stricter
        criteria = {
            'min_detection_rate': 70,  # Higher expectation for video
            'min_avg_confidence': 0.5,
            'min_consistency': 50,
            'min_inference_fps': 20
        }
        
        detection_rate = (self.metrics.frames_with_detections / max(self.metrics.total_frames, 1)) * 100
        perf_stats = self.detector.get_performance_stats()
        inference_fps = perf_stats.get('avg_inference_fps', 0)
        
        checks = [
            detection_rate >= criteria['min_detection_rate'],
            self.metrics.avg_confidence >= criteria['min_avg_confidence'],
            self.metrics.detection_consistency >= criteria['min_consistency'],
            inference_fps >= criteria['min_inference_fps']
        ]
        
        return all(checks)
    
    def _get_metrics_dict(self) -> Dict[str, float]:
        """Get metrics as dictionary"""
        detection_rate = (self.metrics.frames_with_detections / max(self.metrics.total_frames, 1)) * 100
        
        return {
            'total_frames': self.metrics.total_frames,
            'detection_rate_percent': detection_rate,
            'avg_confidence': self.metrics.avg_confidence,
            'avg_detections_per_frame': self.metrics.avg_detections_per_frame,
            'detection_consistency': self.metrics.detection_consistency,
            'centroid_stability': self.metrics.centroid_stability,
            'avg_mask_area': self.metrics.avg_mask_area
        }
    
    def _print_metrics(self):
        """Print detailed metrics"""
        print("\n📈 Detailed Metrics:")
        print("-" * 30)
        metrics_dict = self._get_metrics_dict()
        for key, value in metrics_dict.items():
            print(f"{key}: {value:.2f}")
        
        # Performance metrics
        perf_stats = self.detector.get_performance_stats()
        if perf_stats:
            print("\n⚡ Performance Stats:")
            print("-" * 30)
            for key, value in perf_stats.items():
                print(f"{key}: {value:.2f}")
    
    def save_test_results(self, filename: str):
        """Save test results to JSON file"""
        try:
            results = {
                'test_results': self.test_results,
                'current_metrics': self._get_metrics_dict(),
                'performance_stats': self.detector.get_performance_stats(),
                'timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Test results saved to {filename}")
        
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def run_complete_test_suite(self, video_path: Optional[str] = None):
        """Run complete test suite for connector mask detection"""
        print("🧪 CONNECTOR MASK DETECTION - COMPLETE TEST SUITE")
        print("="*60)
        
        if not self.initialize_components():
            print("❌ Failed to initialize components")
            return
        
        # Test 1: Webcam detection
        print("\n1️⃣  Starting webcam detection test...")
        webcam_success = self.test_webcam_detection(duration_seconds=20)
        
        # Test 2: Video detection (if video provided)
        video_success = False
        if video_path:
            print("\n2️⃣  Starting video detection test...")
            video_success = self.test_video_detection(video_path, max_frames=200)
        else:
            print("\n⚠️  Skipping video test - no video path provided")
        
        # Final report
        self._print_final_report(webcam_success, video_success)
    
    def _print_final_report(self, webcam_success: bool, video_success: bool):
        """Print final test report"""
        print("\n" + "="*60)
        print("🏁 STEP 5 COMPLETE - FINAL REPORT")
        print("="*60)
        
        print(f"Webcam Detection Test: {'✅ PASSED' if webcam_success else '❌ FAILED'}")
        if video_success is not False:
            print(f"Video Detection Test: {'✅ PASSED' if video_success else '❌ FAILED'}")
        
        overall_success = webcam_success and (video_success or video_success is False)
        
        print(f"\nOverall Step 5 Status: {'✅ READY FOR STEP 6' if overall_success else '❌ NEEDS ATTENTION'}")
        
        if overall_success:
            print("\n💡 Recommendations:")
            print("   ✅ Connector detection is working reliably")
            print("   ✅ Ready to proceed to Step 6 (tracking and push detection)")
        else:
            print("\n⚠️  Issues to address:")
            if not webcam_success:
                print("   - Improve webcam detection reliability")
            print("   - Consider adjusting confidence thresholds")
            print("   - Verify model performance on your specific use case")

if __name__ == "__main__":
    print("🔍 Connector Mask Detection Tester - Step 5")
    print("="*50)
    
    # Get inputs
    model_path = "model/best-seg3.pt"
    if not model_path:
        print("❌ Model path required")
        exit(1)
    
    print("\nTest options:")
    print("1. Quick webcam test (20s)")
    print("2. Complete test suite (webcam + video)")
    print("3. Webcam test only (custom duration)")
    print("4. Video test only")
    
    choice = input("Enter choice (1-4): ").strip()
    
    tester = ConnectorMaskTester(model_path)
    
    if choice == "1":
        # Quick test
        if tester.initialize_components():
            tester.test_webcam_detection(20)
    
    elif choice == "2":
        # Complete suite
        video_path = "input videos/SampleClip1_2.mp4"
        tester.run_complete_test_suite(video_path if video_path else None)
    
    elif choice == "3":
        # Custom webcam test
        duration = int(input("Enter test duration in seconds (default 30): ") or "30")
        if tester.initialize_components():
            tester.test_webcam_detection(duration)
    
    elif choice == "4":
        # Video only
        video_path = "input videos/SampleClip1_2.mp4"
        if not video_path:
            print("❌ Video path required")
            exit(1)
        max_frames = int(input("Enter max frames to process (default 300): ") or "300")
        if tester.initialize_components():
            tester.test_video_detection(video_path, max_frames)
    
    else:
        print("Invalid choice. Running quick webcam test...")
        if tester.initialize_components():
            tester.test_webcam_detection(20)