import cv2
import torch
from ultralytics import YOLO
import numpy as np

def load_model(model_path, device='cpu'):
    """Load the YOLO model with specified device"""
    try:
        model = YOLO(model_path)
        
        # Force model to use specified device
        model.to(device)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_video_with_detection(model_path, video_path, device='cpu', conf_threshold=0.5):
    """Process video and display with bounding boxes"""
    
    # Load the model with specified device
    model = load_model(model_path, device)
    if model is None:
        return
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} at {fps} FPS")
    
    # Create window for display
    cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or failed to read frame")
            break
        
        frame_count += 1
        
        try:
            # Run YOLO detection with specified device and confidence threshold
            results = model(frame, device=device, conf=conf_threshold, verbose=False)
            
            # Draw bounding boxes on frame
            annotated_frame = results[0].plot()
            
            # Add frame counter and device info
            cv2.putText(annotated_frame, f"Frame: {frame_count} | Device: {device}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('YOLO Detection', annotated_frame)
            
            # Print detection info for first few frames
            if frame_count <= 5:
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    print(f"Frame {frame_count}: {len(detections)} objects detected")
                    for i, box in enumerate(detections):
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                        print(f"  Object {i+1}: {class_name} (confidence: {conf:.2f})")
                else:
                    print(f"Frame {frame_count}: No objects detected")
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # If CUDA error occurs, fall back to CPU
            if "CUDA" in str(e) and device != 'cpu':
                print("CUDA error detected, falling back to CPU...")
                model = load_model(model_path, 'cpu')
                device = 'cpu'
                continue
            else:
                break
        
        # Break on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping video playback...")
            break
        elif key == ord(' '):  # Space bar to pause
            print("Paused - press any key to continue...")
            cv2.waitKey(0)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed")

def save_detection_video(model_path, video_path, output_path, device='cpu', conf_threshold=0.5):
    """Save video with detections to file"""
    
    model = load_model(model_path, device)
    if model is None:
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        try:
            # Run detection
            results = model(frame, device=device, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Write frame to output video
            out.write(annotated_frame)
            
            if frame_count % 30 == 0:  # Print progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)...")
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # If CUDA error occurs, fall back to CPU
            if "CUDA" in str(e) and device != 'cpu':
                print("CUDA error detected, falling back to CPU...")
                model = load_model(model_path, 'cpu')
                device = 'cpu'
                continue
            else:
                # Write original frame if detection fails
                out.write(frame)
    
    # Clean up
    cap.release()
    out.release()
    print(f"Detection video saved to {output_path}")

def test_model_compatibility(model_path):
    """Test model compatibility with different devices"""
    print("=== Testing Model Compatibility ===")
    
    # Test CPU
    print("\nTesting CPU inference...")
    try:
        model_cpu = YOLO(model_path)
        model_cpu.to('cpu')
        
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model_cpu(dummy_image, device='cpu', verbose=False)
        print("✅ CPU inference: SUCCESS")
    except Exception as e:
        print(f"❌ CPU inference: FAILED - {e}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\nTesting CUDA inference...")
        try:
            model_cuda = YOLO(model_path)
            model_cuda.to('cuda')
            
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = model_cuda(dummy_image, device='cuda', verbose=False)
            print("✅ CUDA inference: SUCCESS")
        except Exception as e:
            print(f"❌ CUDA inference: FAILED - {e}")
            print("Recommendation: Use CPU inference instead")
    else:
        print("\n⚠️  CUDA not available - CPU inference recommended")

if __name__ == "__main__":
    # Your file paths
    MODEL_PATH = "model/best2.pt"
    VIDEO_PATH = "input videos/SampleClip1 - Made with Clipchamp.mp4"
    
    # Test model compatibility first
    test_model_compatibility(MODEL_PATH)
    
    # Determine best device to use
    device = 'cpu'  # Default to CPU to avoid CUDA issues
    
    # Uncomment below to try CUDA if you want to test it
    # if torch.cuda.is_available():
    #     try:
    #         # Quick test with CUDA
    #         test_model = YOLO(MODEL_PATH)
    #         test_model.to('cuda')
    #         dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    #         test_model(dummy, device='cuda', verbose=False)
    #         device = 'cuda'
    #         print("Using CUDA for inference")
    #     except:
    #         device = 'cpu'
    #         print("CUDA failed, using CPU for inference")
    
    print(f"\nUsing device: {device}")
    
    # Option 1: Display video with real-time detection
    print("Starting real-time video detection display...")
    print("Press 'q' to quit, 'space' to pause")
    process_video_with_detection(MODEL_PATH, VIDEO_PATH, device=device, conf_threshold=0.6)
    
    # Option 2: Save detection video (uncomment to use)
    # OUTPUT_PATH = "output_with_detections.mp4"
    # print(f"Saving detection video to {OUTPUT_PATH}...")
    # save_detection_video(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH, device=device, conf_threshold=0.3)