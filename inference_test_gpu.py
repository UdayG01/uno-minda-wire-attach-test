import cv2
import torch
import torchvision
from ultralytics import YOLO
import numpy as np
import os

def check_cuda_setup():
    """Check CUDA setup and compatibility"""
    print("=== CUDA Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        # Test basic CUDA operations
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print("✅ Basic CUDA operations: SUCCESS")
        except Exception as e:
            print(f"❌ Basic CUDA operations: FAILED - {e}")
            return False
        
        # Test torchvision NMS on CUDA
        try:
            boxes = torch.tensor([[0, 0, 100, 100], [10, 10, 110, 110]], dtype=torch.float32).cuda()
            scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
            keep = torchvision.ops.nms(boxes, scores, 0.5)
            print("✅ Torchvision CUDA NMS: SUCCESS")
            return True
        except Exception as e:
            print(f"❌ Torchvision CUDA NMS: FAILED - {e}")
            print("This is the source of your original error!")
            return False
    else:
        print("❌ CUDA not available")
        return False

def install_cuda_compatible_packages():
    """Print commands to install CUDA-compatible packages"""
    print("\n=== To Fix CUDA NMS Issues ===")
    print("Run these commands to install CUDA-compatible versions:")
    print("\n# For CUDA 11.8:")
    print("pip uninstall torch torchvision ultralytics")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("pip install ultralytics")
    print("\n# For CUDA 12.1:")
    print("pip uninstall torch torchvision ultralytics")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("pip install ultralytics")
    print("\n# Alternative: Force compilation with CUDA support")
    print("pip install torch torchvision --force-reinstall --no-cache-dir")

def load_model_gpu(model_path, gpu_id=0):
    """Load YOLO model optimized for GPU inference"""
    try:
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f'cuda:{gpu_id}'
        else:
            raise RuntimeError("CUDA not available")
        
        # Load model
        model = YOLO(model_path)
        
        # Move model to GPU with explicit device setting
        model.to(device)
        
        # Optimize for inference
        model.model.eval()
        
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"Model device: {next(model.model.parameters()).device}")
        
        return model, device
    except Exception as e:
        print(f"❌ Error loading model on GPU: {e}")
        return None, None

def warm_up_gpu_model(model, device, input_size=(640, 640)):
    """Warm up GPU model for optimal performance"""
    print("🔥 Warming up GPU model...")
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # Run several warm-up iterations
        with torch.no_grad():
            for i in range(10):
                _ = model.model(dummy_input)
        
        # Clear cache
        torch.cuda.empty_cache()
        print("✅ GPU warm-up completed")
    except Exception as e:
        print(f"❌ GPU warm-up failed: {e}")

def process_video_gpu_optimized(model_path, video_path, gpu_id=0, conf_threshold=0.5, imgsz=640):
    """GPU-optimized video processing with YOLO detection"""
    
    # Check CUDA setup first
    if not check_cuda_setup():
        print("❌ CUDA setup issues detected. Please fix before running GPU inference.")
        install_cuda_compatible_packages()
        return
    
    # Load model on GPU
    model, device = load_model_gpu(model_path, gpu_id)
    if model is None:
        return
    
    # Warm up model
    warm_up_gpu_model(model, device, (imgsz, imgsz))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # Total video duration in seconds
    
    print(f"📹 Video: {width}x{height} @ {fps}FPS, {total_frames} frames ({video_duration:.2f}s)")
    print(f"🚀 Processing on {device} with confidence threshold {conf_threshold}")
    
    # Create window
    cv2.namedWindow('GPU YOLO Detection', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    inference_times = []
    
    # Detection tracking variables
    detection_log = []
    last_detection_time = None
    detection_count = 0
    
    # Enable GPU memory optimization
    with torch.cuda.amp.autocast():  # Mixed precision for faster inference
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # Start timing
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                # Run inference with GPU optimizations
                results = model(
                    frame,
                    device=device,
                    conf=conf_threshold,
                    imgsz=imgsz,
                    verbose=False,
                    half=True,  # Use FP16 for faster inference
                    agnostic_nms=True  # Faster NMS
                )
                
                # End timing
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                inference_times.append(inference_time)
                
                # Calculate timestamp for this frame
                timestamp = frame_count / fps
                
                # Draw annotations
                annotated_frame = results[0].plot()
                
                # Add performance info
                avg_time = np.mean(inference_times[-30:])  # Last 30 frames average
                fps_inference = 1000.0 / avg_time if avg_time > 0 else 0
                
                info_text = [
                    f"Frame: {frame_count}/{total_frames}",
                    f"Time: {timestamp:.2f}s",
                    f"GPU: {device}",
                    f"Inference: {inference_time:.1f}ms",
                    f"FPS: {fps_inference:.1f}",
                    f"Memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('GPU YOLO Detection', annotated_frame)
                
                # Check for detections and log with timestamps
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    detection_count += 1
                    
                    # Calculate time since last detection
                    time_since_last = None
                    if last_detection_time is not None:
                        time_since_last = timestamp - last_detection_time
                    
                    # Log detection details
                    detection_info = {
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'time_since_last': time_since_last,
                        'detections': []
                    }
                    
                    print(f"🎯 Frame {frame_count}: {len(detections)} objects detected at {timestamp:.2f}s ({inference_time:.1f}ms)")
                    
                    for i, box in enumerate(detections):
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                        
                        detection_info['detections'].append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': box.xyxy.cpu().numpy().tolist()[0]
                        })
                        
                        print(f"   • {class_name}: {conf:.3f}")
                    
                    if time_since_last is not None:
                        print(f"   ⏱️  Time since last detection: {time_since_last:.2f}s")
                    
                    detection_log.append(detection_info)
                    last_detection_time = timestamp
                
                # Print periodic updates for non-detection frames
                elif frame_count <= 5 or frame_count % 100 == 0:
                    print(f"📭 Frame {frame_count}: No objects detected at {timestamp:.2f}s ({inference_time:.1f}ms)")
                
                # Memory management
                if frame_count % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error processing frame {frame_count}: {e}")
                if "CUDA" in str(e):
                    print("💡 CUDA error detected. Try reinstalling PyTorch with CUDA support.")
                    break
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping video playback...")
                break
            elif key == ord(' '):
                print("⏸️  Paused - press any key to continue...")
                cv2.waitKey(0)
            elif key == ord('r'):  # Reset GPU memory
                torch.cuda.empty_cache()
                print("🧹 GPU memory cleared")
    
    # Cleanup and stats
    cap.release()
    cv2.destroyAllWindows()
    
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        avg_fps = 1000.0 / avg_inference_time
        print(f"\n📊 Performance Statistics:")
        print(f"   Average inference time: {avg_inference_time:.2f}ms")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Frames processed: {frame_count}")
        print(f"   GPU memory peak: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
    
    torch.cuda.empty_cache()
    print("✅ GPU video processing completed")

def save_detection_video_gpu(model_path, video_path, output_path, gpu_id=0, conf_threshold=0.5, imgsz=640):
    """Save GPU-accelerated detection video"""
    
    if not check_cuda_setup():
        print("❌ CUDA setup issues detected.")
        return
    
    model, device = load_model_gpu(model_path, gpu_id)
    if model is None:
        return
    
    warm_up_gpu_model(model, device, (imgsz, imgsz))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    inference_times = []
    
    print(f"🎬 Saving GPU-accelerated detection video...")
    print(f"📹 Input: {width}x{height} @ {fps}FPS, {total_frames} frames")
    print(f"🚀 Processing on {device}")
    
    with torch.cuda.amp.autocast():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # GPU inference with timing
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                results = model(
                    frame,
                    device=device,
                    conf=conf_threshold,
                    imgsz=imgsz,
                    verbose=False,
                    half=True,
                    agnostic_nms=True
                )
                
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                inference_times.append(inference_time)
                
                # Generate annotated frame
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
                # Progress reporting
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_time = np.mean(inference_times[-50:])
                    eta_seconds = (total_frames - frame_count) * avg_time / 1000
                    print(f"⚡ Progress: {frame_count}/{total_frames} ({progress:.1f}%) "
                          f"- {avg_time:.1f}ms/frame - ETA: {eta_seconds:.0f}s")
                
                # Memory management
                if frame_count % 200 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error processing frame {frame_count}: {e}")
                out.write(frame)  # Write original frame on error
    
    # Cleanup
    cap.release()
    out.release()
    
    if inference_times:
        avg_time = np.mean(inference_times)
        total_time = sum(inference_times) / 1000  # Convert to seconds
        print(f"\n🎉 Video saved successfully to {output_path}")
        print(f"📊 Statistics:")
        print(f"   Total processing time: {total_time:.1f}s")
        print(f"   Average inference time: {avg_time:.2f}ms")
        print(f"   Average FPS: {1000/avg_time:.2f}")
        print(f"   Speedup vs realtime: {(total_frames/fps)/total_time:.2f}x")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "model/seg-con2.pt"
    VIDEO_PATH = "input videos/SampleClip1_2.mp4"
    GPU_ID = 0  # Change if you have multiple GPUs
    CONF_THRESHOLD = 0.25
    IMG_SIZE = 640  # YOLO input size (higher = better accuracy, slower inference)
    
    print("🚀 GPU-Optimized YOLO Inference")
    print("=" * 50)
    
    # Check system first
    check_cuda_setup()
    
    # Option 1: Real-time GPU detection display
    print("\n🎮 Starting GPU-accelerated real-time detection...")
    print("Controls: 'q' to quit, 'space' to pause, 'r' to clear GPU memory")
    process_video_gpu_optimized(
        MODEL_PATH, 
        VIDEO_PATH, 
        gpu_id=GPU_ID, 
        conf_threshold=CONF_THRESHOLD,
        imgsz=IMG_SIZE
    )
    
    # Option 2: Save GPU-accelerated detection video (uncomment to use)
    OUTPUT_PATH = "output videos/gpu_output_with_detections.mp4"
    print(f"\n💾 Saving GPU-accelerated detection video...")
    save_detection_video_gpu(
        MODEL_PATH, 
        VIDEO_PATH, 
        OUTPUT_PATH, 
        gpu_id=GPU_ID, 
        conf_threshold=CONF_THRESHOLD,
        imgsz=IMG_SIZE
    )