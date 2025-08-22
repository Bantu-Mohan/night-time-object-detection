              # Night-Time Object Detection System

A comprehensive solution for enhancing object detection in night-time and low-light environments by integrating advanced image enhancement techniques with YOLOv8.

## Project Overview

This project addresses the challenges of object detection in low-light conditions by combining:

1. **Advanced Image Enhancement Pipeline** - Improves visibility in night-time images using gamma correction, CLAHE, and denoising techniques
2. **YOLOv8 Integration** - Leverages state-of-the-art object detection with enhanced images for better accuracy
3. **Real-time Processing** - Supports both image and video processing with performance monitoring

The system is designed for applications such as surveillance, autonomous driving, and defense where reliable night-time object detection is critical.

## Features

- **Robust Pre-Processing Pipeline**
  - Gamma Correction to brighten dark areas without oversaturating well-lit regions
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast
  - Optional Denoising to reduce high-frequency noise while preserving edges

- **YOLOv8 Integration**
  - Support for different YOLOv8 model sizes (nano, small, medium, large, xlarge)
  - Configurable confidence thresholds for detections
  - Fine-tuning capabilities for custom datasets

- **Real-Time Detection**
  - Process both images and videos (including camera streams)
  - Performance monitoring with FPS display
  - Visualization of detection results with bounding boxes and confidence scores

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- NumPy
- Ultralytics (YOLOv8)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/night-time-object-detection.git
   cd night-time-object-detection
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy matplotlib ultralytics
   ```

## Usage

### Command Line Interface

The system provides a command-line interface for easy use:

```bash
python main.py --input <path_to_image_or_video> --output <output_directory>
```

### Processing Images

```bash
python main.py --input path/to/night_image.jpg --mode image --gamma 1.8 --model m
```

### Processing Videos

```bash
python main.py --input path/to/night_video.mp4 --mode video --conf 0.3
```

### Using Camera Feed

```bash
python main.py --input 0 --mode video  # 0 is the camera index
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--input`, `-i` | Path to input image, video, or camera index |
| `--output`, `-o` | Directory to save output files (default: ./output) |
| `--mode`, `-m` | Processing mode: "image" or "video" (auto-detected if not specified) |
| `--model` | YOLOv8 model size: "n", "s", "m", "l", "x" (default: "n") |
| `--conf` | Confidence threshold (default: 0.25) |
| `--gamma` | Gamma correction value (default: 1.5) |
| `--no-clahe` | Disable CLAHE enhancement |
| `--no-denoise` | Disable denoising |
| `--no-save` | Don't save output files |
| `--no-display` | Don't display results |

## Project Structure

```
night_detection/
├── main.py                  # Main application integrating all components
├── image_enhancement.py     # Image enhancement pipeline implementation
├── yolo_detector.py         # YOLOv8 integration for object detection
├── README.md                # Project documentation
└── output/                  # Default directory for output files
```

## API Reference

### ImageEnhancer Class

The `ImageEnhancer` class provides methods for enhancing night-time or low-light images:

```python
from image_enhancement import ImageEnhancer

# Create an enhancer instance
enhancer = ImageEnhancer()

# Enhance an image with default parameters
enhanced_image = enhancer.enhance_image(image)

# Enhance with custom parameters
enhanced_image = enhancer.enhance_image(
    image, 
    gamma=1.8,               # Higher values brighten the image more
    apply_clahe=True,        # Enable/disable CLAHE
    apply_denoise=True,      # Enable/disable denoising
    clahe_clip_limit=2.0,    # CLAHE clip limit
    clahe_grid_size=(8, 8),  # CLAHE grid size
    denoise_d=15,            # Denoising diameter
    denoise_sigma_color=75,  # Denoising color sigma
    denoise_sigma_space=75   # Denoising space sigma
)

# Visualize enhancement results
enhancer.visualize_enhancement(original_image, enhanced_image, "comparison.jpg")
```

### YOLODetector Class

The `YOLODetector` class provides methods for object detection using YOLOv8:

```python
from yolo_detector import YOLODetector

# Create a detector instance
detector = YOLODetector(model_size='m', confidence=0.3)

# Detect objects in an image
annotated_image, detections = detector.detect(image)

# Detect objects in an enhanced image
annotated_image, detections = detector.detect(original_image, enhanced_image)

# Process video or camera feed
detector.detect_video(
    video_path="path/to/video.mp4",  # None for camera
    camera_id=0,                     # Camera index if video_path is None
    output_path="output_video.mp4",  # None to skip saving
    enhancer=enhancer,               # ImageEnhancer instance
    use_enhanced=True,               # Whether to use enhanced frames
    show_fps=True                    # Display FPS counter
)

# Fine-tune on custom dataset
detector.fine_tune(
    data_yaml_path="path/to/data.yaml",
    epochs=50,
    batch_size=16,
    img_size=640
)
```

## Performance Considerations

- **Processing Speed**: The enhancement pipeline may reduce processing speed, especially with denoising enabled. For real-time applications, consider:
  - Using a smaller YOLOv8 model (nano or small)
  - Disabling denoising (`--no-denoise`)
  - Reducing input resolution

- **GPU Acceleration**: YOLOv8 automatically uses GPU if available, significantly improving performance.

- **Memory Usage**: Larger YOLOv8 models require more memory. If you encounter memory issues, use a smaller model.

## Extending the Project

### Fine-tuning for Specific Environments

For optimal performance in specific night-time environments, fine-tune the YOLOv8 model:

1. Collect and annotate images from your target environment
2. Create a dataset in YOLOv8 format
3. Use the `fine_tune` method to adapt the model

### Adding New Enhancement Techniques

To implement additional enhancement techniques:

1. Add new methods to the `ImageEnhancer` class
2. Integrate them into the `enhance_image` method
3. Update the CLI arguments in `main.py` to expose the new options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The YOLOv8 model is developed by Ultralytics
- This project was inspired by the challenges of night-time computer vision applications
