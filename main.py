#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Night-Time Object Detection System
----------------------------------
This is the main application that integrates the image enhancement pipeline
with YOLOv8 object detection for improved detection in night-time or low-light
conditions.

Author: Manus AI
Date: April 27, 2025
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

# Import our custom modules
from image_enhancement import ImageEnhancer
from yolo_detector import YOLODetector

def process_image(image_path, output_dir, model_size='n', conf=0.25, 
                 gamma=1.5, use_clahe=True, use_denoise=True,
                 save_enhanced=True, show_results=True):
    """
    Process a single image with the night-time object detection system.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save output images.
        model_size (str): Size of YOLOv8 model to use ('n', 's', 'm', 'l', 'x').
        conf (float): Confidence threshold for detections.
        gamma (float): Gamma value for correction.
        use_clahe (bool): Whether to apply CLAHE enhancement.
        use_denoise (bool): Whether to apply denoising.
        save_enhanced (bool): Whether to save the enhanced image.
        show_results (bool): Whether to display results.
    
    Returns:
        tuple: (original_image, enhanced_image, detection_results)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None
    
    # Initialize enhancer and detector
    enhancer = ImageEnhancer()
    detector = YOLODetector(model_size=model_size, confidence=conf)
    
    # Enhance image
    enhanced_image = enhancer.enhance_image(
        image, 
        gamma=gamma, 
        apply_clahe=use_clahe, 
        apply_denoise=use_denoise
    )
    
    # Perform detection on enhanced image
    annotated_image, detections = detector.detect(image, enhanced_image, show_original=True)
    
    # Save results
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save enhanced image if requested
    if save_enhanced:
        enhanced_path = os.path.join(output_dir, f"{image_name}_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced_image)
        print(f"Enhanced image saved to: {enhanced_path}")
    
    # Save annotated image
    detection_path = os.path.join(output_dir, f"{image_name}_detection.jpg")
    cv2.imwrite(detection_path, annotated_image)
    print(f"Detection results saved to: {detection_path}")
    
    # Save comparison visualization
    comparison_path = os.path.join(output_dir, f"{image_name}_comparison.jpg")
    enhancer.visualize_enhancement(image, enhanced_image, comparison_path)
    print(f"Comparison visualization saved to: {comparison_path}")
    
    # Display results if requested
    if show_results:
        # Create a comparison image for display
        cv2.imshow("Original vs Enhanced vs Detection", np.hstack([
            cv2.resize(image, (640, 480)),
            cv2.resize(enhanced_image, (640, 480)),
            cv2.resize(annotated_image, (640, 480))
        ]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print detection results
    print(f"\nDetected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} (Confidence: {det['confidence']:.2f})")
    
    return image, enhanced_image, detections

def process_video(video_path, output_dir, model_size='n', conf=0.25,
                 gamma=1.5, use_clahe=True, use_denoise=True,
                 save_output=True, show_results=True):
    """
    Process a video with the night-time object detection system.
    
    Args:
        video_path (str): Path to the input video or camera index.
        output_dir (str): Directory to save output video.
        model_size (str): Size of YOLOv8 model to use ('n', 's', 'm', 'l', 'x').
        conf (float): Confidence threshold for detections.
        gamma (float): Gamma value for correction.
        use_clahe (bool): Whether to apply CLAHE enhancement.
        use_denoise (bool): Whether to apply denoising.
        save_output (bool): Whether to save the output video.
        show_results (bool): Whether to display results.
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize enhancer and detector
    enhancer = ImageEnhancer()
    detector = YOLODetector(model_size=model_size, confidence=conf)
    
    # Determine video source
    if video_path.isdigit():
        video_source = int(video_path)
        output_name = f"camera_{video_source}_output.mp4"
    else:
        video_source = video_path
        output_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_output.mp4"
    
    # Set output path
    output_path = os.path.join(output_dir, output_name) if save_output else None
    
    # Process video
    print(f"Processing video from: {video_source}")
    print(f"Enhancement settings: Gamma={gamma}, CLAHE={use_clahe}, Denoise={use_denoise}")
    print(f"Detection settings: Model=YOLOv8{model_size}, Confidence={conf}")
    if output_path:
        print(f"Output will be saved to: {output_path}")
    
    # Configure enhancer parameters
    def enhance_frame(frame):
        return enhancer.enhance_image(
            frame, 
            gamma=gamma, 
            apply_clahe=use_clahe, 
            apply_denoise=use_denoise
        )
    
    # Start video processing
    detector.detect_video(
        video_path=None if isinstance(video_source, int) else video_source,
        camera_id=video_source if isinstance(video_source, int) else 0,
        output_path=output_path,
        enhancer=enhancer if (use_clahe or use_denoise or gamma != 1.0) else None,
        use_enhanced=True,
        show_fps=True
    )

def main():
    """
    Main function to parse arguments and run the night-time object detection system.
    """
    parser = argparse.ArgumentParser(description="Night-Time Object Detection System")
    
    # Input/output arguments
    parser.add_argument("--input", "-i", required=True, help="Path to input image, video, or camera index")
    parser.add_argument("--output", "-o", default="./output", help="Directory to save output files")
    
    # Mode selection
    parser.add_argument("--mode", "-m", choices=["image", "video"], help="Processing mode (auto-detected if not specified)")
    
    # YOLOv8 settings
    parser.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="n", help="YOLOv8 model size (default: n)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    # Enhancement settings
    parser.add_argument("--gamma", type=float, default=1.5, help="Gamma correction value (default: 1.5)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE enhancement")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    
    # Output settings
    parser.add_argument("--no-save", action="store_true", help="Don't save output files")
    parser.add_argument("--no-display", action="store_true", help="Don't display results")
    
    args = parser.parse_args()
    
    # Determine processing mode if not specified
    if args.mode is None:
        if args.input.isdigit() or args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            args.mode = "video"
        else:
            args.mode = "image"
    
    # Process based on mode
    if args.mode == "image":
        process_image(
            image_path=args.input,
            output_dir=args.output,
            model_size=args.model,
            conf=args.conf,
            gamma=args.gamma,
            use_clahe=not args.no_clahe,
            use_denoise=not args.no_denoise,
            save_enhanced=not args.no_save,
            show_results=not args.no_display
        )
    else:  # video mode
        process_video(
            video_path=args.input,
            output_dir=args.output,
            model_size=args.model,
            conf=args.conf,
            gamma=args.gamma,
            use_clahe=not args.no_clahe,
            use_denoise=not args.no_denoise,
            save_output=not args.no_save,
            show_results=not args.no_display
        )

if __name__ == "__main__":
    main()
