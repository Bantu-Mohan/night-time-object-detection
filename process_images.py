#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process synthetic night-time images with enhancement pipeline
------------------------------------------------------------
This script processes the synthetic night-time images with our
enhancement pipeline and saves the enhanced versions for comparison.

Author: Manus AI
Date: April 27, 2025
"""

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from image_enhancement import ImageEnhancer
from yolo_detector import YOLODetector

# Create output directories
os.makedirs('datasets/synthetic/enhanced', exist_ok=True)
os.makedirs('datasets/synthetic/results', exist_ok=True)

# Initialize the image enhancer
enhancer = ImageEnhancer()

# Process each darkened image
darkened_dir = 'datasets/synthetic/darkened'
enhanced_dir = 'datasets/synthetic/enhanced'
results_dir = 'datasets/synthetic/results'

# Get list of darkened images
darkened_images = [f for f in os.listdir(darkened_dir) if f.endswith('.jpg')]

print(f"Processing {len(darkened_images)} night-time images...")

# Process each image and measure time
processing_times = []

for image_file in darkened_images:
    # Load darkened image
    image_path = os.path.join(darkened_dir, image_file)
    darkened_image = cv2.imread(image_path)
    
    if darkened_image is None:
        print(f"Error: Could not load image {image_path}")
        continue
    
    # Measure processing time
    start_time = time.time()
    
    # Apply enhancement pipeline
    enhanced_image = enhancer.enhance_image(
        darkened_image,
        gamma=1.8,               # Brighten the dark image
        apply_clahe=True,        # Apply CLAHE for contrast enhancement
        apply_denoise=True,      # Apply denoising
        clahe_clip_limit=3.0,    # Increase clip limit for better contrast
        clahe_grid_size=(8, 8),  # Default grid size
        denoise_d=15,            # Denoising diameter
        denoise_sigma_color=75,  # Color sigma for bilateral filter
        denoise_sigma_space=75   # Space sigma for bilateral filter
    )
    
    # Calculate processing time
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
    processing_times.append(processing_time)
    
    print(f"Enhanced {image_file} in {processing_time:.2f} ms")
    
    # Save enhanced image
    enhanced_path = os.path.join(enhanced_dir, image_file.replace('_night', '_enhanced'))
    cv2.imwrite(enhanced_path, enhanced_image)
    
    # Create comparison visualization
    comparison_path = os.path.join(results_dir, image_file.replace('_night.jpg', '_comparison.jpg'))
    enhancer.visualize_enhancement(darkened_image, enhanced_image, comparison_path)

# Print average processing time
avg_time = np.mean(processing_times)
print(f"\nAverage processing time: {avg_time:.2f} ms per image")

print("\nEnhancement complete. Enhanced images saved to datasets/synthetic/enhanced/")
print("Comparison visualizations saved to datasets/synthetic/results/")

# Return the average processing time for later use
with open('datasets/synthetic/results/processing_time.txt', 'w') as f:
    f.write(f"{avg_time:.2f}")

if __name__ == "__main__":
    print("\nProcessing completed successfully!")
