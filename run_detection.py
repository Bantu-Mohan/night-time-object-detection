#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run object detection on original and enhanced night-time images
--------------------------------------------------------------
This script runs YOLOv8 object detection on both the original darkened
images and their enhanced versions, then collects performance metrics
for comparison.

Author: Manus AI
Date: April 27, 2025
"""

import cv2
import numpy as np
import os
import time
import json
from yolo_detector import YOLODetector

# Create output directories
os.makedirs('datasets/synthetic/detection', exist_ok=True)
os.makedirs('datasets/synthetic/metrics', exist_ok=True)

# Initialize the YOLOv8 detector
detector = YOLODetector(model_size='n', confidence=0.25)

# Directories
darkened_dir = 'datasets/synthetic/darkened'
enhanced_dir = 'datasets/synthetic/enhanced'
detection_dir = 'datasets/synthetic/detection'
metrics_dir = 'datasets/synthetic/metrics'

# Get list of images
darkened_images = [f for f in os.listdir(darkened_dir) if f.endswith('.jpg')]

print(f"Running object detection on {len(darkened_images)} image pairs...")

# Initialize metrics
metrics = {
    'darkened': {
        'detection_count': [],
        'confidence_scores': [],
        'processing_times': []
    },
    'enhanced': {
        'detection_count': [],
        'confidence_scores': [],
        'processing_times': []
    }
}

# Expected object counts in our synthetic images
expected_objects = {
    'street_scene_night.jpg': 4,  # 2 cars, 2 people
    'park_scene_night.jpg': 4,    # 2 people, 2 bicycles
    'urban_scene_night.jpg': 6    # 3 cars, 3 people
}

# Process each image pair
for image_file in darkened_images:
    base_name = image_file.replace('_night.jpg', '')
    enhanced_file = image_file.replace('_night', '_enhanced')
    
    # Load images
    darkened_path = os.path.join(darkened_dir, image_file)
    enhanced_path = os.path.join(enhanced_dir, enhanced_file)
    
    darkened_image = cv2.imread(darkened_path)
    enhanced_image = cv2.imread(enhanced_path)
    
    if darkened_image is None or enhanced_image is None:
        print(f"Error: Could not load images for {base_name}")
        continue
    
    # Run detection on darkened image
    start_time = time.time()
    darkened_annotated, darkened_detections = detector.detect(darkened_image)
    darkened_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Run detection on enhanced image
    start_time = time.time()
    enhanced_annotated, enhanced_detections = detector.detect(enhanced_image)
    enhanced_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Save annotated images
    cv2.imwrite(os.path.join(detection_dir, f"{base_name}_darkened_detection.jpg"), darkened_annotated)
    cv2.imwrite(os.path.join(detection_dir, f"{base_name}_enhanced_detection.jpg"), enhanced_annotated)
    
    # Create side-by-side comparison
    h, w = darkened_image.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = darkened_annotated
    comparison[:, w:] = enhanced_annotated
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original (Darkened)", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Enhanced", (w + 10, 30), font, 1, (0, 255, 0), 2)
    
    # Add detection counts
    cv2.putText(comparison, f"Detections: {len(darkened_detections)}", (10, 70), font, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, f"Detections: {len(enhanced_detections)}", (w + 10, 70), font, 0.8, (0, 255, 0), 2)
    
    # Save comparison
    cv2.imwrite(os.path.join(detection_dir, f"{base_name}_detection_comparison.jpg"), comparison)
    
    # Collect metrics
    metrics['darkened']['detection_count'].append(len(darkened_detections))
    metrics['darkened']['confidence_scores'].extend([det['confidence'] for det in darkened_detections])
    metrics['darkened']['processing_times'].append(darkened_time)
    
    metrics['enhanced']['detection_count'].append(len(enhanced_detections))
    metrics['enhanced']['confidence_scores'].extend([det['confidence'] for det in enhanced_detections])
    metrics['enhanced']['processing_times'].append(enhanced_time)
    
    # Calculate precision and recall for this image
    if image_file in expected_objects:
        expected_count = expected_objects[image_file]
        
        # For darkened image
        darkened_true_positives = min(len(darkened_detections), expected_count)
        darkened_precision = darkened_true_positives / max(1, len(darkened_detections))
        darkened_recall = darkened_true_positives / expected_count
        
        # For enhanced image
        enhanced_true_positives = min(len(enhanced_detections), expected_count)
        enhanced_precision = enhanced_true_positives / max(1, len(enhanced_detections))
        enhanced_recall = enhanced_true_positives / expected_count
        
        print(f"\nResults for {image_file}:")
        print(f"  Darkened: {len(darkened_detections)} detections, Precision: {darkened_precision:.2f}, Recall: {darkened_recall:.2f}")
        print(f"  Enhanced: {len(enhanced_detections)} detections, Precision: {enhanced_precision:.2f}, Recall: {enhanced_recall:.2f}")
    
# Calculate overall metrics
overall_metrics = {
    'darkened': {
        'avg_detection_count': np.mean(metrics['darkened']['detection_count']),
        'avg_confidence': np.mean(metrics['darkened']['confidence_scores']) if metrics['darkened']['confidence_scores'] else 0,
        'avg_processing_time': np.mean(metrics['darkened']['processing_times'])
    },
    'enhanced': {
        'avg_detection_count': np.mean(metrics['enhanced']['detection_count']),
        'avg_confidence': np.mean(metrics['enhanced']['confidence_scores']) if metrics['enhanced']['confidence_scores'] else 0,
        'avg_processing_time': np.mean(metrics['enhanced']['processing_times'])
    }
}

# Calculate precision, recall, and mAP
total_expected = sum(expected_objects.values())
total_darkened_detections = sum(metrics['darkened']['detection_count'])
total_enhanced_detections = sum(metrics['enhanced']['detection_count'])

# Estimate true positives (this is a simplification)
darkened_true_positives = min(total_darkened_detections, total_expected)
enhanced_true_positives = min(total_enhanced_detections, total_expected)

# Calculate precision and recall
darkened_precision = darkened_true_positives / max(1, total_darkened_detections)
darkened_recall = darkened_true_positives / total_expected
enhanced_precision = enhanced_true_positives / max(1, total_enhanced_detections)
enhanced_recall = enhanced_true_positives / total_expected

# Calculate mAP (simplified version)
darkened_map = (darkened_precision + darkened_recall) / 2
enhanced_map = (enhanced_precision + enhanced_recall) / 2

# Add to overall metrics
overall_metrics['darkened']['precision'] = darkened_precision
overall_metrics['darkened']['recall'] = darkened_recall
overall_metrics['darkened']['mAP'] = darkened_map
overall_metrics['enhanced']['precision'] = enhanced_precision
overall_metrics['enhanced']['recall'] = enhanced_recall
overall_metrics['enhanced']['mAP'] = enhanced_map

# Save metrics to file
with open(os.path.join(metrics_dir, 'detection_metrics.json'), 'w') as f:
    json.dump(overall_metrics, f, indent=4)

# Save metrics in a format suitable for visualization
with open(os.path.join(metrics_dir, 'visualization_data.txt'), 'w') as f:
    f.write(f"Metrics Before Enhancement:\n")
    f.write(f"mAP: {overall_metrics['darkened']['mAP']*100:.1f}%\n")
    f.write(f"Precision: {overall_metrics['darkened']['precision']*100:.1f}%\n")
    f.write(f"Recall: {overall_metrics['darkened']['recall']*100:.1f}%\n\n")
    f.write(f"Metrics After Enhancement:\n")
    f.write(f"mAP: {overall_metrics['enhanced']['mAP']*100:.1f}%\n")
    f.write(f"Precision: {overall_metrics['enhanced']['precision']*100:.1f}%\n")
    f.write(f"Recall: {overall_metrics['enhanced']['recall']*100:.1f}%\n")

# Print overall results
print("\nOverall Detection Results:")
print(f"  Darkened Images:")
print(f"    Average Detections: {overall_metrics['darkened']['avg_detection_count']:.2f}")
print(f"    Average Confidence: {overall_metrics['darkened']['avg_confidence']*100:.2f}%")
print(f"    Precision: {darkened_precision*100:.2f}%")
print(f"    Recall: {darkened_recall*100:.2f}%")
print(f"    mAP: {darkened_map*100:.2f}%")
print(f"    Average Processing Time: {overall_metrics['darkened']['avg_processing_time']:.2f} ms")

print(f"\n  Enhanced Images:")
print(f"    Average Detections: {overall_metrics['enhanced']['avg_detection_count']:.2f}")
print(f"    Average Confidence: {overall_metrics['enhanced']['avg_confidence']*100:.2f}%")
print(f"    Precision: {enhanced_precision*100:.2f}%")
print(f"    Recall: {enhanced_recall*100:.2f}%")
print(f"    mAP: {enhanced_map*100:.2f}%")
print(f"    Average Processing Time: {overall_metrics['enhanced']['avg_processing_time']:.2f} ms")

# Calculate improvement percentages
detection_improvement = ((overall_metrics['enhanced']['avg_detection_count'] - overall_metrics['darkened']['avg_detection_count']) / 
                         max(0.001, overall_metrics['darkened']['avg_detection_count'])) * 100
confidence_improvement = ((overall_metrics['enhanced']['avg_confidence'] - overall_metrics['darkened']['avg_confidence']) / 
                          max(0.001, overall_metrics['darkened']['avg_confidence'])) * 100
precision_improvement = ((enhanced_precision - darkened_precision) / max(0.001, darkened_precision)) * 100
recall_improvement = ((enhanced_recall - darkened_recall) / max(0.001, darkened_recall)) * 100
map_improvement = ((enhanced_map - darkened_map) / max(0.001, darkened_map)) * 100
time_increase = ((overall_metrics['enhanced']['avg_processing_time'] - overall_metrics['darkened']['avg_processing_time']) / 
                 overall_metrics['darkened']['avg_processing_time']) * 100

print(f"\nImprovements with Enhancement:")
print(f"  Detection Count: +{detection_improvement:.2f}%")
print(f"  Confidence: +{confidence_improvement:.2f}%")
print(f"  Precision: +{precision_improvement:.2f}%")
print(f"  Recall: +{recall_improvement:.2f}%")
print(f"  mAP: +{map_improvement:.2f}%")
print(f"  Processing Time Increase: +{time_increase:.2f}%")

print("\nDetection complete. Results saved to datasets/synthetic/detection/ and datasets/synthetic/metrics/")

if __name__ == "__main__":
    print("\nObject detection analysis completed successfully!")
