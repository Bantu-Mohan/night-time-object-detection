#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Integration for Night-Time Object Detection
-------------------------------------------------
This module implements the integration of YOLOv8 with the image enhancement
pipeline for improved object detection in night-time or low-light conditions.

Author: Manus AI
Date: April 27, 2025
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from pathlib import Path

class YOLODetector:
    """
    A class that provides methods for object detection using YOLOv8 model
    with enhanced night-time images.
    """
    
    def __init__(self, model_size='n', confidence=0.25):
        """
        Initialize the YOLODetector class.
        
        Args:
            model_size (str): Size of the YOLOv8 model to use ('n', 's', 'm', 'l', 'x').
                             Default is 'n' (nano).
            confidence (float): Confidence threshold for detections. Default is 0.25.
        """
        # Load the YOLOv8 model
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.confidence = confidence
        
        # Class names for COCO dataset (default for YOLOv8)
        self.class_names = self.model.names
    
    def detect(self, image, enhanced_image=None, show_original=False):
        """
        Perform object detection on the input image.
        
        Args:
            image (numpy.ndarray): Original input image.
            enhanced_image (numpy.ndarray, optional): Enhanced image. If None, detection
                                                    is performed on the original image.
            show_original (bool): Whether to show detection results on the original image
                                 even when using enhanced image for detection.
        
        Returns:
            tuple: (annotated_image, detections) where:
                  - annotated_image is the image with detection boxes
                  - detections is a list of detection results
        """
        # Use enhanced image for detection if provided, otherwise use original
        detection_image = enhanced_image if enhanced_image is not None else image
        
        # Perform detection
        results = self.model(detection_image, conf=self.confidence)
        
        # Get detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': class_name
                })
        
        # Determine which image to annotate
        if enhanced_image is not None and not show_original:
            annotated_image = results[0].plot()
        else:
            # Draw detections on the original image
            annotated_image = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                class_name = det['class_name']
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_image, detections
    
    def detect_video(self, video_path=None, camera_id=0, output_path=None, 
                    enhancer=None, use_enhanced=True, show_fps=True):
        """
        Perform real-time object detection on video or camera feed.
        
        Args:
            video_path (str, optional): Path to video file. If None, camera is used.
            camera_id (int): Camera ID to use if video_path is None. Default is 0.
            output_path (str, optional): Path to save output video. If None, video is not saved.
            enhancer (ImageEnhancer, optional): Image enhancer object for preprocessing.
            use_enhanced (bool): Whether to use enhanced images for detection. Default is True.
            show_fps (bool): Whether to display FPS on the output. Default is True.
        
        Returns:
            None
        """
        # Open video capture
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(camera_id)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if output path is provided
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps_display = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance image if enhancer is provided
            if enhancer is not None and use_enhanced:
                enhanced_frame = enhancer.enhance_image(frame)
                # Perform detection on enhanced frame
                annotated_frame, detections = self.detect(frame, enhanced_frame, show_original=True)
            else:
                # Perform detection on original frame
                annotated_frame, detections = self.detect(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 10:  # Update FPS every 10 frames
                end_time = time.time()
                fps_display = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            if show_fps:
                cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display detection count
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video if specified
            if output_path is not None:
                out.write(annotated_frame)
            
            # Display the frame
            cv2.imshow('Night-Time Object Detection', annotated_frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if output_path is not None:
            out.release()
        cv2.destroyAllWindows()
    
    def fine_tune(self, data_yaml_path, epochs=50, batch_size=16, img_size=640):
        """
        Fine-tune the YOLOv8 model on a custom dataset.
        
        Args:
            data_yaml_path (str): Path to the YAML file containing dataset configuration.
            epochs (int): Number of training epochs. Default is 50.
            batch_size (int): Batch size for training. Default is 16.
            img_size (int): Image size for training. Default is 640.
        
        Returns:
            None
        """
        # Fine-tune the model
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=10,
            save=True
        )
        
        # Update the model to use the fine-tuned weights
        best_weights = Path('runs/detect/train/weights/best.pt')
        if best_weights.exists():
            self.model = YOLO(best_weights)
            print(f"Model updated with fine-tuned weights: {best_weights}")
        else:
            print("Fine-tuning completed but best weights file not found.")
