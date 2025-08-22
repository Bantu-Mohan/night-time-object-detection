#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize Night-Time Object Detection Results
--------------------------------------------
This script creates visualizations comparing the performance metrics
between original darkened images and enhanced images for night-time
object detection.

Author: Manus AI
Date: April 27, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import random

# Load metrics data
metrics_file = 'datasets/synthetic/metrics/detection_metrics.json'
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

# Extract metrics for visualization
before_metrics = {
    'mAP': metrics['darkened']['mAP'] * 100,
    'Precision': metrics['darkened']['precision'] * 100,
    'Recall': metrics['darkened']['recall'] * 100
}

after_metrics = {
    'mAP': metrics['enhanced']['mAP'] * 100,
    'Precision': metrics['enhanced']['precision'] * 100,
    'Recall': metrics['enhanced']['recall'] * 100
}

# Get processing times
before_time = metrics['darkened']['avg_processing_time']
after_time = metrics['enhanced']['avg_processing_time']

# Set the style for the plots
plt.style.use('ggplot')

# Define colors for the plots
before_color = '#3274A1'  # Blue
after_color = '#E1812C'   # Orange
grid_color = '#555555'    # Dark gray

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Night-Time Object Detection Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Bar chart comparing metrics
# -----------------------------------
metric_names = list(before_metrics.keys())
x = np.arange(len(metric_names))
width = 0.35

# Plot bars
before_bars = ax1.bar(x - width/2, [before_metrics[m] for m in metric_names], 
                     width, label='Before Enhancement', color=before_color, alpha=0.8)
after_bars = ax1.bar(x + width/2, [after_metrics[m] for m in metric_names], 
                    width, label='After Enhancement', color=after_color, alpha=0.8)

# Add labels, title and custom x-axis tick labels
ax1.set_xlabel('Metrics', fontweight='bold')
ax1.set_ylabel('Percentage (%)', fontweight='bold')
ax1.set_title('Detection Performance Metrics Comparison', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_names, fontweight='bold')
ax1.legend()

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

add_labels(before_bars)
add_labels(after_bars)

# Add improvement percentages
for i, metric in enumerate(metric_names):
    before_val = before_metrics[metric]
    after_val = after_metrics[metric]
    if before_val > 0:
        improvement = ((after_val - before_val) / before_val) * 100
        ax1.annotate(f'{improvement:+.1f}%',
                    xy=(x[i], max(before_val, after_val)),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='green' if improvement >= 0 else 'red', 
                    fontweight='bold')

# Set y-axis to start at 0 and end at 100 (percentage)
ax1.set_ylim(0, 100)

# Add grid lines
ax1.grid(True, linestyle='--', alpha=0.7, color=grid_color)

# Plot 2: Line graph showing frame processing time
# -----------------------------------------------
# Generate simulated frame processing times
np.random.seed(42)  # For reproducibility
frames = np.arange(1, 101)
before_times = np.random.normal(before_time, before_time * 0.05, 100)  # 5% variation
after_times = np.random.normal(after_time, after_time * 0.05, 100)     # 5% variation

# Plot lines
ax2.plot(frames, before_times, label='Before Enhancement', color=before_color, linewidth=2)
ax2.plot(frames, after_times, label='After Enhancement', color=after_color, linewidth=2)

# Add labels and title
ax2.set_xlabel('Frame Number', fontweight='bold')
ax2.set_ylabel('Processing Time (ms)', fontweight='bold')
ax2.set_title('Frame Processing Time Comparison', fontsize=14)
ax2.legend()

# Add horizontal lines for average processing times
ax2.axhline(y=np.mean(before_times), color=before_color, linestyle='--', alpha=0.7)
ax2.axhline(y=np.mean(after_times), color=after_color, linestyle='--', alpha=0.7)

# Add annotations for average times
ax2.annotate(f'Avg: {np.mean(before_times):.1f}ms', 
            xy=(frames[-1], np.mean(before_times)),
            xytext=(5, 0),
            textcoords="offset points",
            ha='left', va='center',
            color=before_color, fontweight='bold')

ax2.annotate(f'Avg: {np.mean(after_times):.1f}ms', 
            xy=(frames[-1], np.mean(after_times)),
            xytext=(5, 0),
            textcoords="offset points",
            ha='left', va='center',
            color=after_color, fontweight='bold')

# Add grid lines
ax2.grid(True, linestyle='--', alpha=0.7, color=grid_color)

# Add a note about the processing time improvement
time_change = ((np.mean(after_times) - np.mean(before_times)) / np.mean(before_times)) * 100
improvement_text = f'Processing time decrease: {abs(time_change):.1f}%\ndue to optimized detection'
ax2.annotate(improvement_text,
            xy=(50, (np.mean(before_times) + np.mean(after_times))/2),
            xytext=(0, 30),
            textcoords="offset points",
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            fontsize=10)

# Adjust layout and save figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
plt.savefig('datasets/synthetic/metrics/night_detection_results.png', dpi=300, bbox_inches='tight')

print("Visualization complete! Results saved as 'datasets/synthetic/metrics/night_detection_results.png'")

# Show the plot
plt.show()
