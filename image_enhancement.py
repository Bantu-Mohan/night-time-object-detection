#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Enhancement Pipeline for Night-Time Object Detection
---------------------------------------------------------
This module implements various image enhancement techniques to improve
visibility in night-time or low-light images, making them more suitable
for object detection tasks.

Techniques implemented:
1. Gamma Correction
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Denoising (Bilateral Filter)

Author: Manus AI
Date: April 27, 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageEnhancer:
    """
    A class that provides methods for enhancing night-time or low-light images.
    """
    
    def __init__(self):
        """Initialize the ImageEnhancer class."""
        pass
    
    def adjust_gamma(self, image, gamma=1.5):
        """
        Apply gamma correction to the input image.
        
        Args:
            image (numpy.ndarray): Input image.
            gamma (float): Gamma value. Values > 1 will brighten the image,
                          values < 1 will darken it. Default is 1.5.
        
        Returns:
            numpy.ndarray: Gamma-corrected image.
        """
        # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.
        
        Args:
            image (numpy.ndarray): Input image.
            clip_limit (float): Threshold for contrast limiting. Default is 2.0.
            tile_grid_size (tuple): Size of grid for histogram equalization. Default is (8, 8).
        
        Returns:
            numpy.ndarray: CLAHE-enhanced image.
        """
        # Convert to LAB color space for better results with CLAHE
        if len(image.shape) == 3:  # Color image
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            
            # Merge the CLAHE enhanced L channel with the original A and B channels
            enhanced_lab = cv2.merge((cl, a, b))
            
            # Convert back to BGR color space
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_image
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
    
    def apply_denoising(self, image, d=15, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        
        Args:
            image (numpy.ndarray): Input image.
            d (int): Diameter of each pixel neighborhood. Default is 15.
            sigma_color (float): Filter sigma in the color space. Default is 75.
            sigma_space (float): Filter sigma in the coordinate space. Default is 75.
        
        Returns:
            numpy.ndarray: Denoised image.
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def enhance_image(self, image, gamma=1.5, apply_clahe=True, apply_denoise=True,
                     clahe_clip_limit=2.0, clahe_grid_size=(8, 8),
                     denoise_d=15, denoise_sigma_color=75, denoise_sigma_space=75):
        """
        Apply a complete enhancement pipeline to the input image.
        
        Args:
            image (numpy.ndarray): Input image.
            gamma (float): Gamma value for correction. Default is 1.5.
            apply_clahe (bool): Whether to apply CLAHE. Default is True.
            apply_denoise (bool): Whether to apply denoising. Default is True.
            clahe_clip_limit (float): Clip limit for CLAHE. Default is 2.0.
            clahe_grid_size (tuple): Grid size for CLAHE. Default is (8, 8).
            denoise_d (int): Diameter for bilateral filter. Default is 15.
            denoise_sigma_color (float): Color sigma for bilateral filter. Default is 75.
            denoise_sigma_space (float): Space sigma for bilateral filter. Default is 75.
        
        Returns:
            numpy.ndarray: Enhanced image.
        """
        # Apply gamma correction
        enhanced = self.adjust_gamma(image, gamma)
        
        # Apply CLAHE if requested
        if apply_clahe:
            enhanced = self.apply_clahe(enhanced, clahe_clip_limit, clahe_grid_size)
        
        # Apply denoising if requested
        if apply_denoise:
            enhanced = self.apply_denoising(enhanced, denoise_d, denoise_sigma_color, denoise_sigma_space)
        
        return enhanced
    
    def visualize_enhancement(self, original, enhanced, save_path=None):
        """
        Visualize the original and enhanced images side by side.
        
        Args:
            original (numpy.ndarray): Original image.
            enhanced (numpy.ndarray): Enhanced image.
            save_path (str, optional): Path to save the visualization. If None, the visualization
                                      is displayed but not saved. Default is None.
        """
        # Convert BGR to RGB for matplotlib
        if len(original.shape) == 3:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = original
            enhanced_rgb = enhanced
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display enhanced image
        axes[1].imshow(enhanced_rgb)
        axes[1].set_title('Enhanced Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Create an instance of the ImageEnhancer class
    enhancer = ImageEnhancer()
    
    # Load a sample image (this is just a placeholder, will need a real image)
    # image_path = "path/to/night_image.jpg"
    # if os.path.exists(image_path):
    #     image = cv2.imread(image_path)
    #     
    #     # Apply the enhancement pipeline
    #     enhanced_image = enhancer.enhance_image(image)
    #     
    #     # Visualize the results
    #     enhancer.visualize_enhancement(image, enhanced_image, "enhanced_comparison.png")
    # else:
    #     print(f"Image not found at {image_path}")
