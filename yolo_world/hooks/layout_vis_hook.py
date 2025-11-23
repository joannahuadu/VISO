from mmengine.hooks import Hook
from mmengine.visualization import Visualizer
from ..utils.runner_manager import GlobalClass
import cv2
import numpy as np
import torch
import os
import os.path as osp
from typing import Optional, Union, List, Tuple, Dict
from mmengine.utils import mkdir_or_exist

DATA_BATCH = Optional[Union[dict, tuple, list]]

def repeat_elements(arr, repeat_factor):
    """Repeat elements in a 2D array
    Args:
        arr: Input 2D array
        repeat_factor: Repeat factor
    Returns:
        Repeated 2D array
    """
    return np.repeat(np.repeat(arr, repeat_factor, axis=0), repeat_factor, axis=1)

class LayoutVisHook(Hook):
    """Hook to create a combined visualization layout with detection results, GTMask, and generated Mask.
    
    The layout is as follows:
    11 2 3 4
    11 5 6 7
    
    Where:
    11 - Detection results (takes up 2x1 space)
    2,3,4 - GTMask images with decreasing resolution from left to right
    5,6,7 - Generated Mask images with decreasing resolution from left to right
    
    Args:
        output_dir (str): Directory to save the combined visualization
        margin (int): Margin between images in pixels
        max_size (int, optional): Maximum size for the largest resolution image. If None, 
                                 will use the original size.
    """
    
    def __init__(self, output_dir, margin=10, max_size=None):
        self.visualizer = None
        self.output_dir = output_dir
        self.margin = margin
        self.max_size = max_size
        mkdir_or_exist(output_dir)
        print(f"LayoutVisHook initialized with output_dir: {output_dir}")
    
    def _find_image_files(self, storage_dir, img_prefix):
        """查找与给定前缀匹配的图像文件。
        
        Args:
            storage_dir: 存储目录
            img_prefix: 图像前缀
            
        Returns:
            gt_mask_path: GT掩码图像路径
            gen_mask_path: 生成的掩码图像路径
        """
        gt_mask_path = None
        gen_mask_path = None
        
        # 列出所有文件
        all_files = os.listdir(storage_dir)
        print(f"Total files in directory: {len(all_files)}")
        
        # 尝试不同的匹配策略
        # 1. 精确匹配
        for filename in all_files:
            if img_prefix in filename and "_gt_mask" in filename and "multi_channel" not in filename:
                gt_mask_path = osp.join(storage_dir, filename)
                print(f"Found GT mask (strategy 1): {filename}")
            elif img_prefix in filename and "_mask" in filename and "multi_channel" not in filename and "gt_mask" not in filename:
                gen_mask_path = osp.join(storage_dir, filename)
                print(f"Found generated mask (strategy 1): {filename}")
        
        # 2. 如果没找到，尝试更宽松的匹配
        if gt_mask_path is None or gen_mask_path is None:
            # 提取更短的前缀
            short_prefix = img_prefix
            if '___' in short_prefix:
                short_prefix = short_prefix.split('___')[0]
            
            print(f"Trying with shorter prefix: {short_prefix}")
            
            for filename in all_files:
                if gt_mask_path is None and short_prefix in filename and "_gt_mask" in filename and "multi_channel" not in filename:
                    gt_mask_path = osp.join(storage_dir, filename)
                    print(f"Found GT mask (strategy 2): {filename}")
                elif gen_mask_path is None and short_prefix in filename and "_mask" in filename and "multi_channel" not in filename and "gt_mask" not in filename:
                    gen_mask_path = osp.join(storage_dir, filename)
                    print(f"Found generated mask (strategy 2): {filename}")
        
        # 3. 如果还没找到，尝试最宽松的匹配
        if gt_mask_path is None:
            print("Trying to find any GT mask file")
            for filename in all_files:
                if "_gt_mask" in filename and "multi_channel" not in filename:
                    gt_mask_path = osp.join(storage_dir, filename)
                    print(f"Found GT mask (strategy 3): {filename}")
                    break
        
        if gen_mask_path is None:
            print("Trying to find any generated mask file")
            for filename in all_files:
                if "_mask" in filename and "multi_channel" not in filename and "gt_mask" not in filename:
                    gen_mask_path = osp.join(storage_dir, filename)
                    print(f"Found generated mask (strategy 3): {filename}")
                    break
        
        return gt_mask_path, gen_mask_path
    
    def _find_detection_result(self, test_out_dir, img_name, img_prefix):
        """查找检测结果图像。
        
        Args:
            test_out_dir: 测试输出目录
            img_name: 图像名称
            img_prefix: 图像前缀
            
        Returns:
            det_result_path: 检测结果图像路径
        """
        # 首先尝试直接匹配
        det_result_path = osp.join(test_out_dir, img_name)
        if osp.exists(det_result_path):
            print(f"Found detection result (direct match): {det_result_path}")
            return det_result_path
        
        # 检查是否存在 vis_image 子目录
        vis_image_dir = osp.join(test_out_dir, 'vis_image')
        if osp.exists(vis_image_dir) and osp.isdir(vis_image_dir):
            print(f"Found vis_image directory: {vis_image_dir}")
            
            # 列出 vis_image 目录中的所有文件
            all_files = os.listdir(vis_image_dir)
            print(f"Total files in vis_image directory: {len(all_files)}")
            
            # 尝试不同的匹配策略
            # 1. 使用完整的图像名称
            for filename in all_files:
                if img_name in filename:
                    det_result_path = osp.join(vis_image_dir, filename)
                    print(f"Found detection result (strategy 1): {filename}")
                    return det_result_path
            
            # 2. 使用图像前缀
            for filename in all_files:
                if img_prefix in filename:
                    det_result_path = osp.join(vis_image_dir, filename)
                    print(f"Found detection result (strategy 2): {filename}")
                    return det_result_path
            
            # 3. 使用更短的前缀
            short_prefix = img_prefix
            if '___' in short_prefix:
                short_prefix = short_prefix.split('___')[0]
            
            print(f"Trying with shorter prefix: {short_prefix}")
            
            for filename in all_files:
                if short_prefix in filename:
                    det_result_path = osp.join(vis_image_dir, filename)
                    print(f"Found detection result (strategy 3): {filename}")
                    return det_result_path
            
            # 4. 如果还没找到，返回第一个文件（如果有的话）
            if all_files:
                det_result_path = osp.join(vis_image_dir, all_files[0])
                print(f"Using first available file as detection result: {all_files[0]}")
                return det_result_path
        
        # 如果在 vis_image 目录中没有找到，尝试在测试输出目录中查找
        all_files = os.listdir(test_out_dir)
        for filename in all_files:
            if img_prefix in filename and osp.isfile(osp.join(test_out_dir, filename)):
                det_result_path = osp.join(test_out_dir, filename)
                print(f"Found detection result in test_out_dir: {filename}")
                return det_result_path
        
        # 如果还是没找到，返回 None
        print(f"No detection result found for {img_name}")
        return None

    def after_test_iter(self,
                      runner,
                      batch_idx: int,
                      data_batch: DATA_BATCH = None,
                      outputs: Optional[dict] = None) -> None:
        """Process after each test iteration.
        
        Args:
            runner: Runner
            batch_idx: Batch index
            data_batch: Data batch
            outputs: Outputs from model
        """
        try:
            if self.visualizer is None:
                self.visualizer = Visualizer.get_current_instance()
            
            # Get image name
            img_name = GlobalClass.get_instance('img_name').value
            print(f"Processing image: {img_name}")
            
            # 从文件名中提取基本部分，去掉扩展名
            img_basename = img_name.split('.')[0]
            
            # 定义用于匹配文件的前缀
            # 文件名格式可能是 P1122__1024__3144___0_gt_mask_0.png
            # 我们需要匹配 P1122__1024__3144___0 这部分
            img_prefix = img_basename
            if '___' in img_basename:
                img_prefix = img_basename.split('___')[0] + '___' + img_basename.split('___')[1]
            
            # Define paths for different visualization results
            test_out_dir = runner.cfg.test_evaluator.outfile_prefix
            print(f"Test output directory: {test_out_dir}")
            
            # 1. 查找检测结果图像
            det_result_path = self._find_detection_result(test_out_dir, img_name, img_prefix)
            if det_result_path is None:
                print("No detection result found, skipping this image")
                return
            
            # Try to find the visualization images in the visualizer's backend storage
            backend = self.visualizer._vis_backends.get('LocalVisBackend')
            if backend is None:
                print("LocalVisBackend not found in visualizer")
                return
                
            # Get the storage directory from the backend
            storage_dir = backend._save_dir
            print(f"Visualization storage directory: {storage_dir}")
            
            # Find GT mask and generated mask images using our helper method
            gt_mask_path, gen_mask_path = self._find_image_files(storage_dir, img_prefix)
            
            if gt_mask_path is None or gen_mask_path is None:
                print(f"GT mask or generated mask image not found for {img_prefix}")
                return
                
            # Load detection result image
            det_img = cv2.imread(det_result_path)
            if det_img is None:
                print(f"Failed to load detection result image: {det_result_path}")
                return
            det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
            print(f"Detection image loaded, shape: {det_img.shape}")
            
            # Load GT mask and generated mask images
            gt_mask_img = cv2.imread(gt_mask_path)
            if gt_mask_img is None:
                print(f"Failed to load GT mask image: {gt_mask_path}")
                return
            gt_mask_img = cv2.cvtColor(gt_mask_img, cv2.COLOR_BGR2RGB)
            print(f"GT mask image loaded, shape: {gt_mask_img.shape}")
            
            gen_mask_img = cv2.imread(gen_mask_path)
            if gen_mask_img is None:
                print(f"Failed to load generated mask image: {gen_mask_path}")
                return
            gen_mask_img = cv2.cvtColor(gen_mask_img, cv2.COLOR_BGR2RGB)
            print(f"Generated mask image loaded, shape: {gen_mask_img.shape}")
            
            # Split the GT mask and generated mask images into three parts (different resolutions)
            gt_masks = self._split_image_horizontally(gt_mask_img, 3)
            gen_masks = self._split_image_horizontally(gen_mask_img, 3)
            
            # Resize all images to have consistent heights
            # Detection result should be twice as tall as the mask images
            if self.max_size is not None:
                mask_height = self.max_size // 2
                # Preserve aspect ratio when resizing detection image
                det_img = self._resize_image(det_img, height=self.max_size)
            else:
                mask_height = gt_masks[0].shape[0]
                # Preserve aspect ratio when resizing detection image
                det_img = self._resize_image(det_img, height=mask_height * 2 + self.margin)
                
            for i in range(3):
                gt_masks[i] = self._resize_image(gt_masks[i], height=mask_height)
                gen_masks[i] = self._resize_image(gen_masks[i], height=mask_height)
            
            # Create the layout
            # 11 2 3 4
            # 11 5 6 7
            layout = self._create_layout(det_img, gt_masks, gen_masks)
            
            # 使用更简洁的文件名保存
            # 去除文件名中可能存在的特殊字符
            clean_img_name = img_basename.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '_').replace(',', '')
            output_path = osp.join(self.output_dir, f"{clean_img_name}_layout.png")
            cv2.imwrite(output_path, cv2.cvtColor(layout, cv2.COLOR_RGB2BGR))
            print(f"Layout saved to: {output_path}")
            
            # Also add to visualizer
            self.visualizer.add_image(f"{clean_img_name}_combined_layout", layout)
            
            print(f"Successfully created layout for {clean_img_name}")
            
        except Exception as e:
            import traceback
            print(f"Error in LayoutVisHook: {e}")
            print(traceback.format_exc())
    
    def _split_image_horizontally(self, image, num_parts):
        """Split an image horizontally into equal parts.
        
        Args:
            image: Input image
            num_parts: Number of parts to split into
            
        Returns:
            List of image parts
        """
        width = image.shape[1] // num_parts
        parts = []
        
        for i in range(num_parts):
            start_x = i * width
            end_x = (i + 1) * width
            parts.append(image[:, start_x:end_x])
            
        return parts
    
    def _resize_image(self, image, height=None, width=None):
        """Resize an image to the specified height or width while maintaining aspect ratio.
        
        Args:
            image: Input image
            height: Target height (optional)
            width: Target width (optional)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if height is not None and width is None:
            # Resize based on height
            aspect_ratio = w / h
            new_width = int(height * aspect_ratio)
            return cv2.resize(image, (new_width, height))
        elif width is not None and height is None:
            # Resize based on width
            aspect_ratio = h / w
            new_height = int(width * aspect_ratio)
            return cv2.resize(image, (width, new_height))
        elif height is not None and width is not None:
            # Resize to exact dimensions
            return cv2.resize(image, (width, height))
        else:
            # No resize
            return image
    
    def _create_layout(self, det_img, gt_masks, gen_masks):
        """Create the layout with the specified arrangement.
        
        Args:
            det_img: Detection result image
            gt_masks: List of GT mask images at different resolutions
            gen_masks: List of generated mask images at different resolutions
            
        Returns:
            Combined layout image
        """
        # Get dimensions
        det_height, det_width = det_img.shape[:2]
        mask_height, mask_widths = gt_masks[0].shape[0], [m.shape[1] for m in gt_masks]
        gen_mask_widths = [m.shape[1] for m in gen_masks]
        
        # Calculate the width needed for the mask images (3 images + 2 margins between them)
        mask_section_width = sum(mask_widths) + self.margin * (len(mask_widths) - 1)
        gen_mask_section_width = sum(gen_mask_widths) + self.margin * (len(gen_mask_widths) - 1)
        
        # Calculate total width and height
        total_width = max(det_width, 0) + self.margin + max(mask_section_width, gen_mask_section_width)
        total_height = mask_height * 2 + self.margin
        
        # Create canvas with white background
        canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place detection result image (position 11 - takes up 2 rows)
        # Make sure it doesn't exceed the canvas height
        det_img_resized = self._resize_image(det_img, height=total_height)
        h, w = det_img_resized.shape[:2]
        canvas[:h, :w] = det_img_resized
        
        # Place GT mask images (positions 2, 3, 4 - top row)
        x_offset = det_width + self.margin
        for i, mask in enumerate(gt_masks):
            h, w = mask.shape[:2]
            if i > 0:
                x_offset += mask_widths[i-1] + self.margin
            canvas[:mask_height, x_offset:x_offset+w] = mask
        
        # Place generated mask images (positions 5, 6, 7 - bottom row)
        x_offset = det_width + self.margin
        y_offset = mask_height + self.margin
        for i, mask in enumerate(gen_masks):
            h, w = mask.shape[:2]
            if i > 0:
                x_offset += gen_mask_widths[i-1] + self.margin
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = mask
        
        # Add borders between images for better visual separation
        # Vertical line after detection image
        cv2.line(canvas, 
                 (det_width, 0), 
                 (det_width, total_height), 
                 (0, 0, 0), 
                 thickness=1)
        
        # Horizontal line between top and bottom rows
        cv2.line(canvas, 
                 (det_width, mask_height), 
                 (total_width, mask_height), 
                 (0, 0, 0), 
                 thickness=1)
        
        # Add labels for better understanding
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)  # Black
        font_thickness = 1
        
        # Label for detection result
        cv2.putText(canvas, 'Detection Result', 
                    (10, 20), 
                    font, font_scale, font_color, font_thickness)
        
        # Labels for GT masks
        x_offset = det_width + self.margin
        for i, mask in enumerate(gt_masks):
            w = mask.shape[1]
            label = f'GT Mask L{i+1}'
            cv2.putText(canvas, label, 
                        (x_offset + 5, 20), 
                        font, font_scale, font_color, font_thickness)
            x_offset += w + self.margin
        
        # Labels for generated masks
        x_offset = det_width + self.margin
        for i, mask in enumerate(gen_masks):
            w = mask.shape[1]
            label = f'Gen Mask L{i+1}'
            cv2.putText(canvas, label, 
                        (x_offset + 5, mask_height + 20), 
                        font, font_scale, font_color, font_thickness)
            x_offset += w + self.margin
        
        return canvas 
