from mmengine.hooks import Hook
from mmengine.visualization import Visualizer
from ..utils.runner_manager import GlobalClass
import cv2
import numpy as np
import torch
from typing import Optional, Union, List, Tuple
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

class GTMaskVisHook(Hook):
    """Hook to visualize ground truth masks during validation.
    
    This hook generates ground truth masks from bounding boxes and visualizes them
    with white regions for objects and black regions for background.
    """
    
    def __init__(self, num_classes=None):
        self.visualizer = None
        self.featmap_strides = [8, 16, 32]
        self.featmap_sizes = [(1024//8, 1024//8), (1024//16, 1024//16), (1024//32, 1024//32)]
        self.num_classes = num_classes
    
    def before_val_iter(self,
                      runner,
                      batch_idx: int,
                      data_batch: DATA_BATCH = None) -> None:
        """Process before each validation iteration.
        
        Args:
            runner: Runner
            batch_idx: Batch index
            data_batch: Data batch
        """
        self._process_batch(runner, batch_idx, data_batch)
    
    def before_test_iter(self,
                      runner,
                      batch_idx: int,
                      data_batch: DATA_BATCH = None) -> None:
        """Process before each test iteration.
        
        Args:
            runner: Runner
            batch_idx: Batch index
            data_batch: Data batch
        """
        self._process_batch(runner, batch_idx, data_batch)
    
    
    def get_mask_gt(self, gt_bboxes, gt_labels):
        '''
        将gt_bboxes转成实例分割的mask，输出的mask的大小为[num_words, H, W]，如果特征图上的点对应有物体，则对应word的mask上的点为1，否则为0
        Args
            gt_bboxes: [batch, num_pred, 5], 5 means (cx, cy, w, h, a), a \in [-pi/2, pi/2]
            gt_labels: [batch, num_pred, 1], 1 means class id
            featmap_sizes: Sequence[tensor[H, W]], len(seq)=num_levels
            featmap_strides: Sequence[tensor[int]], len(seq)=num_levels
            is_split_attn: bool, 决定gt类型
            num_classes: int, 类别数
        Returns:
            mask_gt:
                if is_split_attn=False, list([batch, 1, H, W]), len(seq)=num_levels
                if is_split_attn=True, list([batch, num_words, H, W]), len(seq)=num_levels
            
        '''
        batch_size, num_pred, _ = gt_bboxes.shape
        device = gt_bboxes.device
        num_levels = len(self.featmap_sizes)
        
        mask_gt = []
        
        for level, (featmap_size, stride) in enumerate(zip(self.featmap_sizes, self.featmap_strides)):
            H, W = featmap_size
            mask_level = torch.zeros((batch_size, self.num_classes, H, W), device=device, dtype=torch.uint8)
            
            scale_factor = torch.tensor([1/stride, 1/stride, 1/stride, 1/stride, 1], device=device)
            scaled_bboxes = gt_bboxes * scale_factor[None, None, :]
            
            for b in range(batch_size):
                for n in range(num_pred):
                    x, y, w, h, angle = scaled_bboxes[b, n].cpu().numpy()
                    if w>0 and h>0:
                        class_id = gt_labels[b, n].item()
                        class_id = int(round(class_id))
                        angle_deg = np.degrees(angle)
                        
                        rect = ((x, y), (w, h), angle_deg)
                        box = cv2.boxPoints(rect)
                        
                        box = np.intp(box)
                        
                        mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask, [box], 1)
                        
                        mask_level[b, class_id] = mask_level[b, class_id] | torch.from_numpy(mask).to(device)
            
            mask_gt.append(mask_level)

        mask_gt = [mask.float() for mask in mask_gt]
        return mask_gt

    def _process_batch(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None) -> None:
        """Common processing for both validation and test iterations.
        
        Args:
            runner: Runner
            batch_idx: Batch index
            data_batch: Data batch
        """
        if self.visualizer is None:
            self.visualizer = Visualizer.get_current_instance()
        
        # Get model
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Get bbox_head
        if not hasattr(model, 'bbox_head'):
            return
        
        bbox_head = model.bbox_head
        
        # Get ground truth bounding boxes from data_batch
        if 'data_samples' not in data_batch or len(data_batch['data_samples']) == 0:
            return
            
        data_sample = data_batch['data_samples'][0]
        if not hasattr(data_sample, 'gt_instances'):
            return
            
        gt_instances = data_sample.gt_instances
        if not hasattr(gt_instances, 'bboxes') or len(gt_instances.bboxes) == 0:
            return
            
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        
        # Generate GT masks using the model's get_mask_gt method
        try:
            # 使用模型自带的get_mask_gt方法生成GT masks
            # 这个方法已经正确处理了旋转框
            gt_masks = self._generate_gt_masks(bbox_head, gt_bboxes, gt_labels)
            
            # Visualize GT masks
            self._visualize_gt_masks(gt_masks)
        except Exception as e:
            # If there's an error, print it but don't crash
            print(f"Error generating GT masks: {e}")
    
    def _generate_gt_masks(self, bbox_head, gt_bboxes, gt_labels):
        """Generate ground truth masks using the model's get_mask_gt method.
        
        Args:
            bbox_head: The model's bbox_head
            gt_bboxes: Ground truth bounding boxes
            gt_labels: Ground truth labels
            
        Returns:
            List of masks for each level
        """
        # 处理gt_bboxes和gt_labels的格式
        if hasattr(gt_bboxes, 'tensor'):
            # 如果gt_bboxes是RotatedBoxes对象，获取其tensor
            gt_bboxes_tensor = gt_bboxes.tensor
        else:
            # 如果gt_bboxes已经是tensor
            gt_bboxes_tensor = gt_bboxes
        
        # 确保gt_bboxes是3D tensor [batch_size, num_boxes, 5]
        if gt_bboxes_tensor.dim() == 2:
            gt_bboxes_tensor = gt_bboxes_tensor.unsqueeze(0)
        
        # 确保gt_labels是3D tensor [batch_size, num_boxes, 1]
        if gt_labels.dim() == 1:
            gt_labels = gt_labels.unsqueeze(0).unsqueeze(-1)
        elif gt_labels.dim() == 2:
            gt_labels = gt_labels.unsqueeze(-1)
        
        # 调用模型的get_mask_gt方法
        return self.get_mask_gt(gt_bboxes_tensor, gt_labels)
    
    
    
    def _visualize_gt_masks(self, masks: List[torch.Tensor]) -> None:
        """Visualize ground truth masks.
        
        Args:
            masks: List of masks, each with shape [batch_size, 1, H, W]
        """
        # Get visualizer
        _visualizer = self.visualizer
        
        # Assert batch size is 1
        assert masks[0].shape[0] == 1
        
        # Create heatmaps for each scale
        heatmaps = []
        max_size = max(mask.shape[2] for mask in masks)  # Find the maximum size
        
        for i in range(len(masks)):
            mask = masks[i]
            # 确保mask是单通道的 [batch_size, 1, H, W]
            if mask.shape[1] > 1:
                # 如果是多通道的，合并为单通道
                mask = mask.sum(dim=1, keepdim=True).clamp(0, 1)
            
            mask_img = mask[0, 0].cpu().numpy()
            # Convert to binary mask (0 or 1)
            mask_img_uint8 = (mask_img * 255).astype(np.uint8)
            repeat_factor = max_size // mask.shape[2]
            
            heatmap_resized = repeat_elements(mask_img_uint8, repeat_factor)
            
            # For GT masks, we want white for objects and black for background
            # No need for colormap, just grayscale
            heatmap = np.stack([heatmap_resized] * 3, axis=-1)  # Convert to RGB
            
            heatmaps.append(heatmap)
        
        # Combine heatmaps horizontally
        combined_heatmap = np.hstack(heatmaps)
        
        # Get image name
        img_name = GlobalClass.get_instance('img_name').value
        img_name = img_name.split('.')[0]
        img_name = img_name + '_gt_mask'
        
        # Add image to visualizer
        _visualizer.add_image(img_name, combined_heatmap) 
