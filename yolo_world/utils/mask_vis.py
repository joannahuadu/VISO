from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer
from .runner_manager import GlobalClass
import cv2
import numpy as np
import torch

def repeat_elements(arr, repeat_factor):
    """重复数组中的每个元素
    Args:
        arr: 输入的二维数组
        repeat_factor: 重复的倍数
    Returns:
        重复后的二维数组
    """
    return np.repeat(np.repeat(arr, repeat_factor, axis=0), repeat_factor, axis=1)

def mask_visulize(masks):
    """将mask调用visualizer画出来，heatmap形式
    Args:
        masks: shape = list(tensor(batchsize, 1, h, w)) len(masks) = 尺度数
    """
    _visualizer: Visualizer = Visualizer.get_current_instance()
    assert masks[0].shape[0] == 1 # 当前没有考虑batchsize > 1的情况
    
    heatmaps = []
    max_size = max(mask.shape[2] for mask in masks)  # 找到最大的尺寸
    for i in range(len(masks)):
        mask = masks[i]
        mask_img = mask[0, 0].cpu().numpy()
        mask_img_uint8 = (mask_img * 255).astype(np.uint8)
        repeat_factor = max_size // mask.shape[2]
        
        heatmap_resized = repeat_elements(mask_img_uint8, repeat_factor)
        
        # 将调整尺寸后的 heatmap 转换为彩色图像
        heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # 使用 JET 颜色映射
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        heatmaps.append(heatmap)
    
    combined_heatmap = np.hstack(heatmaps)
    
    total_curr_iter = GlobalClass.get_instance('batch_idx').value + GlobalClass.get_instance('runner').value.iter + 1
    _visualizer.add_image('combined_mask', combined_heatmap, total_curr_iter)

import torch

def featuremap_visulize(feature_maps):
    """将feature map调用visualizer画出来
    Args:
        feature_maps: shape = list(tensor(batchsize, c, h, w)) len(feature_maps) = 尺度数
    """
    _visualizer: Visualizer = Visualizer.get_current_instance()
    assert feature_maps[0].shape[0] == 1 # 当前没有考虑batchsize > 1的情况
    
    heatmaps = []
    max_size = max(feature_map.shape[2] for feature_map in feature_maps)  # 找到最大的尺寸

    for i in range(len(feature_maps)):
        feature_map = feature_maps[i][0]
        
        resized_feature_map = torch.tensor([repeat_elements(channel.cpu().numpy(), max_size // channel.shape[1]) for channel in feature_map])
        
        image = _visualizer.draw_featmap(resized_feature_map)  # 返回: np.ndarray: RGB image.
        
        heatmaps.append(image)
    
    combined_heatmap = np.hstack(heatmaps)
    
    total_curr_iter = GlobalClass.get_instance('batch_idx').value + GlobalClass.get_instance('runner').value.iter + 1
    _visualizer.add_image('combined_featuremap', combined_heatmap, total_curr_iter)
