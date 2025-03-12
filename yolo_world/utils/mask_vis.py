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
        masks: shape = list(tensor(batchsize, num_words, h, w)) len(masks) = 尺度数
    """
    _visualizer: Visualizer = Visualizer.get_current_instance()
    assert masks[0].shape[0] == 1 # 当前没有考虑batchsize > 1的情况
    if masks[0].shape[1]==1: # 此时是单通道mask，只需要显示三个尺度的mask即可，画在一行
        heatmaps = []
        max_size = max(mask.shape[2] for mask in masks)  # 找到最大的尺寸
        for i in range(len(masks)): # 这里是三个尺度
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
        
        img_name = GlobalClass.get_instance('img_name').value
        img_name = img_name.split('.')[0]
        img_name = img_name + '_mask'
        _visualizer.add_image(img_name, combined_heatmap)
        # # 把mask保存到npy文件中
        # save_path = 'paper_pic/mask_npy'
        # mkdir_or_exist(save_path)
        # # 三个尺度保存，区分加_0,_1,_2，保存到npy
        # for i in range(len(masks)):
        #     np.save(f'{save_path}/{img_name}_{i}.npy', masks[i].cpu().numpy())
            
        
        
        
    # else:  # 此时是多通道mask，需要分别显示每个通道的mask，num_words行，每行三个尺度的mask，最左边标一下通道对应的word
    #     class_texts = GlobalClass.get_instance('class_texts').value  # list(str)
    #     num_words = masks[0].shape[1]
    #     max_size = max(mask.shape[2] for mask in masks)  # 找到最大的尺寸
        
    #     # 创建一个大的画布来容纳所有的heatmap
    #     canvas_height = num_words * max_size
    #     canvas_width = (len(masks) + 1) * max_size  # +1 for the text column
    #     canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
    #     for word_idx in range(num_words):
    #         # 添加文字``
    #         text_img = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    #         cv2.putText(text_img, class_texts[word_idx], (10, max_size // 2),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #         canvas[word_idx * max_size:(word_idx + 1) * max_size, 0:max_size] = text_img
            
    #         for scale_idx, mask in enumerate(masks):
    #             mask_img = mask[0, word_idx].cpu().numpy()
    #             mask_img_uint8 = (mask_img * 255).astype(np.uint8)
    #             repeat_factor = max_size // mask.shape[2]
                
    #             heatmap_resized = repeat_elements(mask_img_uint8, repeat_factor)
                
    #             # 将调整尺寸后的 heatmap 转换为彩色图像
    #             heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # 使用 JET 颜色映射
    #             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
    #             # 将heatmap放入画布中
    #             canvas[word_idx * max_size:(word_idx + 1) * max_size,
    #                    (scale_idx + 1) * max_size:(scale_idx + 2) * max_size] = heatmap
    #     img_name = GlobalClass.get_instance('img_name').value
    #     img_name = img_name.split('.')[0]
    #     _visualizer.add_image(img_name+'multi_channel_mask', canvas)
    else:
        class_texts = GlobalClass.get_instance('class_texts').value  # list(str)
        num_words = masks[0].shape[1]
        max_size = max(mask.shape[2] for mask in masks)  # 找到最大的尺寸

        for word_idx in range(num_words):
            # 为每个类别创建一个新的画布
            canvas_width = len(masks) * max_size
            canvas = np.zeros((max_size, canvas_width, 3), dtype=np.uint8)
            
            for scale_idx, mask in enumerate(masks):
                mask_img = mask[0, word_idx].cpu().numpy()
                mask_img_uint8 = (mask_img * 255).astype(np.uint8)
                repeat_factor = max_size // mask.shape[2]
                
                heatmap_resized = repeat_elements(mask_img_uint8, repeat_factor)
                
                # 将调整尺寸后的 heatmap 转换为彩色图像
                heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # 使用 JET 颜色映射
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # 将heatmap放入画布中
                canvas[0:max_size, scale_idx * max_size:(scale_idx + 1) * max_size] = heatmap
            
            # 获取图像名称
            img_name = GlobalClass.get_instance('img_name').value
            img_name = img_name.split('.')[0]
            
            # 为每个类别调用add_image
            _visualizer.add_image(f'{img_name}_multi_channel_mask_{class_texts[word_idx]}', canvas)

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
    img_name = GlobalClass.get_instance('img_name').value
    img_name = img_name.split('.')[0]
    _visualizer.add_image(img_name+'combined_featuremap', combined_heatmap)

