import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np
import cv2
import torch
import numpy as np
import cv2

def get_mask_gt(gt_bboxes, featmap_sizes, featmap_strides):
    '''
    intput:
    gt_bboxes: [batch, num_pred, 5], 5 means (cx, cy, w, h, a), a \in [-pi/2, pi/2]
    featmap_sizes: Sequence[tensor[H, W]], len(seq)=num_levels
    featmap_strides: Sequence[tensor[int]], len(seq)=num_levels
    
    output:
    mask_gt: list([batch, 1, H, W]), len(seq)=num_levels
    
    将gt_bboxes转成实例分割的mask，输出的mask的大小为[H, W]，如果特征图上的点对应有物体，则mask上的点为1，否则为0
    '''
    batch_size, num_pred, _ = gt_bboxes.shape
    device = gt_bboxes.device
    num_levels = len(featmap_sizes)
    
    mask_gt = []
    
    for level, (featmap_size, stride) in enumerate(zip(featmap_sizes, featmap_strides)):
        H, W = featmap_size
        mask_level = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.uint8)
        scale_factor = torch.tensor([1/stride, 1/stride, 1/stride, 1/stride, 1], device=device)
        scaled_bboxes = gt_bboxes * scale_factor[None, None, :]
        
        for b in range(batch_size):
            for n in range(num_pred):
                x, y, w, h, angle = scaled_bboxes[b, n].cpu().numpy()
                
                angle_deg = np.degrees(angle)
                
                rect = ((x, y), (w, h), angle_deg)
                box = cv2.boxPoints(rect)
                
                box = np.intp(box)  # 改用 np.intp
                
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [box], 1)
                
                mask_level[b, 0] = mask_level[b, 0] | torch.from_numpy(mask).to(device)
        mask_gt.append(mask_level)

    mask_gt = [mask.float() for mask in mask_gt]
    return mask_gt

import matplotlib.pyplot as plt

def test_get_mask_gt():
    batch_size = 2
    num_pred = 3
    gt_bboxes = torch.tensor([
        [[50, 50, 30, 20, 0],
         [70, 70, 25, 35, np.pi/4],
         [40, 80, 40, 25, -np.pi/6]],
        [[60, 40, 35, 28, np.pi/3],
         [80, 60, 30, 30, -np.pi/4],
         [30, 70, 22, 38, 0]]
    ], dtype=torch.float32)

    featmap_sizes = [torch.Size([100, 200]), torch.Size([25, 50])]
    featmap_strides = [torch.tensor([1]), torch.tensor([4])]

    mask_gt = get_mask_gt(gt_bboxes, featmap_sizes, featmap_strides)
    assert len(mask_gt) == len(featmap_sizes)
    # mask 中最大数是1
    # mask 中最小数是0
    # mask有多个1
    assert all([mask.max() == 1 for mask in mask_gt])
    assert all([mask.min() == 0 for mask in mask_gt])
    assert all([mask.sum() > 0 for mask in mask_gt])
    # 数据类型
    print(mask_gt[0].dtype)
    # 可视化mask
    num_levels = len(mask_gt)
    fig, axes = plt.subplots(batch_size, num_levels, figsize=(num_levels * 10, batch_size * 5))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for level in range(num_levels):
        masks = mask_gt[level]
        for b in range(batch_size):
            ax = axes[b, level]
            ax.imshow(masks[b, 0].cpu().numpy())
            ax.set_title(f'batch {b}, level {level}')
    
    
    plt.tight_layout()
    plt.savefig('masks_visualization.png')
    plt.close()

test_get_mask_gt()
