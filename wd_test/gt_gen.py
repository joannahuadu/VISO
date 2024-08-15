import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_mask_gt(self, gt_bboxes, featmap_sizes, featmap_strides):
    '''
    input:
    gt_bboxes: [batch, num_pred, 4] xyxy
    featmap_sizes: Sequence[tuple(H, W)], len(seq)=num_levels
    featmap_strides: Sequence[int], len(seq)=num_levels

    output:
    mask_gt: list([batch, 1, H, W]), len(list)=num_levels
    '''
    batch_size, num_pred, _ = gt_bboxes.shape
    device = gt_bboxes.device
    num_levels = len(featmap_sizes)

    mask_gt = []
    for level, (featmap_size, stride) in enumerate(zip(featmap_sizes, featmap_strides)):
        H, W = featmap_size
        mask_level = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.uint8)
        scale_factor = torch.tensor([1/stride, 1/stride, 1/stride, 1/stride], device=device)
        scaled_bboxes = gt_bboxes * scale_factor[None, None, :]

        for b in range(batch_size):
            for n in range(num_pred):
                x1, y1, x2, y2 = scaled_bboxes[b, n].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                mask_level[b, 0] |= torch.from_numpy(mask).to(device)

        mask_gt.append(mask_level)

    mask_gt = [mask.float() for mask in mask_gt]
    return mask_gt

def test_get_mask_gt():
    # 创建固定的测试数据
    batch_size = 2
    num_pred = 3
    num_levels = 3

    # 固定的gt_bboxes
    gt_bboxes = torch.tensor([
        [  # Batch 1
            [10, 15, 50, 50],  # Box 1
            [30, 30, 70, 70],  # Box 2
            [80, 80, 120, 120]  # Box 3
        ],
        [  # Batch 2
            [20, 20, 60, 60],  # Box 1
            [40, 40, 80, 80],  # Box 2
            [100, 100, 140, 140]  # Box 3
        ]
    ], dtype=torch.float32)

    # 创建featmap_sizes和featmap_strides
    featmap_sizes = [(64, 128), (32, 64), (16, 32)]
    featmap_strides = [1, 2, 4]

    # 调用get_mask_gt函数
    mask_gt = get_mask_gt(None, gt_bboxes, featmap_sizes, featmap_strides)

    # 验证输出
    assert len(mask_gt) == num_levels, "输出的mask数量应该等于特征图的数量"
    for level, (H, W) in enumerate(featmap_sizes):
        assert mask_gt[level].shape == (batch_size, 1, H, W), f"Level {level} 的mask形状不正确"

    # 可视化结果
    fig, axes = plt.subplots(batch_size, num_levels, figsize=(15, 10))
    for b in range(batch_size):
        for l in range(num_levels):
            ax = axes[b, l] if batch_size > 1 else axes[l]
            mask = mask_gt[l][b, 0].cpu().numpy()
            ax.imshow(mask, cmap='gray')
            ax.set_title(f'Batch {b}, Level {l}')
            
            # 添加边界框
            stride = featmap_strides[l]
            for bbox in gt_bboxes[b]:
                x1, y1, x2, y2 = (bbox / stride).int().tolist()
                rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red')
                ax.add_patch(rect)
            
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('mask_gt_visualization.png')
    plt.close()

    print("测试完成，可视化结果已保存为 'mask_gt_visualization.png'")

# 运行测试
test_get_mask_gt()
