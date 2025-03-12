import os
import sys
from PIL import Image
import numpy as np

def process_detection(det_path):
    """
    读取检测结果图像，假设原图尺寸为2048×524（左边为GT，右边为检测结果），
    裁剪左半部分后缩放为517×517。
    """
    det_img = Image.open(det_path)
    width, height = det_img.size  # (2048,524)  注意PIL的size为(宽,高)
    # 裁剪左半部分
    cropped = det_img.crop((0, 0, width // 2, height))
    # 缩放到517×517
    det_resized = cropped.resize((517, 517), Image.BILINEAR)
    return det_resized

def process_mask(mask_path):
    """
    读取GTMask或生成的Mask图像，假设尺寸为384×128（宽×高），
    横向均分为三个128×128的小块，每个小块每个像素横向、纵向重复2次放大为256×256，
    返回三个放大后的Image对象。
    """
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)
    H, W = mask_arr.shape[:2]
    # 水平方向分为3份，注意原始尺寸为384×128, 故每份宽度应为128
    piece_width = W // 3
    pieces = []
    for i in range(3):
        # 取出第i块  (所有行，指定列)
        piece = mask_arr[:, i * piece_width:(i + 1) * piece_width]
        # 重复放大：每个像素横向、纵向各重复2次
        piece_up = np.repeat(np.repeat(piece, 2, axis=0), 2, axis=1)
        pieces.append(Image.fromarray(piece_up))
    return pieces

def main(input_dir):
    files = os.listdir(input_dir)
    groups = {}
    
    # 分组依据：以"___0"为界（将"___0"及其之前的部分作为组的标识）
    for f in files:
        # 只处理png文件，并排除combined_featuremap文件
        if not f.endswith(".png") or "combined_featuremap" in f:
            continue
        if "___0" not in f:
            continue
        base = f.split("___0")[0] + "___0"
        if base not in groups:
            groups[base] = {}
        
        # 检测结果文件：要求文件名中出现两次"png"，且排除包含"gt_mask"以及"_mask_0.png"（生成mask）的文件
        if f.count("png") == 2 and ("gt_mask" not in f) and ("_mask_0.png" not in f):
            groups[base]['det'] = os.path.join(input_dir, f)
        # GT Mask文件
        elif "_gt_mask_0.png" in f:
            groups[base]['gt_mask'] = os.path.join(input_dir, f)
        # 生成的 Mask 文件（排除包含"gt_mask"）
        elif "_mask_0.png" in f and "gt_mask" not in f:
            groups[base]['gen_mask'] = os.path.join(input_dir, f)
    
    if not groups:
        print("未在目录中匹配到任何符合条件的文件。")
        return

    # 创建输出文件夹
    out_dir = os.path.join(input_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # 对每个组进行处理（只处理同时包含检测结果、GTMask和生成Mask的组）
    for base, paths in groups.items():
        if 'det' not in paths or 'gt_mask' not in paths or 'gen_mask' not in paths:
            print(f"跳过组 {base}，因为缺少必需的文件。")
            continue
        
        try:
            # 处理检测结果
            det_img = process_detection(paths['det'])
            
            # 处理GT Mask（分割后放大）
            gt_pieces = process_mask(paths['gt_mask'])
            
            # 处理生成的 Mask
            gen_pieces = process_mask(paths['gen_mask'])
        except Exception as e:
            print(f"处理组 {base} 时发生错误: {e}")
            continue

        # 创建最终合成画布：
        # 整体宽度 = 517 (检测结果) + 5 + 256 + 5 + 256 + 5 + 256 = 1300
        # 整体高度 = 517
        composite = Image.new('RGB', (1300, 517), color=(255, 255, 255))
        # 粘贴检测结果图：放在最左侧，占据整个左边区域 (0,0)
        composite.paste(det_img, (0, 0))
        
        # 右侧起始横坐标（检测结果图右侧留5像素）
        x0 = 517 + 5
        # GT Mask 在上排，起始纵坐标为0；生成的 Mask 在下排，起始纵坐标为256+5=266
        y_top = 0
        y_bottom = 256 + 5
        
        # 每张mask的宽度为256，水平间隔5像素
        for i in range(3):
            x = x0 + i * (256 + 5)
            composite.paste(gt_pieces[i], (x, y_top))
            composite.paste(gen_pieces[i], (x, y_bottom))
        
        # 输出文件命名为 <base>_composite.jpg
        out_path = os.path.join(out_dir, f"{base}_composite.jpg")
        composite.save(out_path, "JPEG")
        print(f"保存组 {base} 的合成图像到: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python script.py <图片所在目录>")
    else:
        main(sys.argv[1])
