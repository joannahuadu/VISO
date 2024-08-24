import os

# 设置路径
image_dir = 'data/split_fMoW_1024/samples/wind_farm/images'
ann_dir = 'data/split_fMoW_1024/samples/wind_farm/annfiles'

# 确保 annfiles 目录存在
os.makedirs(ann_dir, exist_ok=True)

# 遍历 images 目录中的所有文件
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 可以根据需要添加其他图片格式
        # 构造对应的 txt 文件名
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(ann_dir, txt_filename)
        
        # 创建空白的 txt 文件
        with open(txt_path, 'w') as f:
            pass  # 不写入任何内容，仅创建文件

print("空白 txt 文件创建完成。")
