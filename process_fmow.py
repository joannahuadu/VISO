import os
import json
import shutil

def process_directories(root_dir, output_img_dir, output_txt_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    
    # 遍历 root_dir 下的所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if file.endswith("_rgb.jpg"):
                # 移动图像文件
                shutil.copy(filepath, os.path.join(output_img_dir, file))
            elif file.endswith("_rgb.json"):
                # 处理 JSON 文件并转换为 TXT
                with open(filepath, 'r') as json_file:
                    data = json.load(json_file)
                    txt_filename = os.path.splitext(file)[0] + ".txt"
                    txt_filepath = os.path.join(output_txt_dir, txt_filename)
                    with open(txt_filepath, 'w') as txt_file:
                        for box in data["bounding_boxes"]:
                            a, b, c, d = box["box"]
                            category = box["category"]
                            txt_file.write(f"{a} {b} {c} {d} {category} {0}\n")

root_directory = "/mnt/data1/workspace/wmq/YOLO-World/data/fMoW/val/port"  # 根目录路径
output_images_directory = "/mnt/data1/workspace/wmq/YOLO-World/data/split_fMoW/val/port/images"  # 输出图像文件夹路径
output_texts_directory = "/mnt/data1/workspace/wmq/YOLO-World/data/split_fMoW/val/port/annfiles"  # 输出文本文件夹路径

process_directories(root_directory, output_images_directory, output_texts_directory)
