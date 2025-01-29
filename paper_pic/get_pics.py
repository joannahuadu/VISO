import os
import shutil

# 定义目标路径
target_dir = "./paper_pic/wildfire"

# 定义子文件夹名称
subfolders = [
    "Sen2Fire_wildfire_or_fire_or_smoke",
    "Sen2Fire_wildfire_or_fire",
    "Sen2Fire_wildfire"
]

# 创建子文件夹
for subfolder in subfolders:
    os.makedirs(os.path.join(target_dir, subfolder), exist_ok=True)

# 定义文件路径列表
file_paths = [
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire_or_smoke/20240826_192109/vis_data/vis_image/scene_3_patch_1_25.png_35.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire_or_smoke/20240826_192109/vis_data/vis_image/scene_3_patch_5_35.png_4.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire_or_smoke/20240826_192109/vis_data/vis_image/scene_3_patch_6_36.png_293.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire_or_smoke/20240826_192109/vis_data/vis_image/scene_3_patch_7_21.png_113.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire/20240826_193424/vis_data/vis_image/scene_3_patch_6_36.png_293.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire/20240826_193424/vis_data/vis_image/scene_3_patch_5_35.png_4.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire/20240826_193424/vis_data/vis_image/scene_3_patch_5_34.png_210.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire_or_fire/20240826_193424/vis_data/vis_image/scene_3_patch_4_35.png_171.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire/20240826_194858/vis_data/vis_image/scene_1_patch_10_11.png_215.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire/20240826_194858/vis_data/vis_image/scene_3_patch_3_34.png_216.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire/20240826_194858/vis_data/vis_image/scene_3_patch_4_35.png_171.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire/20240826_194858/vis_data/vis_image/scene_3_patch_5_35.png_4.png",
    "work_dirs/example/remoteclip/Sen2Fire_wildfire/20240826_194858/vis_data/vis_image/scene_3_patch_6_36.png_293.png"
]

# 移动文件到对应的子文件夹
for file_path in file_paths:
    if "Sen2Fire_wildfire_or_fire_or_smoke" in file_path:
        subfolder = "Sen2Fire_wildfire_or_fire_or_smoke"
    elif "Sen2Fire_wildfire_or_fire" in file_path:
        subfolder = "Sen2Fire_wildfire_or_fire"
    elif "Sen2Fire_wildfire" in file_path:
        subfolder = "Sen2Fire_wildfire"
    else:
        continue  # 如果不匹配任何子文件夹，跳过

    # 获取文件名
    file_name = os.path.basename(file_path)
    # 构建目标文件路径
    dest_path = os.path.join(target_dir, subfolder, file_name)
    # 移动文件
    shutil.move(file_path, dest_path)

print("文件移动完成！")
