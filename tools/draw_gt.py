import os
import cv2

# 定义图像目录和标注目录
image_dir = 'data/split_fMoW/example/airport/images'
annotation_dir = 'data/split_fMoW/example/airport/annfiles'
output_dir = 'data/split_fMoW/example/airport/labeled_images'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历图像目录中的所有图像文件
for image_filename in os.listdir(image_dir):
    if image_filename.endswith('.jpg'):  # 确保是图像文件
        # 构建图像文件路径
        image_path = os.path.join(image_dir, image_filename)
        
        # 构建对应的标注文件路径
        annotation_filename = image_filename.replace('.jpg', '.txt')
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        
        # 检查图像是否成功读取
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        # 读取标注文件
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as file:
                for line in file:
                    # 解析标注文件中的每一行
                    parts = line.strip().split()
                    if len(parts) != 6:
                        print(f"Invalid annotation format in file: {annotation_path}")
                        continue
                    
                    x, y, w, h, label, _ = parts
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # 在图像上绘制矩形框
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 150)
                    # 确定框在图像里面
                    assert x > 0 and y > 0 and x + w < image.shape[1] and y + h < image.shape[0]
                        
                    # 在图像上绘制标签
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 150)
        
        # 保存处理后的图像
        output_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(output_path, image)

print("Processing completed successfully.")
