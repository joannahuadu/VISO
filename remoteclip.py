# # from huggingface_hub import hf_hub_download
# # import torch, open_clip
# # from PIL import Image
# # from IPython.display import display


# # model_name = 'ViT-B-32' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
# # model, _, preprocess = open_clip.create_model_and_transforms(model_name)
# # tokenizer = open_clip.get_tokenizer(model_name)

# # path_to_your_checkpoints = 'weights/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'

# # ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cpu")
# # message = model.load_state_dict(ckpt)
# # print(message.haha)
# # model = model.cuda().eval()

# from datasets import load_dataset
# fw = load_dataset("xiang709/VRSBench", split="train", streaming=True)


import zipfile
import os

def split_zip(input_zip_path, output_dir, max_size_mb=100):
    # 打开原始 ZIP 文件
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        # 获取 ZIP 文件中的所有文件
        file_names = zip_ref.namelist()

        # 当前拆分的编号
        part_number = 1
        current_size = 0
        part_zip = None
        
        # 创建一个目录存储分割的 ZIP 文件
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开并写入第一个 ZIP 包
        part_zip_path = os.path.join(output_dir, f'part_{part_number}.zip')
        part_zip = zipfile.ZipFile(part_zip_path, 'w', zipfile.ZIP_DEFLATED)

        for file_name in file_names:
            file_info = zip_ref.getinfo(file_name)
            file_size = file_info.file_size

            # 如果当前文件加上新的文件超过了最大限制，开始新的一包
            if current_size + file_size > max_size_mb * 1024 * 1024 * 1024:
                # 关闭当前的 ZIP 文件
                part_zip.close()
                part_number += 1
                part_zip_path = os.path.join(output_dir, f'part_{part_number}.zip')
                part_zip = zipfile.ZipFile(part_zip_path, 'w', zipfile.ZIP_DEFLATED)
                current_size = 0  # 重置当前大小

            # 将文件写入 ZIP 包
            with zip_ref.open(file_name) as source_file:
                part_zip.writestr(file_name, source_file.read())
            current_size += file_size  # 更新当前大小

        # 关闭最后一个 ZIP 包
        if part_zip:
            part_zip.close()
        print(f"Splitting complete. {part_number} parts created.")

# 使用示例
input_zip_path = '/mnt/data1/workspace/data/data/refGeo/images/geochat.zip'  # 输入的大 ZIP 文件路径
output_dir = '/mnt/data1/workspace/data/data/refGeo/split_geochat'  # 输出分包的文件夹
split_zip(input_zip_path, output_dir, max_size_mb=10)  # 最大分包为 100MB