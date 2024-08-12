
create env
```
conda create -n yolo_world python=3.8 -y
conda activate yolo_world
```
install pytorch
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# for nx, find version in https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# for nx, install the torchvision corresponding the version of torch
```

install mmcv
```
pip install openmim
pip install addict
mim install mmcv==2.0.1
```
for nx, install mmcv
```
sudo apt-get update
sudo apt-get install libopenblas-base # 缺少这个库
pip install openmim
pip install addict
mim install mmcv==2.0.1
```

install mmdet, mmrotate, mmyolo, transformers
```
pip install mmdet==3.0.0
pip install mmrotate==1.0.0rc1
pip install mmyolo==0.6.0
pip install transformers==4.42.3
```

install some utils
```
pip install timm==1.0.7 # but for nx, torch version is 1.11, so timm==0.6.13
pip install ninja==1.11.1.1
pip install matplotlib
pip install tensorboard
pip install open_clip_torch
pip uninstall requests
pip install urllib3==1.21.1
pip install filelock==3.14.0


```

install spconv(for 3090)
```
https://github.com/traveller59/spconv?tab=readme-ov-file
```
install spconv(for nx)
```
export CUMM_CUDA_ARCH_LIST="7.2"
git clone https://github.com/FindDefinition/cumm
cd ./cumm
git checkout v0.4.11
pip install .
git clone git clone https://github.com/traveller59/spconv
cd ./spconv
git checkout v2.3.6
# You need to remove cumm in requires section in pyproject.toml after install editable cumm and before install spconv due to pyproject limit (can't find editable installed cumm).
pip install .
pip install pccm==0.4.11
pip install ccimport==0.4.2

cd ./yolo_world
pip install -e ./
```

git clone 
get model weights and data
```
scp -P 2222 wd@101.6.21.31:/home/becool1/wd/YOLO-World/weights.zip /workplace/yolo_world
scp -P 2222 wd@101.6.21.31:/home/becool1/wd/YOLO-World/DOTA.zip /workplace/yolo_world
unzip weights.zip
unzip DOTA.zip
```


split data
```
bash ./tools/split.sh
```


test
```
./tools/dist_test.sh ./configs/val_dota/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_vis.py ./weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth "0"
```

get model checkpoint, and inference to test
```
scp -P 2222 wd@101.6.21.31:/home/becool1/wd/YOLO-World/work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_trainval/best_dota_mAP_epoch_80.pth ./work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_trainval/best_dota_mAP_epoch_80.pth

python ./tools/test.py ./configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota.py ./work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_trainval/best_dota_mAP_epoch_80.pth
```

maybe lack of some lib
```
sudo apt install libgl1
sudo apt install libglib2.0-0
pip uninstall mmcv
mim install mmcv==2.0.1
```
