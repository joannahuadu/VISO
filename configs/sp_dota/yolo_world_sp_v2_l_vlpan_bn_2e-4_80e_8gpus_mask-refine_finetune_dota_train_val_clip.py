_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val.py')

text_model_name = '/public/home/wang_mq22/workplace/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

_base_.model.backbone.text_model.model_name=text_model_name
_base_.model.backbone.text_model.frozen_modules=[]

train_batch_size_per_gpu = 8

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
