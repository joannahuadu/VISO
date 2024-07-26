_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]

load_from = ''
text_model_name = 'ViT-B-32'
text_pretrained = 'weights/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt'

model = dict(
    backbone=dict(
        text_model=dict(
            type='OpenCLIPLanguageBackbone',
            model_name=text_model_name,
            pretrained=text_pretrained,
            frozen_modules=['all']),
    )
)
