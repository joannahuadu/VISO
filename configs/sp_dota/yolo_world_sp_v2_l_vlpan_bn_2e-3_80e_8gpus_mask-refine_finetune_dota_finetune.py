_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]

# hyper-parameters
train_batch_size_per_gpu = 4

load_from="/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota/best_dota_mAP_epoch_80.pth"

