_base_ = ('yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_1280ft_dotaval.py')

# hyper-parameters
num_classes = 15
# model settings
model = dict(num_test_classes=num_classes)

dota_val_dataset = dict(
    dataset=dict(data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_dota/'),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v1_class_texts.json')
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader