# run this script in the root directory of the project
# bash ./tools/split.sh
# make sure ./data/DOTA exists, and ./data/DOTA/{train, val, test}/{images, labelTxt} are prepared

echo "begin split"

# python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_train_2048.json &
python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_val_2048.json &
# python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_test_2048.json &
# python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_trainval_2048.json &

# python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_train.json &
python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_val.json &
# python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_test.json &
# python third_party/mmrotate/tools/data/dota/split/img_split.py --base-json third_party/mmrotate/tools/data/dota/split/split_configs/ss_trainval.json &

wait
echo "split done"
