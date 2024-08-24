import glob
import os.path as osp
from PIL import Image
import os
import numpy as np
import json
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from mmengine.dataset import BaseDataset

from ..registry import DATASETS

@DATASETS.register_module()
class fMoWDataset(BaseDataset): 
    METAINFO = {
        'classes':('nuclear_powerplant', 'office_building', 'border_checkpoint', 'interchange', 'amusement_park', 'helipad', 'airport_hangar', 'archaeological_site', 'solar_farm', 'tunnel_opening', 'place_of_worship', 'car_dealership', 'flooded_road', 'aquaculture', 'tower', 'hospital', 'impoverished_settlement', 'space_facility', 'runway', 'surface_mine', 'gas_station', 'fire_station', 'recreational_facility', 'construction_site', 'electric_substation', 'port', 'shopping_mall', 'parking_lot_or_garage', 'swimming_pool', 'burial_site', 'road_bridge', 'race_track', 'waste_disposal', 'airport', 'ground_transportation_station', 'zoo', 'prison', 'shipyard', 'smokestack', 'oil_or_gas_facility', 'fountain', 'storage_tank', 'factory_or_powerplant', 'debris_or_rubble', 'single-unit_residential', 'lighthouse', 'stadium', 'dam', 'police_station', 'toll_booth', 'wind_farm', 'golf_course', 'multi-unit_residential', 'railway_bridge', 'educational_institution', 'airport_terminal', 'park', 'water_treatment_facility', 'lake_or_pond', 'crop_field', 'military_facility', 'barn'),
        # 'classes':
        # ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        #  'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        #  'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
        #  'harbor', 'swimming-pool', 'helicopter'),
        # palette is a list of color tuples, which is used for visualization.
    }

    def __init__(self, 
                 data_root: Optional[str] = '',
                 mode: str = 'val',
                 test_mode: bool = False,
                 meta_label: str = 'cloud_cover',
                 **kwargs) -> None:
        '''
            Maybe label could be:
            - "utm"
            - "country_code"
            - "cloud_cover"
        '''
        self.meta_label = meta_label
        self.mode = mode
        txt_files = osp.join(data_root, "fMoW_"+self.mode+".json")
        with open(txt_files, "r+", encoding='utf-8') as f:
            self.dict_list=json.load(f)
            f.close()
        super().__init__(test_mode=test_mode, data_root=data_root, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
            for i, c in enumerate(self.metainfo['classes'])
            }  # i
        data_list = []

        for di in self.dict_list:
            data_info = {}
            img_path = di["img_dir"]
            data_info['img_path'] = img_path
            img_name = osp.basename(img_path)
            data_info['file_name'] = img_name
            img_id = img_name.rsplit('.', 1)[0] 
            data_info['img_id'] = img_id
            
            meta_path = di["meta_dir"]
            data_info["meta_path"] = meta_path
            
            instances = []
            instance = {}
            bbox = di["box"]
            cls_name = di["category"]
            if cls_name in self.metainfo['classes']:
                instance['bbox'] = [float(i) for i in bbox]
                instance['bbox_label'] = cls_map[cls_name]
                instance['ignore_flag'] = 0
                instances.append(instance)
            data_info['instances'] = instances
            metas = []
            meta = {}
            if self.meta_label in di:
                meta[self.meta_label] = di[self.meta_label]
                metas.append(meta)
            data_info['metas'] = metas
            data_list.append(data_info)

        return data_list
    
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]

