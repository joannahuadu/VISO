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
    def __init__(self, 
                 data_root: Optional[str] = '',
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
        if test_mode:
            self.mode = 'val'
        else:
            self.mode = 'train'
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
            instance['bbox'] = [float(i) for i in bbox]
            if self.meta_label in di:
                instance[self.meta_label] = di[self.meta_label]
            
            data_info['instances'] = [instance]
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
        return [instance['cov_scores'] for instance in instances]

