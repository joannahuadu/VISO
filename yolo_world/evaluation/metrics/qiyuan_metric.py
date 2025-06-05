import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
import json
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.structures.bbox import rbox2qbox, qbox2hbox

from mmrotate.evaluation import DOTAMetric
from mmyolo.registry import METRICS
from mmengine.fileio import dump, get_local_path, load

from mmdet.evaluation import CocoMetric

@METRICS.register_module()
class QiyuanMetric(DOTAMetric):
    """Customized COCO metric with modified results2json for rbox."""

    def __init__(self, 
                 jsonfile: Optional[Union[str, List[str]]] = None,
                 *args, 
                 **kwargs):
        """
        Args:
            merge_patches (bool): Whether to merge patches' predictions into
                full image's results and generate a zip file for DOTA online
                evaluation.
        """
        super().__init__(*args, **kwargs)
        assert jsonfile is not None, \
            'jsonfile must be specified for QiyuanMetric.'

        with open(jsonfile, 'r') as f:
            json_data = json.load(f)

        self.img_ids = {
            item['file_name']: item['id']
            for item in json_data['images']
        }
        self.images = json_data['images']
        self.categories = json_data['categories']
    
    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
        
    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.

        You can submit it at:
        https://captain-whu.github.io/DOTA/evaluation.html

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            file_name = oriname + '.jpg'
            image_id = self.img_ids.get(file_name)
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[image_id].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5],
                                                  cls_dets[:,
                                                           -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    else:
                        raise NotImplementedError
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'The outfile_prefix should be a non-exist path, '
                             f'but {outfile_prefix} is existing. '
                             f'Please delete it firstly.')
        os.makedirs(outfile_prefix, exist_ok=True)

        bbox_json_results = []
        instance_id = 0
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for label, dets in enumerate(dets_per_cls):
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                else:
                    raise NotImplementedError
                for qbox, score in zip(qboxes, scores):
                    instance_id+=1
                    data = dict()
                    data['image_id'] = img_id
                    data['id'] = instance_id
                    data['bbox'] = self.xyxy2xywh(qbox2hbox(qbox))
                    data['score'] = float(score)
                    data['category_id'] = label + 1
                    bbox_json_results.append(data)
        
        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}/pred.json'
        coco_format_results = {
            "images": self.images,
            "annotations": bbox_json_results,
            "categories": self.categories,
        }

        dump(coco_format_results, result_files['bbox'])

        return osp.dirname(f'{outfile_prefix}/pred.json')


@METRICS.register_module()
class QiyuanCOCOMetric(CocoMetric):
    """Customized COCO metric with modified results2json for rbox."""

    def __init__(self, 
                 jsonfile: Optional[Union[str, List[str]]] = None,
                 *args, 
                 **kwargs):
        """
        Args:
            merge_patches (bool): Whether to merge patches' predictions into
                full image's results and generate a zip file for DOTA online
                evaluation.
        """
        super().__init__(*args, **kwargs)
        assert jsonfile is not None, \
            'jsonfile must be specified for QiyuanMetric.'

        with open(jsonfile, 'r') as f:
            json_data = json.load(f)

        self.images = json_data['images']
        self.categories = json_data['categories']
        self.cat_ids = {4: 1, 2: 2, 6: 3, 5: 4, 8: 5, 9: 6, 1: 7, 3: 8, 0: 9, 7: 10}
    
    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        instance_id = 0
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                instance_id+=1
                data = dict()
                data['image_id'] = image_id
                data['id'] = instance_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}/pred.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        
        coco_format_results = {
            "images": self.images,
            "annotations": bbox_json_results,
            "categories": self.categories,
        }
        
        dump(coco_format_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files
