"""
Author: Sijie Hu
Date: 18/03/2024
Description: This script adapted from mmdet.formatting.py to contain
 functions for collecting and formatting disparity data.
Notes: Currently, only support transforms for YOLOX, you should take
    care of `cteate_input` for other transforms.
"""

import sys
from typing import Union, Sequence, Optional, Tuple, Dict, List

import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor
from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task
from mmdeploy.utils.config_utils import (get_backend, get_input_shape,
                                         is_dynamic_shape)

from mmdeploy.codebase.mmdet.deploy.object_detection import ObjectDetection, MMDET_TASK


def process_model_config_mm(model_cfg: Config,
                         img_pairs: List[List[str]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (Sequence[List(str)] | Sequence[List(np.ndarray)]): Input image(s), accepted
            data type are List[List(str)], List[List(np.ndarray)].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """

    cfg = model_cfg.copy()

    if isinstance(img_pairs[0][0], np.ndarray):
        raise NotImplementedError('Currently, only support image path input')
        sys.exit(1)

    pipeline = cfg.test_pipeline

    for i, transform in enumerate(pipeline):
        # for static exporting
        if input_shape is not None:
            if transform.type == 'Resize_Disparity':
                pipeline[i].keep_ratio = False
                pipeline[i].scale = tuple(input_shape)
            elif transform.type == 'Pad_Disparity' and 'size' in transform:
                pipeline[i].size = tuple(input_shape)

    pipeline = [
        transform for transform in pipeline
        if transform.type != 'LoadAnnotations'
    ]
    cfg.test_pipeline = pipeline
    return cfg


@MMDET_TASK.register_module(Task.OBJECT_DETECTION_MM.value)
class ObjectDetection_MM(ObjectDetection):

    def create_input(
        self,
        img_pairs: List[List[str]],
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create multimodal inputs for detector.

        Args:
            img_pairs (List[List[str]]): Input image pair(s), accept
                only a list of image paths.
            input_shape (Sequence[int]): The input shape of the model.
            data_preprocessor (BaseDataPreprocessor): A data preprocessor to
                preprocess input data.

        Returns:
            Tuple[Dict, torch.Tensor]: A tuple of multimodal inputs and model inputs.
        """
        from mmcv.transforms import Compose
        assert len(img_pairs[0]) == 2, \
            'The input should be a list of image pairs (rgb, disparity).'
        dynamic_flag = is_dynamic_shape(self.deploy_cfg)
        cfg = process_model_config_mm(self.model_cfg, img_pairs, input_shape)
        # Drop pad_to_square when static shape. Because static shape should
        # ensure the shape before input image.

        pipeline = cfg.test_pipeline
        if not dynamic_flag:
            transform = pipeline[0]
            if 'transforms' in transform:
                transform_list = transform['transforms']
                for i, step in enumerate(transform_list):
                    if step['type'] == 'Pad_Disparity' and 'pad_to_square' in step \
                            and step['pad_to_square']:
                        transform_list.pop(i)
                        break
        test_pipeline = Compose(pipeline)
        data = []
        for img_pair in img_pairs:
            # prepare data
            # TODO: remove img_id.
            print(img_pair)
            data_ = dict(img_path=img_pair[0], disp_path=img_pair[1], img_id=0)
            # build the data pipeline
            data_ = test_pipeline(data_)
            data.append(data_)

        data = pseudo_collate(data)
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data['inputs']
        else:
            return data, BaseTask.get_tensor_from_input(data)