# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Callable, List, Optional, Union

import torch
from mmengine.dataset import BaseDataset

from mmaction.utils import ConfigType


class QADataset(BaseDataset, metaclass=ABCMeta):
    """Base class for VideoQA datasets.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict, optional): Path to a directory where
            videos are held. Defaults to None.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``RGB``, ``Flow``, ``Pose``,
            ``Audio``. Defaults to ``RGB``.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,

                 start_index: int = 0,
                 modality: str = 'feature',
                 **kwargs) -> None:

        self.start_index = start_index
        self.modality = modality
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index
        if self.modality != 'feature':
            raise NotImplementedError
        
        


        return data_info
