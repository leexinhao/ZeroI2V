# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset
import pickle

@DATASETS.register_module()
class TubeDataset(BaseActionDataset):
    """Rawframe dataset for spatio-tempo action detection.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        filename_tmpl (str): Template for each filename.
            Defaults to ``img_{:05}.jpg``.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Defaults to False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 1, since raw frames count from 1.
            Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(img=''),
                 filename_tmpl: str = '{:05}.png',
                 num_classes: Optional[int] = None,
                 start_index: int = 1,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 split: int=1,
                 **kwargs) -> None:
        self.filename_tmpl = filename_tmpl
        self.split = split

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        anno_infos = pickle.load(self.ann_file, encoding='iso-8859-1')
        if self.test_mode:
            self.dataset_samples = anno_infos['test_videos'][self.split-1]
        else:
            self.dataset_samples = anno_infos['train_videos'][self.split-1]

        self.label_map = anno_infos['labels']

        for vid in self.dataset_samples:
            video_info = {}
            video_info['total_frames'] = anno_infos['nframes'][vid]
            video_info['resolution'] = anno_infos['resolution'][vid]
            video_info['gttubes'] = anno_infos['gttubes'][vid]
            if self.modality == 'RGB':
                video_info['frame_dir']= osp.join(self.data_prefix['img'], 'Frames', vid)
            elif self.modality == 'Flow':
                video_info['frame_dir']= osp.join(self.data_prefix['img'], 'FlowBrox04', vid)
            else:
                raise NotImplementedError(f"Not support {self.modality}!")

            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['filename_tmpl'] = self.filename_tmpl
        return data_info
