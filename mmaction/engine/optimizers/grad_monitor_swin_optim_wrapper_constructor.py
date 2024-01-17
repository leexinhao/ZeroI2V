# Copyright (c) OpenMMLab. All rights reserved.
from functools import reduce
from operator import mul
from typing import List

import torch.nn as nn
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class GradMonitorSwinOptimWrapperConstructor(DefaultOptimWrapperConstructor):

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = 'base',
                   **kwargs) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module. Defaults to ``'base'``.
        """
        if 'custom_keys' in self.paramwise_cfg.keys():
            raise NotImplementedError('请不要在SwinOptimWrapperConstructor中使用custom_keys参数，它没实现')
        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param], 'layer_name': f'{prefix}.{name}' if prefix else name}
            if not param.requires_grad:
                params.append(param_group)
                continue

            param_group['lr'] = self.base_lr
            if self.base_wd is not None:
                param_group['weight_decay'] = self.base_wd

            processing_keys = [
                key for key in self.paramwise_cfg if key in f'{prefix}.{name}' # 因为用的是in，这个prefix有没有都不碍事
            ]
            if processing_keys:
                param_group['lr'] *= \
                    reduce(mul, [self.paramwise_cfg[key].get('lr_mult', 1.)
                                 for key in processing_keys])
                if self.base_wd is not None:
                    param_group['weight_decay'] *= \
                        reduce(mul, [self.paramwise_cfg[key].
                               get('decay_mult', 1.)
                                     for key in processing_keys])

            params.append(param_group)

            for key, value in param_group.items():
                if key == 'params' or key == 'layer_name': # NOTE: 这里要记得跳过layer_name
                    continue
                full_name = f'{prefix}.{name}' if prefix else name
                print_log(
                    f'paramwise_options -- '
                    f'{full_name}: {key} = {round(value, 8)}',
                    logger='current')

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix)
