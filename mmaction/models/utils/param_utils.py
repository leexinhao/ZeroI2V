import re
import torch
from torch import nn
from collections import OrderedDict
from typing import List
from mmengine.runner.checkpoint import _load_checkpoint
import warnings
from mmengine.logging import MMLogger

def inflate_2dcnn_weights(model, pretrained: str, logger: MMLogger, inflate_init_type='center',  revise_keys=None) -> None:
    """Inflate the convnext2d parameters to convnext3d.

    The differences between convnext3d and convnext2d mainly lie in an extra
    axis of conv kernel. To utilize the pretrained parameters in 2d model,
    the weight of conv2d models should be inflated to fit in the shapes of
    the 3d counterpart.
    """
    state_dict_2d = _load_checkpoint(pretrained, map_location='cpu')
    if 'state_dict' in state_dict_2d:
        state_dict_2d = state_dict_2d['state_dict']

    if revise_keys is not None:
        for p, r in revise_keys:
            state_dict_2d = OrderedDict(
                {re.sub(p, r, k): v
                    for k, v in state_dict_2d.items()})

    inflated_param_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            inflate_conv2d_params(
                module, state_dict_2d, name, inflated_param_names, inflate_init_type)
        elif isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
            copy_params(module, state_dict_2d, name, inflated_param_names)
    # check if any parameters in the 2d checkpoint are not loaded
    remaining_names = set(
        state_dict_2d.keys()) - set(inflated_param_names)

    for name, param in model.named_parameters():  # 加载预训练parameters
        if name in remaining_names and param.data.shape != state_dict_2d[name].data.shape:
            print('----- name is same but shape is not same:')
            print(name, param.data.shape, state_dict_2d[name].data.shape)
            print('-----')

        if name in remaining_names and param.data.shape == state_dict_2d[name].data.shape:
            param.data.copy_(state_dict_2d[name].data)
            remaining_names.remove(name)

    if remaining_names:
        logger.info(f'These parameters in the 2d checkpoint are not loaded'
                    f': {remaining_names}')
            
def copy_params(m: nn.Module, state_dict_2d: OrderedDict,
                module_name_2d: str,
                inflated_param_names: List[str] = None) -> None:
    """Inflate a norm module from 2d to 3d.

    Args:
        m (nn.Module): The destination module.
        state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
        module_name_2d (str): The name of corresponding m module in the
            2d model.
        inflated_param_names (List[str]): List of parameters that have been
            inflated.
    """
    for param_name, param in m.named_parameters():
        param_2d_name = f'{module_name_2d}.{param_name}'
        param_2d = state_dict_2d[param_2d_name]
        if param.data.shape != param_2d.shape:
            warnings.warn(
                f'The parameter of {module_name_2d} is not loaded due to incompatible shapes {param.data.shape} != {param_2d.shape}.')
            return

        param.data.copy_(param_2d)
        if inflated_param_names is not None:
            inflated_param_names.append(param_2d_name)

    for param_name, param in m.named_buffers():
        param_2d_name = f'{module_name_2d}.{param_name}'
        # some buffers like num_batches_tracked may not exist in old
        # checkpoints
        if param_2d_name in state_dict_2d:
            param_2d = state_dict_2d[param_2d_name]
            param.data.copy_(param_2d)
            if inflated_param_names is not None:
                inflated_param_names.append(param_2d_name)


def inflate_conv2d_params(conv3d: nn.Module, state_dict_2d: OrderedDict,
                          module_name_2d: str,
                          inflated_param_names: List[str]=None,
                          inflate_init_type='center',
                          delete_old=False) -> None:
    """Inflate a conv module from 2d to 3d.

    Args:
        conv3d (nn.Module): The destination conv3d module.
        state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
        module_name_2d (str): The name of corresponding conv module in the
            2d model.
        inflated_param_names (List[str]): List of parameters that have been
            inflated.
    """
    weight_2d_name = module_name_2d + '.weight'
    conv2d_weight = state_dict_2d[weight_2d_name]
    kernel_t = conv3d.weight.data.shape[2]

    conv3d.weight.data.copy_(inflate_conv2d_weight(
        conv2d_weight, kernel_t, inflate_init_type))
    if inflated_param_names is not None:
        inflated_param_names.append(weight_2d_name)
    if delete_old:
        del state_dict_2d[weight_2d_name]
        
    if getattr(conv3d, 'bias') is not None:
        bias_2d_name = module_name_2d + '.bias'
        conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
        if inflated_param_names is not None:
            inflated_param_names.append(bias_2d_name)
        if delete_old:
            del state_dict_2d[bias_2d_name]

def inflate_conv2d_weight(weight_2d, time_dim, inflate_init_type='center'):
    if inflate_init_type == 'center':
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    elif inflate_init_type == 'i3d':
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    else:
        raise NotImplementedError(
            f"Not support inflate_init_type:{inflate_init_type}")
    return weight_3d
