from typing import Optional, List, Union, Dict
from collections import defaultdict
import torch
from torch.optim import Optimizer
import logging
import time

from mmengine.registry import OPTIM_WRAPPERS

from mmengine.logging import MessageHub, print_log
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.optim import OptimWrapper, AmpOptimWrapper

# import torchviz
# NOTE: 要使用GradMoitorSwinOptimWrapperConstructor这里的代码才能正常运行


def should_monitor_grad(params, max_norm: float, norm_type: str, grad_type='norm'):
    """
    Modify from: 
    https://pytorch.org/docs/1.12/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    params = list(
    filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) == 0:
        return False

    assert grad_type == 'norm', grad_type
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = params[0].grad.device
    if norm_type == torch.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in params]
        total_norm = norms[0] if len(
            norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
        # print(f"'\033[31m {len(params)} {total_norm}\033[0m")

    if torch.logical_or(total_norm.isnan(), total_norm.isinf()) or total_norm > max_norm:
        return True
    else:
        return False


@OPTIM_WRAPPERS.register_module()
class GradMonitorOptimWrapper(OptimWrapper):
    def __init__(self, optimizer: Optimizer, accumulative_counts: int = 1, clip_grad: Optional[dict] = None, monitor_grad: Optional[dict] = None):
        super().__init__(optimizer, accumulative_counts, clip_grad)
        self.monitor_grad = monitor_grad

    def _monitor_grad(self):
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])
        return should_monitor_grad(params, self.monitor_grad.get('max_norm'),
                                    self.monitor_grad.get('norm_type'), self.monitor_grad.get('type', 'norm'))

    def record_params_grad(self, key):
        assert 'grad' in key or 'Grad' in key, key
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None or not param.requires_grad:
                        continue
                    if param.grad.is_sparse:
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad
                    v = to_unscale.clone().abs().max()
                    self.params_info_records[group['layer_name']].append(
                        (key, v))

    def record_params_value(self, key):
        assert 'value' in key or 'Value' in key, key
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                assert len(group["params"]) == 1, len(group["params"])
                for param in group["params"]:
                    if not param.requires_grad:  # 只关心可训练参数
                        continue
                    if param.data.is_sparse:
                        if param.data.dtype is torch.float16:
                            param.data = param.data.coalesce()
                        to_unscale = param.data._values()
                    else:
                        to_unscale = param.data
                    v = to_unscale.clone().abs().max()
                    self.params_info_records[group['layer_name']].append(
                        (key, v))

    def log_params_info(self):
        with torch.no_grad():
            for layer_name in self.params_info_records.keys():
                level = logging.INFO
                log_info = f'{layer_name}: '
                for k, v in self.params_info_records[layer_name]:
                    if torch.isinf(v) or torch.isnan(v):
                        level = logging.WARNING
                    log_info += f'{k}={v} '

                print_log(log_info, logger='current', level=level)

            del self.params_info_records

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """

        if self.monitor_grad is not None:
            monitor_grad = self._monitor_grad()  # 检查总的梯度norm是否大于阈值，大于阈值的话输出所有层梯度来debug
        else:
            monitor_grad = False

        if monitor_grad:
            self.params_info_records = defaultdict(list)
            self.record_params_grad('Max grad after backward')
        else:
            self.params_info_records = None  # 冗余增加可靠

        if self.clip_grad_kwargs:
            self._clip_grad()
            if monitor_grad:
                self.record_params_grad('Max grad after clip')  # 检查clip后的梯度

        if monitor_grad:
            self.record_params_value('Max value before step')

        self.optimizer.step(**kwargs)
        # 检查梯度更新前后的值

        if monitor_grad:
            self.record_params_value('Max value after step')
            self.log_params_info()  # 统一log以免太多行了


@OPTIM_WRAPPERS.register_module()
class GradMonitorAmpOptimWrapper(AmpOptimWrapper):
    def __init__(self, loss_scale: str = 'dynamic', dtype: Union[str, torch.dtype] = None, monitor_grad: Optional[dict] = None, **kwargs):
        super().__init__(loss_scale, dtype, **kwargs)
        self.monitor_grad = monitor_grad

    def _monitor_grad(self):
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])
        
        return should_monitor_grad(params, float(self.monitor_grad.get('max_norm')),
                                    float(self.monitor_grad.get('norm_type')), self.monitor_grad.get('type', 'norm'))

    def record_params_grad(self, key):
        assert 'grad' in key or 'Grad' in key, key
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None or not param.requires_grad:
                        continue
                    if param.grad.is_sparse:
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad
                    v = to_unscale.clone().abs().max()
                    self.params_info_records[group['layer_name']].append(
                        (key, v))

    def record_params_value(self, key):
        assert 'value' in key or 'Value' in key, key
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                assert len(group["params"]) == 1, len(group["params"])
                for param in group["params"]:
                    if not param.requires_grad:  # 只关心可训练参数
                        continue
                    if param.data.is_sparse:
                        if param.data.dtype is torch.float16:
                            param.data = param.data.coalesce()
                        to_unscale = param.data._values()
                    else:
                        to_unscale = param.data
                    v = to_unscale.clone().abs().max()
                    self.params_info_records[group['layer_name']].append(
                        (key, v))

    def log_params_info(self):
        with torch.no_grad():
            for layer_name in self.params_info_records.keys():
                level = logging.INFO
                log_info = f'{layer_name}: '
                for k, v in self.params_info_records[layer_name]:
                    if torch.isinf(v) or torch.isnan(v):
                        level = logging.WARNING
                    log_info += f'{k}={v} '

                print_log(log_info, logger='current', level=level)

            del self.params_info_records

    def step(self, **kwargs):
        """Update parameters with :attr:`loss_scaler`.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """

        self.message_hub.update_scalar(f'train/grad_scale', self.loss_scaler.get_scale())
        self.loss_scaler.unscale_(self.optimizer)

        if self.monitor_grad is not None:
            monitor_grad = self._monitor_grad()  # 检查总的梯度norm是否大于阈值，大于阈值的话输出所有层梯度来debug
        else:
            monitor_grad = False

        if monitor_grad:
            self.params_info_records = defaultdict(list)
            self.record_params_grad('Max grad after backward')
        else:
            self.params_info_records = None  # 冗余增加可靠

        if self.clip_grad_kwargs:
            # self.loss_scaler.unscale_(self.optimizer) 移到前面执行了
            self._clip_grad()
            if monitor_grad:
                self.record_params_grad('Max grad after clip')  # 检查clip后的梯度

        if monitor_grad:
            self.record_params_value('Max value before step')

        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)

        if monitor_grad:
            self.record_params_value('Max value after step')
            self.log_params_info()  # 统一log以免太多行了


    def update_params(self,
                      loss: torch.Tensor,
                      step_kwargs: Optional[Dict] = None,
                      zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}

        # torch.cuda.synchronize()
        # start_all = time.time()

        loss = self.scale_loss(loss)

        # dot = torchviz.make_dot(loss)  # make_dot返回一个dot（一个Diagraph对象）
        # dot.render(filename='/mnt/cache/lixinhao/mmaction2-next/1x1x8graph', view=False, format='html')
        # exit(0)

        # torch.cuda.synchronize()
        # start_backward = time.time()

        self.backward(loss)

        # torch.cuda.synchronize()
        # start_step = time.time()

        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)


        # torch.cuda.synchronize()
        # end_time= time.time()

        # scale_loss_time = start_backward - start_all
        # backward_time = start_step - start_backward
        # step_time = end_time - start_step
        # all_time = end_time - start_backward
        # print(f'scale_loss:{scale_loss_time} backward:{backward_time} step:{step_time} all:{all_time}')