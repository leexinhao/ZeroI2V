# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmaction.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class EpochNotifyHook(Hook):
    """告知模型目前训练进度以在训练过程中动态改变网络结构（例如Vanilla Network）。
    """

    priority = 'NORMAL'

    def before_run(self, runner) -> None:
        """
        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model = model.backbone
        runner.logger.info(f'Use epoch notify hook, max_epochs={runner.max_epochs}')
        setattr(model, 'current_epoch', 0)
        setattr(model, 'max_epochs', runner.max_epochs)

    def before_train_epoch(self, runner) -> None:
        """Check the begin_epoch/iter is smaller than max_epochs/iters.

        Args:
            runner (Runner): The runner of the training process.
        """

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model = model.backbone
        setattr(model, 'current_epoch', runner.epoch + 1)
        runner.logger.info(f'Train notify, current epoch: {model.current_epoch}:{runner.epoch + 1} max epochs: {model.max_epochs}')



    def before_val_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before
        validation.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model = model.backbone
        runner.logger.info( f'Valid notify, current epoch: {model.current_epoch} max epochs: {model.max_epochs}')


    def before_test_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before test.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model = model.backbone
        runner.logger.info( f'Test notify, current epoch: {model.current_epoch} max epochs: {model.max_epochs}')
