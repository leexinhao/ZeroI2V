# ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video

This repo is the official implementation of ["ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video"](https://arxiv.org/abs/2310.01324) 

If you find our work useful in your research, please cite:
```
@article{li2023zeroi2v,
  title={ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video},
  author={Li, Xinhao and Wang, Limin},
  journal={arXiv preprint arXiv:2310.01324},
  year={2023}
}
```

We will publish our source codes and pretrained model weights after the review process.

## Introduction

In this paper, we present a zero-cost adaptation paradigm (ZeroI2V) to transfer the image transformers to video recognition tasks (i.e., introduce zero extra cost to the adapted models during inference).

<img src="img/image-20231004113411368.png" alt="image-20231004113411368" style="zoom:80%;" />


## Models

### Kinetics 400

| Backbone |  Pretrain   | GFLOPs | Param | New Param (M) | acc@1 | Views | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B/16 | CLIP | 422 | 86 | 0 | 83.0 | 8x1x3 | [checkpoint]() |
| ViT-L/14 | CLIP | 7783 | 304 | 0 | 87.2 | 32x1x3 | [checkpoint]() |

### Something Something V2

| Backbone |  Pretrain   | GFLOPs | Param | New Param (M) | acc@1 | Views | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B/16 | CLIP |  422   | 86 | 0 | 67.7 | 8x3x1 | [checkpoint]() |
| ViT-L/14 | CLIP | 7783 | 304 | 0 | 72.2 | 32x3x1 |[checkpoint]()|

## TODO
- [ ] Release source codes
- [ ] Pretrained model weights





