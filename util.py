import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import os
import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as al

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from albumentations.augmentations import functional as F
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet


# import torch_optimizer as toptim

def load_checkpoint(checkpoint_path, model, model_name=None, is_different_class_num=False, not_dict_model=False,
                    strict=False):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path)
    if not_dict_model:
        pretrained_dict = checkpoint_dict
    else:
        pretrained_dict = checkpoint_dict['state_dict']

    if is_different_class_num:
        load_different_state_dict(model_name, model, pretrained_dict)
    else:
        model.load_state_dict(pretrained_dict, strict=strict)
    if not_dict_model:
        return model, None, None, 1
    optimizer_state = checkpoint_dict['optimizer']
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer_state, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    if hasattr(model, 'module'):
        model = model.module
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


model_classifier_map = {
    'alexnet': ['classifier', 6],
    'vgg': ['classifier', 6],
    'mobilenet': ['classifier', 1],
    'mnasnet': ['classifier', 6],
    'resnet': ['fc'],
    'inception': ['fc'],
    'googlenet': ['fc'],
    'shufflenet': ['fc'],
    'densenet': ['classifier'],
    'resnext': ['fc'],
    'wide_resnet': ['fc'],
    'efficientnet': ['_fc'],
    'bagnet': ['fc'],
    'rexnet': ['output', 1],
}


def load_different_state_dict(model_name, model, pretrained_dict):
    for m_key in model_classifier_map:
        if m_key in model_name:
            cls_layers = model_classifier_map[m_key]
            if len(cls_layers) == 1:
                fc_weight_key = cls_layers[0] + ".weight"
                fc_bias_key = cls_layers[0] + ".bias"
            else:
                fc_weight_key = "{}.{}.weight".format(cls_layers[0], cls_layers[1])
                fc_bias_key = "{}.{}.bias".format(cls_layers[0], cls_layers[1])
            # pretrained_dict[fc_weight_key] = pretrained_dict[fc_weight_key][:nums_classes,:]
            pretrained_dict.pop(fc_weight_key)
            pretrained_dict.pop(fc_bias_key)
            model.load_state_dict(pretrained_dict, strict=False)
            return True
    raise Exception("unknown model name", model_name)


def init_model(model_name, num_classes=10, pretrained=True, pooling=None):
    if model_name.startswith("arc_face"):
        _, backbone_name = model_name.split(",")
        from arc_face_net import ArcFaceNet
        return ArcFaceNet(model_name=backbone_name, pretrained=pretrained, class_num=num_classes, pooling=pooling)
    if model_name.startswith("efficientnet_"):
        import timm
        model = timm.create_model(model_name, pretrained=pretrained)
        return model
    if model_name.startswith("efficientnet"):
        if pretrained:
            model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
            return model
        else:
            try:
                model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
            except:
                model = EfficientNet.from_name(model_name, num_classes=num_classes)
            return model
    if model_name == "bagnet":
        import bagnets.pytorchnet
        model = bagnets.pytorchnet.bagnet17(pretrained=pretrained)
        cls_layers = model_classifier_map[model_name]
        in_features = getattr(model, cls_layers[0]).in_features
        setattr(model, cls_layers[0], nn.Linear(in_features, num_classes))
        return model

    if model_name == "rexnet":
        import rexnet
        model = rexnet.ReXNetV1(width_mult=2.0, classes=num_classes)
        return model

    if model_name.startswith("fishnet"):
        import net_factory
        model = getattr(net_factory, model_name)(num_cls=num_classes)
        return model

    for m_key in model_classifier_map:
        if m_key in model_name:
            model_fn = getattr(models, model_name)
            cls_layers = model_classifier_map[m_key]

            if model_name.startswith("inception"):
                model = model_fn(pretrained=pretrained, aux_logits=False)
            else:
                model = model_fn(pretrained=pretrained)

            if len(cls_layers) == 1:
                in_features = getattr(model, cls_layers[0]).in_features
                setattr(model, cls_layers[0], nn.Linear(in_features, num_classes))
            else:
                classifier = getattr(model, cls_layers[0])
                in_features = classifier[cls_layers[1]].in_features
                classifier[cls_layers[1]] = nn.Linear(in_features, num_classes)
            return model

    raise Exception("unknown model name", model_name)


def init_optimizer(optimizer_name, model, optimizer_state, lr, wd, lr_restart_step=1, lr_decay_gamma=0.9,
                   scheduler="step", nesterov=False, num_epochs=None, steps_per_epoch=None):
    if optimizer_name == "sgd":
        optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=nesterov)
    elif optimizer_name == "adam":
        optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adamp":
        from adamp import AdamP
        optimizer_ft = AdamP(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
    elif optimizer_name == "sgdp":
        from adamp import SGDP
        optimizer_ft = SGDP(model.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=nesterov)
    else:
        opt_attr = getattr(toptim, optimizer_name)
        if opt_attr:
            optimizer_ft = opt_attr(model.parameters())
        else:
            raise Exception("unknown optimizer name", optimizer_name)

    if optimizer_state is not None:
        optimizer_ft.load_state_dict(optimizer_state)

    if scheduler == "cosine":
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, lr_restart_step)
        use_lr_schedule_steps = True
    elif scheduler == "cycle":
        exp_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_ft, max_lr=lr, steps_per_epoch=steps_per_epoch,
                                                               epochs=num_epochs, pct_start=0.1)
        use_lr_schedule_steps = False
    elif scheduler == "step":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_restart_step, gamma=lr_decay_gamma)
        use_lr_schedule_steps = False

    return optimizer_ft, exp_lr_scheduler, use_lr_schedule_steps


def get_train_transforms(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                         center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    if use_crop:
        resize = [al.Resize(int(input_size[0] * 1.1), int(input_size[1] * 1.1)),
                  al.RandomSizedCrop(min_max_height=(int(input_size[0] * 0.7), int(input_size[0] * 1.1)),
                                     height=input_size[0],
                                     width=input_size[1])]
    elif use_center_crop:
        resize = [
            al.Resize(int(input_size[0] * (2. - center_crop_ratio)), int(input_size[1] * (2. - center_crop_ratio))),
            al.CenterCrop(input_size[0], input_size[1])]
    else:
        resize = [al.Resize(input_size[0], input_size[1])]
    if use_no_color_aug:
        return al.Compose(
            resize +
            [
                al.Flip(p=0.5),
                al.OneOf([
                    al.RandomRotate90(),
                    al.Rotate(limit=180),
                ], p=0.2),
                al.OneOf([
                    al.ShiftScaleRotate(),
                    al.OpticalDistortion(),
                    al.GridDistortion(),
                    al.ElasticTransform(),
                ], p=0.1),
                al.RandomGridShuffle(p=0.05),
                al.RandomSnow(p=0.05),
                al.RandomRain(p=0.05),
                al.RandomFog(p=0.05),
                al.RandomSunFlare(p=0.05),
                al.RandomShadow(p=0.05),
                al.RandomBrightnessContrast(p=0.05),
                al.GaussNoise(p=0.05),
                al.ISONoise(p=0.05),
                al.MultiplicativeNoise(p=0.05),
                al.ToGray(p=1. if use_gray else 0.),
                al.OneOf([
                    al.MotionBlur(blur_limit=3),
                    al.Blur(blur_limit=3),
                    al.MedianBlur(blur_limit=3),
                    al.GaussianBlur(blur_limit=3),
                ], p=0.05),
                al.CoarseDropout(p=0.05),
                al.Cutout(p=0.05),
                al.GridDropout(p=0.05),
                al.Downscale(p=0.1),
                al.ImageCompression(quality_lower=60, p=0.2),
                al.Normalize(),
                ToTensorV2()
            ])
    else:
        return al.Compose(
            resize +
            [
                al.Flip(p=0.5),
                al.OneOf([
                    al.RandomRotate90(),
                    al.Rotate(limit=180),
                ], p=0.2),
                al.OneOf([
                    al.ShiftScaleRotate(),
                    al.OpticalDistortion(),
                    al.GridDistortion(),
                    al.ElasticTransform(),
                ], p=0.1),
                al.RandomGridShuffle(p=0.05),
                al.OneOf([
                    al.RandomGamma(),
                    al.HueSaturationValue(),
                    al.RGBShift(),
                    al.CLAHE(),
                    al.ChannelShuffle(),
                    al.InvertImg(),
                ], p=0.1),
                al.RandomSnow(p=0.05),
                al.RandomRain(p=0.05),
                al.RandomFog(p=0.05),
                al.RandomSunFlare(p=0.05),
                al.RandomShadow(p=0.05),
                al.RandomBrightnessContrast(p=0.05),
                al.GaussNoise(p=0.05),
                al.ISONoise(p=0.05),
                al.MultiplicativeNoise(p=0.05),
                al.ToGray(p=1. if use_gray else 0.05),
                al.ToSepia(p=0.05),
                al.Solarize(p=0.05),
                al.Equalize(p=0.05),
                al.Posterize(p=0.05),
                al.FancyPCA(p=0.05),
                al.OneOf([
                    al.MotionBlur(blur_limit=3),
                    al.Blur(blur_limit=3),
                    al.MedianBlur(blur_limit=3),
                    al.GaussianBlur(blur_limit=3),
                ], p=0.05),
                al.CoarseDropout(p=0.05),
                al.Cutout(p=0.05),
                al.GridDropout(p=0.05),
                al.ChannelDropout(p=0.05),
                al.Downscale(p=0.1),
                al.ImageCompression(quality_lower=60, p=0.2),
                al.Normalize(),
                ToTensorV2()
            ])


def get_train_transforms_simple_only_bright(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                            center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return al.Compose(
        [
            al.Resize(input_size[0], input_size[1], p=1.0),
            al.HorizontalFlip(p=0.5),
            al.RandomBrightness(p=0.25, limit=0.2),
            al.RandomContrast(p=0.25, limit=0.2),
            al.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=15, p=0.3,
                                border_mode=cv2.BORDER_CONSTANT),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_simple_bright_pad(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                           center_crop_ratio=0.9, use_gray=False):
    def longest_max_size(img, interpolation=cv2.INTER_LINEAR, **params):
        img = F.longest_max_size(img, max_size=input_size, interpolation=interpolation)

        return img

    return al.Compose(
        [
            al.Lambda(longest_max_size),
            al.PadIfNeeded(min_height=input_size, min_width=input_size, always_apply=True, border_mode=0),
            al.HorizontalFlip(p=0.5),
            al.RandomBrightness(p=0.2, limit=0.2),
            al.RandomContrast(p=0.1, limit=0.2),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_simple_bright(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                       center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return al.Compose(
        [
            al.Resize(input_size[0], input_size[1], p=1.0),
            al.HorizontalFlip(p=0.5),
            al.RandomBrightness(p=0.2, limit=0.2),
            al.RandomContrast(p=0.1, limit=0.2),
            al.ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_simple_bright_randomcrop_pad(input_size, use_crop=False, use_no_color_aug=False,
                                                      use_center_crop=False,
                                                      center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    def longest_max_size(img, interpolation=cv2.INTER_LINEAR, **params):
        img = F.longest_max_size(img, max_size=input_size[1], interpolation=interpolation)

        return img

    return al.Compose(
        [al.RandomResizedCrop(height=input_size[0],
                              width=input_size[1],
                              scale=(0.4, 1.0),
                              ratio=(0.85, 1.15),
                              interpolation=0,
                              p=0.5),
         al.Lambda(longest_max_size),
         al.PadIfNeeded(min_height=input_size[1], min_width=input_size[1], always_apply=True, border_mode=0),
         # al.Resize(input_size[0], input_size[1], p=1.0),
         al.HorizontalFlip(p=0.5),
         al.RandomBrightness(p=0.2, limit=0.2),
         al.RandomContrast(p=0.1, limit=0.2),
         al.ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
         al.Normalize(),
         ToTensorV2()
         ])


def get_train_transforms_simple_bright_randomcrop(input_size, use_crop=False, use_no_color_aug=False,
                                                  use_center_crop=False,
                                                  center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return al.Compose(
        [al.RandomResizedCrop(height=input_size[0],
                              width=input_size[1],
                              scale=(0.4, 1.0),
                              interpolation=0,
                              p=0.5),
         al.Resize(input_size[0], input_size[1], p=1.0),
         al.HorizontalFlip(p=0.5),
         al.RandomBrightness(p=0.2, limit=0.2),
         al.RandomContrast(p=0.1, limit=0.2),
         al.ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
         al.Normalize(),
         ToTensorV2()
         ])


def get_train_transforms_resize(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return al.Compose(
        [
            al.Resize(input_size[0], input_size[1], p=1.0),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_simple(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return al.Compose(
        [
            al.Resize(input_size[0], input_size[1], p=1.0),
            al.HorizontalFlip(p=0.5),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_simple_randomcrop(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                           center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size[0], input_size[1])
    return al.Compose(
        [al.RandomResizedCrop(height=input_size[0],
                              width=input_size[1],
                              scale=(0.4, 1.0),
                              interpolation=0,
                              p=0.5),
         al.Resize(input_size[0], input_size[1], p=1.0),
         al.HorizontalFlip(p=0.5),
         al.Normalize(),
         ToTensorV2()
         ])


def get_train_transforms_mmdetection(input_size, use_crop=False, use_no_color_aug=False, use_center_crop=False,
                                     center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size[0], input_size[1])
    return al.Compose(
        [
            al.RandomResizedCrop(height=input_size[0],
                                 width=input_size[1],
                                 scale=(0.4, 1.0),
                                 interpolation=0,
                                 p=0.5),
            al.Resize(input_size[0], input_size[1], p=1.0),
            al.HorizontalFlip(p=0.5),
            al.OneOf([
                al.ShiftScaleRotate(border_mode=0,
                                    shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2),
                                    rotate_limit=(-20, 20)),
                al.OpticalDistortion(border_mode=0,
                                     distort_limit=[-0.5, 0.5], shift_limit=[-0.5, 0.5]),
                al.GridDistortion(num_steps=5, distort_limit=[-0., 0.3], border_mode=0),
                al.ElasticTransform(border_mode=0),
                al.IAAPerspective(),
                al.RandomGridShuffle()
            ], p=0.1),
            al.Rotate(limit=(-25, 25), border_mode=0, p=0.1),
            al.OneOf([
                al.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2)),
                al.HueSaturationValue(hue_shift_limit=(-20, 20),
                                      sat_shift_limit=(-30, 30),
                                      val_shift_limit=(-20, 20)),
                al.RandomGamma(gamma_limit=(30, 150)),
                al.RGBShift(),
                al.CLAHE(clip_limit=(1, 15)),
                al.ChannelShuffle(),
                al.InvertImg(),
            ], p=0.1),
            al.RandomSnow(p=0.05),
            al.RandomRain(p=0.05),
            al.RandomFog(p=0.05),
            al.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=110, p=0.05),
            al.RandomShadow(p=0.05),

            al.GaussNoise(var_limit=(10, 20), p=0.05),
            al.ISONoise(color_shift=(0, 15), p=0.05),
            al.MultiplicativeNoise(p=0.05),
            al.OneOf([
                al.ToGray(p=1. if use_gray else 0.05),
                al.ToSepia(p=0.05),
                al.Solarize(p=0.05),
                al.Equalize(p=0.05),
                al.Posterize(p=0.05),
                al.FancyPCA(p=0.05),
            ], p=0.05),
            al.OneOf([
                al.MotionBlur(blur_limit=(3, 7)),
                al.Blur(blur_limit=(3, 7)),
                al.MedianBlur(blur_limit=3),
                al.GaussianBlur(blur_limit=3),
            ], p=0.05),
            al.CoarseDropout(p=0.05),
            al.Cutout(num_holes=30, max_h_size=37, max_w_size=37, fill_value=0, p=0.05),
            al.GridDropout(p=0.05),
            al.ChannelDropout(p=0.05),
            al.Downscale(scale_min=0.5, scale_max=0.9, p=0.1),
            al.ImageCompression(quality_lower=60, p=0.2),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_atopy(input_size, use_crop=False, use_no_color_aug=False):
    if use_crop:
        resize = [al.Resize(int(input_size * 1.2), int(input_size * 1.2)),
                  al.RandomSizedCrop(min_max_height=(int(input_size * 0.6), int(input_size * 1.2)), height=input_size,
                                     width=input_size)]
    else:
        resize = [al.Resize(input_size, input_size)]
    return al.Compose(
        resize +
        [
            al.Flip(p=0.5),
            al.OneOf([
                al.RandomRotate90(),
                al.Rotate(limit=180),
            ], p=0.5),
            al.OneOf([
                al.ShiftScaleRotate(),
                al.OpticalDistortion(),
                al.GridDistortion(),
                al.ElasticTransform(),
            ], p=0.3),
            al.RandomGridShuffle(p=0.05),
            al.OneOf([
                al.RandomGamma(),
                al.HueSaturationValue(),
                al.RGBShift(),
                al.CLAHE(),
                al.ChannelShuffle(),
                al.InvertImg(),
            ], p=0.1),
            al.RandomSnow(p=0.05),
            al.RandomRain(p=0.05),
            al.RandomFog(p=0.05),
            al.RandomSunFlare(p=0.05),
            al.RandomShadow(p=0.05),
            al.RandomBrightnessContrast(p=0.05),
            al.GaussNoise(p=0.2),
            al.ISONoise(p=0.2),
            al.MultiplicativeNoise(p=0.2),
            al.ToGray(p=0.05),
            al.ToSepia(p=0.05),
            al.Solarize(p=0.05),
            al.Equalize(p=0.05),
            al.Posterize(p=0.05),
            al.FancyPCA(p=0.05),
            al.OneOf([
                al.MotionBlur(blur_limit=3),
                al.Blur(blur_limit=3),
                al.MedianBlur(blur_limit=3),
                al.GaussianBlur(blur_limit=3),
            ], p=0.05),
            al.CoarseDropout(p=0.05),
            al.Cutout(p=0.05),
            al.GridDropout(p=0.05),
            al.ChannelDropout(p=0.05),
            al.Downscale(p=0.1),
            al.ImageCompression(quality_lower=60, p=0.2),
            al.Normalize(),
            ToTensorV2()
        ])


def get_preprocess(input_size, use_crop=False, use_center_crop=False, center_crop_ratio=0.9, use_gray=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    if use_crop:
        resize = [al.Resize(int(input_size[0] * 1.1), int(input_size[1] * 1.1)),
                  al.RandomSizedCrop(min_max_height=(int(input_size[0] * 0.7), int(input_size[0] * 1.1)),
                                     height=input_size[0],
                                     width=input_size[1])]
    elif use_center_crop:
        resize = [
            al.Resize(int(input_size[0] * (2. - center_crop_ratio)), int(input_size[1] * (2. - center_crop_ratio))),
            al.CenterCrop(input_size[0], input_size[1])]
    else:
        resize = [al.Resize(input_size[0], input_size[1])]
    return al.Compose(
        resize +
        [
            al.ToGray(p=1. if use_gray else 0.),
            al.Normalize(),
            ToTensorV2()
        ])


def get_test_transforms(input_size, use_crop=False, center_crop_ratio=0.9, use_gray=False, use_pad=False):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    if use_crop:
        resize = [al.Resize(int(input_size[0] * (2 - center_crop_ratio)),
                            int(input_size[1] * (2 - center_crop_ratio))),
                  al.CenterCrop(height=input_size[0], width=input_size[1])]
    else:
        resize = [al.Resize(input_size[0], input_size[1])]

    if use_pad:
        def longest_max_size(img, interpolation=cv2.INTER_LINEAR, **params):
            img = F.longest_max_size(img, max_size=input_size[1], interpolation=interpolation)

            return img

        resize = [al.Lambda(longest_max_size),
                  al.PadIfNeeded(min_height=input_size[1], min_width=input_size[1], always_apply=True, border_mode=0), ]

    return al.Compose(resize + [
        al.ToGray(p=1. if use_gray else 0.),
        al.Normalize(),
        ToTensorV2()
    ])


def denormalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_tensor=True):
    max_pixel_value = 255.
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value
    denominator = np.reciprocal(std, dtype=np.float32)
    if is_tensor:
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
    img /= denominator
    img += mean
    if is_tensor:
        img = img.astype(np.uint8)
    return img


def get_data_loader(dataset_class, dataset_dirs, transform, batch_size, num_workers, shuffle=True, pin_memory=False,
                    label_file=None):
    if len(dataset_dirs) > 1:
        dataset_list = []
        for tmp_val in dataset_dirs:
            if label_file:
                dataset_list.append(dataset_class(tmp_val, transform, label_file=label_file))
            else:
                dataset_list.append(dataset_class(tmp_val, transform))
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        if label_file:
            dataset = dataset_class(dataset_dirs[0], transform, label_file=label_file)
        else:
            dataset = dataset_class(dataset_dirs[0], transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def get_data_loader_aug_mix(dataset_class, preprocess, dataset_dirs, batch_size, num_workers, shuffle=True,
                            pin_memory=False, no_jsd=False):
    if len(dataset_dirs) > 1:
        dataset_list = []
        for tmp_val in dataset_dirs:
            dataset_list.append(dataset_class(tmp_val, preprocess, no_jsd))
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset = dataset_class(dataset_dirs[0], preprocess, no_jsd)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
