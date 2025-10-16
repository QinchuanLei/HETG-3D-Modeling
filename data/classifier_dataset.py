import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop,
    RandomResizedCrop, RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode
from typing import Any
import json
from torch.optim.lr_scheduler import _LRScheduler
# ===== 图像增强参数 =====
INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

SIZE = (224, 224)
INTERPOLATION = INTERPOLATION_MODES["bicubic"]
PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
CROP_PADDING = 0
RRCROP_SCALE = (0.08, 1.0)

# ===== 构建 transform 函数 =====
def build_transform(image_augmentation,
                    size=SIZE,
                    interpolation=INTERPOLATION,
                    pixel_mean=PIXEL_MEAN,
                    pixel_std=PIXEL_STD,
                    crop_padding=CROP_PADDING,
                    rrcrop_scale=RRCROP_SCALE):
    normalize = Normalize(mean=pixel_mean, std=pixel_std)

    if image_augmentation == "none":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            CenterCrop(size=size),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "flip":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            CenterCrop(size=size),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "randomcrop":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            RandomCrop(size=size, padding=crop_padding),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "randomresizedcrop":
        transform = Compose([
            RandomResizedCrop(size=size, scale=rrcrop_scale, interpolation=interpolation),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Invalid image augmentation method: {image_augmentation}")
        
    return transform
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def load_json(json_location, default_obj=None):
    '''Load a json file.'''
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj
class DatasetWrapper(Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output
    
AVAI_SCHEDS = ["cosine", "linear"]
AVAI_WARMUP_SCHEDS = ["constant", "linear"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]
def build_lr_scheduler(optimizer,
                       lr_scheduler,
                       warmup_iter,
                       max_iter,
                       warmup_type=None,
                       warmup_lr=None,
                       verbose=False):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str): learning rate scheduler name, either "cosine" or "linear".
        warmup_iter (int): number of warmup iterations.
        max_iter (int): maximum iteration (not including warmup iter).
        warmup_type (str): warmup type, either constant or linear.
        warmup_lr (float): warmup learning rate.
        verbose (bool): If ``True``, prints a message to stdout
    """
    if verbose:
        print(f"Building scheduler: {lr_scheduler} with warmup: {warmup_type}")

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_iter)
        )
    elif lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: 1 - x / float(max_iter),
            last_epoch=-1
        )
        
    if warmup_iter > 0:
        if warmup_type not in AVAI_WARMUP_SCHEDS:
            raise ValueError(
                f"warmup_type must be one of {AVAI_WARMUP_SCHEDS}, "
                f"but got {warmup_type}"
            )

        if warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, warmup_iter,
                warmup_lr
            )

        elif warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, warmup_iter,
                warmup_lr
            )

        else:
            raise ValueError

    return scheduler