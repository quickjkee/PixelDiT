# Modified from https://github.com/MCG-NJU/PixNerd and https://github.com/End2End-Diffusion/REPA-E/

import copy
import io
import json
import os
import random
import re
import time
import unicodedata
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import PIL.Image
import torch
import torchvision
import yaml
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir: str):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)

        labels = self._load_h5_file("dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _load_h5_file(self, path):
        if path.endswith('.png'):
            rtn = np.array(PIL.Image.open(io.BytesIO(np.array(self.h5f[path]))))
            rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
        elif path.endswith('.json'):
            rtn = json.loads(np.array(self.h5f[path]).tobytes().decode('utf-8'))
        elif path.endswith('.npy'):
            rtn = np.array(self.h5f[path])
        else:
            raise ValueError(f'Unknown file type: {path}')
        return rtn

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        image_fname = self.filelist[index]
        image = self._load_h5_file(image_fname)

        image_tensor = torch.from_numpy(image).float() / 255.0
        normalized_image = (image_tensor - 0.5) / 0.5

        target = int(self.labels[index])
        metadata = {
            "raw_image": image_tensor,
            "class": target,
        }
        return normalized_image, target, metadata


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM (same as used in JiT / REPA).
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageNetFolderDataset(Dataset):
    """
    Raw-ImageNet loader (JiT-style) that is a drop-in replacement for
    CustomINH5Dataset: it reads images directly from `<data_dir>/train`
    with torchvision.datasets.ImageFolder, so no .h5 preprocessing is needed.

    Returns the same tuple the REPA trainer expects:
      normalized_image : float [3, H, W] in [-1, 1]
      target           : int class label
      metadata         : {"raw_image": float [3, H, W] in [0, 1], "class": int}
    """

    def __init__(self, data_dir: str, image_size: int = 256, random_flip: bool = True, split: str = "train"):
        root = os.path.join(data_dir, split)
        self.dataset = torchvision.datasets.ImageFolder(root)
        self.image_size = image_size
        self.random_flip = random_flip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pil_image, target = self.dataset[index]
        pil_image = pil_image.convert("RGB")
        pil_image = center_crop_arr(pil_image, self.image_size)
        if self.random_flip and random.random() < 0.5:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        arr = np.array(pil_image)  # [H, W, 3] uint8
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0  # [3, H, W] in [0, 1]
        normalized_image = (image_tensor - 0.5) / 0.5  # [-1, 1]

        target = int(target)
        metadata = {
            "raw_image": image_tensor,
            "class": target,
        }
        return normalized_image, target, metadata


def _clean_filename(s: str) -> str:
    s = s.strip().strip('.')
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    illegal_chars = r'[/]'
    s = re.sub(illegal_chars, '_', s)
    s = re.sub(r'_{2,}', '_', s)
    s = s.lower()
    max_length = 200
    s = s[:max_length]
    if not s:
        return 'untitled'
    return s


def _save_fn(image, metadata, root_path):
    image_path = os.path.join(root_path, f"{metadata['filename']}.png")
    PIL.Image.fromarray(image).save(image_path)


class RandomNDataset(Dataset):
    def __init__(self, latent_shape=(4, 64, 64), conditions: Union[int, List[Any], str] = None,
                 seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if isinstance(conditions, int):
            conditions = list(range(conditions))
        elif isinstance(conditions, str):
            if os.path.exists(conditions):
                conditions = open(conditions, "r").read().splitlines()
            else:
                raise FileNotFoundError(conditions)
        elif isinstance(conditions, list):
            conditions = conditions
        self.conditions = conditions
        self.num_conditons = len(conditions)
        self.seeds = seeds

        if num_samples_per_instance > 0:
            max_num_instances = num_samples_per_instance * self.num_conditons
        else:
            max_num_instances = max_num_instances

        if seeds is not None:
            self.max_num_instances = len(seeds) * self.num_conditons
            self.num_seeds = len(seeds)
        else:
            self.num_seeds = (max_num_instances + self.num_conditons - 1) // self.num_conditons
            self.max_num_instances = self.num_seeds * self.num_conditons
        self.latent_shape = latent_shape

    def __getitem__(self, idx):
        condition = self.conditions[idx // self.num_seeds]

        seed = random.randint(0, 1 << 31)
        if self.seeds is not None:
            seed = self.seeds[idx % self.num_seeds]

        filename = f"{_clean_filename(str(condition))}_{seed}"
        generator = torch.Generator().manual_seed(seed)
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        metadata = dict(
            filename=filename,
            seed=seed,
            condition=condition,
            save_fn=_save_fn,
        )
        return latent, condition, metadata

    def __len__(self):
        return self.max_num_instances


class ClassLabelRandomNDataset(RandomNDataset):
    def __init__(self, latent_shape=(4, 64, 64), num_classes=1000, conditions: Union[int, List[Any], str] = None,
                 seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if conditions is None:
            conditions = list(range(num_classes))
        super().__init__(latent_shape, conditions, seeds, max_num_instances, num_samples_per_instance)


def mirco_batch_collate_fn(batch):
    batch = copy.deepcopy(batch)
    new_batch = []
    for micro_batch in batch:
        new_batch.extend(micro_batch)
    x, y, metadata = list(zip(*new_batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except Exception:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata


def collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except Exception:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata


def eval_collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    x = torch.stack(x, dim=0)
    return x, y, metadata


def create_dataloader(dataloader_config_path: str, batch_size: int, skip_rows: int = 0):
    """
    Build a yt_tools IterableDataloader from a YAML config (same mechanism as
    JiT's util/crop.create_dataloader). Imports of yt_tools/omegaconf are lazy
    so this module still loads on clusters where yt_tools is unavailable.
    """
    from omegaconf import OmegaConf
    from yt_tools.utils import instantiate_from_config

    with open(dataloader_config_path) as f:
        dataloader_config = OmegaConf.create(yaml.load(f, Loader=yaml.SafeLoader))
    dataloader_config["params"]["batch_size"] = batch_size
    return instantiate_from_config(dataloader_config, skip_rows=skip_rows)


class YTBatchAdapter:
    """
    Wraps a yt_tools IterableDataloader (which yields already-batched dicts like
    {"image": [B,3,H,W] in [0,255], "label": [...]}) and converts each batch into
    PixelDiT's training contract consumed in src/lightning.py:
        x        : float [B,3,H,W] in [-1, 1]
        y        : list[int] class labels
        metadata : {"raw_image": float [B,3,H,W] in [0,1], "class": list[int]}
    """

    def __init__(self, yt_loader, image_key: str = "image", label_key: str = "label"):
        self.yt_loader = yt_loader
        self.image_key = image_key
        self.label_key = label_key

    def __iter__(self):
        for batch in self.yt_loader:
            img = batch[self.image_key]
            if not torch.is_tensor(img):
                img = torch.as_tensor(np.asarray(img))
            img = img.float()
            # [B,H,W,3] -> [B,3,H,W] if needed
            if img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2).contiguous()
            # [0,255] -> [0,1]
            if img.max() > 1.5:
                img = img / 255.0

            raw_image = img                       # DINOv2 encoder expects [0,1]
            x = (img - 0.5) / 0.5                  # [-1, 1]
            y = [int(v) for v in batch[self.label_key]]
            metadata = {"raw_image": raw_image, "class": y}
            yield x, y, metadata

    def __len__(self):
        return len(self.yt_loader)


class DataModule(LightningDataModule):
    def __init__(self,
                 train_dataset: Dataset = None,
                 eval_dataset: Dataset = None,
                 pred_dataset: Dataset = None,
                 train_batch_size: int = 64,
                 train_num_workers: int = 16,
                 train_prefetch_factor: int = 8,
                 eval_batch_size: int = 32,
                 eval_num_workers: int = 4,
                 pred_batch_size: int = 32,
                 pred_num_workers: int = 4,
                 yt_config_path: str = None,
                 seed: int = None):
        super().__init__()
        self.yt_config_path = yt_config_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pred_dataset = pred_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size
        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers
        self.seed = seed if seed is not None else int(time.time())
        self._train_dataloader: Optional[DataLoader] = None

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # YT-table path (JiT-style): the yt_tools loader yields fully-batched
        # dicts, so we bypass torch Dataset/sampler/collate and adapt batches.
        if self.yt_config_path is not None:
            yt_loader = create_dataloader(self.yt_config_path, self.train_batch_size)
            self._train_dataloader = YTBatchAdapter(yt_loader)
            return self._train_dataloader

        micro_batch_size = getattr(self.train_dataset, "micro_batch_size", None)
        if micro_batch_size is not None:
            assert self.train_batch_size % micro_batch_size == 0
            dataloader_batch_size = self.train_batch_size // micro_batch_size
            train_collate_fn = mirco_batch_collate_fn
        else:
            dataloader_batch_size = self.train_batch_size
            train_collate_fn = collate_fn

        if not isinstance(self.train_dataset, IterableDataset):
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                seed=int(self.seed) if self.seed is not None else 0
            )
        else:
            sampler = None

        self._train_dataloader = DataLoader(
            self.train_dataset,
            dataloader_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(
            self.eval_dataset,
            self.eval_batch_size,
            num_workers=self.eval_num_workers,
            prefetch_factor=2,
            sampler=sampler,
            collate_fn=eval_collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(
            self.pred_dataset,
            batch_size=self.pred_batch_size,
            num_workers=self.pred_num_workers,
            prefetch_factor=4,
            sampler=sampler,
            collate_fn=eval_collate_fn,
        )

