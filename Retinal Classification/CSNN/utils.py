"""
Model and data related util functions for CSNN experiments.

Adapted from Haoyi Zhu's SNN_Image_Classification utilities.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate
from snntorch import utils as snn_utils


# ---------------------------------------------------------------------------
# Dataset location helpers
# ---------------------------------------------------------------------------

def resolve_retinal_imagefolder() -> Path:
	"""Locate the APTOS imagefolder directory used by the retinal dataset."""
	module_dir = Path(__file__).resolve().parent
	retinal_root = module_dir.parent
	repo_root = retinal_root.parent

	candidates = [
		Path.cwd() / "Retinal Classification" / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
		Path.cwd() / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
		retinal_root / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
		repo_root / "Retinal Classification" / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
		repo_root / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
	]

	for candidate in candidates:
		if (candidate / "train").exists() and (candidate / "val").exists():
			return candidate.resolve()

	raise FileNotFoundError("Could not locate the retinal dataset imagefolder directory.")


def _get_targets(dataset) -> np.ndarray:
	if isinstance(dataset, Subset):
		base_targets = _get_targets(dataset.dataset)
		return np.asarray([base_targets[i] for i in dataset.indices], dtype=np.int64)
	if hasattr(dataset, "targets"):
		return np.asarray(dataset.targets, dtype=np.int64)
	if hasattr(dataset, "labels"):
		return np.asarray(dataset.labels, dtype=np.int64)
	raise AttributeError("Dataset does not expose targets or labels.")


def get_targets(dataset) -> np.ndarray:
	"""Public wrapper for extracting dataset targets."""
	return _get_targets(dataset)


def _stratified_subset_indices(targets: Iterable[int], data_prop: float, seed: int) -> np.ndarray:
	if data_prop >= 1:
		return np.arange(len(targets))
	targets = np.asarray(list(targets), dtype=np.int64)
	rng = np.random.default_rng(seed)
	indices = []
	for cls in np.unique(targets):
		cls_idx = np.where(targets == cls)[0]
		n_keep = max(1, int(round(len(cls_idx) * data_prop)))
		chosen = rng.choice(cls_idx, size=n_keep, replace=False)
		indices.extend(chosen.tolist())
	rng.shuffle(indices)
	return np.asarray(indices, dtype=np.int64)


def subset_dataset(dataset, data_prop: float, seed: int = 0):
	if data_prop >= 1:
		return dataset
	targets = _get_targets(dataset)
	indices = _stratified_subset_indices(targets, data_prop, seed)
	return Subset(dataset, indices)


def describe_class_distribution(targets: Iterable[int]) -> dict[int, int]:
	counts = Counter(int(t) for t in targets)
	return {int(k): int(v) for k, v in sorted(counts.items())}


def compute_class_weights(targets: Iterable[int]) -> torch.Tensor:
	counts = Counter(int(t) for t in targets)
	n_total = sum(counts.values())
	n_classes = len(counts)
	weights = {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}
	return torch.tensor([weights[cls] for cls in sorted(weights)], dtype=torch.float32)


def build_weighted_sampler(targets: Iterable[int]) -> WeightedRandomSampler:
	targets = list(int(t) for t in targets)
	counts = Counter(targets)
	weights = [1.0 / counts[t] for t in targets]
	return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_retinal_dataloaders(
	batch_size: int = 32,
	image_size: int = 64,
	data_prop: float = 1.0,
	num_workers: int = 0,
	seed: int = 0,
	augment: bool = False,
	grayscale: bool = True,
	normalize: bool = True,
	balance: bool = False,
):
	"""Build dataloaders for the retinal imagefolder dataset."""
	dataset_root = resolve_retinal_imagefolder()
	train_dir = dataset_root / "train"
	val_dir = dataset_root / "val"

	base_transforms = [transforms.Resize((image_size, image_size))]
	if grayscale:
		base_transforms.append(transforms.Grayscale(num_output_channels=1))

	train_transforms = base_transforms.copy()
	if augment:
		train_transforms.extend(
			[
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(10),
			]
		)
	train_transforms.append(transforms.ToTensor())

	val_transforms = base_transforms + [transforms.ToTensor()]

	if normalize:
		if grayscale:
			train_transforms.append(transforms.Normalize((0.5,), (0.5,)))
			val_transforms.append(transforms.Normalize((0.5,), (0.5,)))
		else:
			train_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
			val_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

	train_dataset = datasets.ImageFolder(str(train_dir), transform=transforms.Compose(train_transforms))
	val_dataset = datasets.ImageFolder(str(val_dir), transform=transforms.Compose(val_transforms))

	train_dataset = subset_dataset(train_dataset, data_prop, seed)

	train_targets = _get_targets(train_dataset)
	class_weights = compute_class_weights(train_targets)

	if balance:
		sampler = build_weighted_sampler(train_targets)
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			sampler=sampler,
			drop_last=False,
			num_workers=num_workers,
		)
	else:
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			drop_last=False,
			num_workers=num_workers,
		)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=num_workers,
	)

	return train_loader, val_loader, class_weights


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_model(
	device: torch.device,
	input_channels: int = 1,
	input_size: int = 64,
	num_classes: int = 5,
	slope: int = 25,
	beta: float = 0.5,
	spike: bool = False,
):
	"""
	Build a CSNN with architecture of 12C5-MP2-64C5-MP2-FC.
	"""
	spike_grad = surrogate.fast_sigmoid(slope=slope)

	conv1 = nn.Conv2d(input_channels, 12, 5)
	pool1 = nn.MaxPool2d(2)
	lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
	conv2 = nn.Conv2d(12, 64, 5)
	pool2 = nn.MaxPool2d(2)
	lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

	with torch.no_grad():
		dummy = torch.zeros(1, input_channels, input_size, input_size)
		feat = pool2(conv2(pool1(conv1(dummy))))
		flat_dim = int(feat.view(1, -1).shape[1])

	m = nn.Sequential(
		conv1,
		pool1,
		lif1,
		conv2,
		pool2,
		lif2,
		nn.Flatten(),
		nn.Linear(flat_dim, num_classes),
		snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
	).to(device)

	return m


def forward_pass(
	m: Callable,
	data: torch.Tensor,
	num_steps: int = 50,
	spike: bool = False,
):
	"""Forward pass function of CSNN."""
	mem_rec = []
	spk_rec = []
	snn_utils.reset(m)

	if spike:
		for step in range(data.size(0)):
			spk_out, mem_out = m(data[step])
			spk_rec.append(spk_out)
			mem_rec.append(mem_out)
	else:
		for _ in range(num_steps):
			spk_out, mem_out = m(data)
			spk_rec.append(spk_out)
			mem_rec.append(mem_out)

	return torch.stack(spk_rec), torch.stack(mem_rec)


def plot_history(train_loss_hist, val_acc_hist, save_path: Path):
	"""Plot training loss and validation accuracy curves."""
	save_path = Path(save_path)
	save_path.mkdir(parents=True, exist_ok=True)

	fig = plt.figure(facecolor="w")
	plt.plot(train_loss_hist)
	plt.title("Train Set Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(save_path / "train_loss.png")
	plt.close(fig)

	fig = plt.figure(facecolor="w")
	plt.plot(val_acc_hist)
	plt.title("Val Set Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.savefig(save_path / "val_acc.png")
	plt.close(fig)

