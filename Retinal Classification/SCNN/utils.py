from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import correlate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Some functions are adapted from https://github.com/npvoid/SDNN_python


# ---------------------------------------------------------------------------
# Dataset location helpers
# ---------------------------------------------------------------------------

def resolve_retinal_dataset_root():
    """Locate the APTOS imagefolder directory used by the retinal dataset."""
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent

    candidates = [
        Path.cwd() / "Retinal Classification" / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
        Path.cwd() / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
        repo_root / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
        repo_root / "Retinal Classification" / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
        module_dir.parent / "dataset" / "aptos2019-blindness-detection" / "imagefolder",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the retinal dataset imagefolder directory."
    )


def _iter_labeled_image_paths(split_dir):
    split_dir = Path(split_dir)
    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))
    return samples


def _subsample_entries(entries, data_prop, seed=0):
    if data_prop >= 1:
        return list(entries)
    rng = np.random.default_rng(seed)
    entries = list(entries)
    if len(entries) == 0:
        return entries
    target_count = max(1, int(len(entries) * data_prop))
    indices = rng.choice(len(entries), target_count, replace=False)
    indices.sort()
    return [entries[i] for i in indices]


# ---------------------------------------------------------------------------
# Augmentation helpers (applied at raw image level before DoG encoding)
# ---------------------------------------------------------------------------

def _augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply random augmentations to a single grayscale image (H, W) float32.
    Augmentations: horizontal/vertical flip, 90-degree rotation increments,
    and small brightness jitter.  Kept lightweight so they compose well with
    the DoG → spike pipeline.
    """
    # Random horizontal flip
    if rng.random() > 0.5:
        img = np.fliplr(img)
    # Random vertical flip
    if rng.random() > 0.5:
        img = np.flipud(img)
    # Random 90° rotation (0, 90, 180, 270)
    k = rng.integers(0, 4)
    if k > 0:
        img = np.rot90(img, k=k)
    # Small brightness jitter (±10% of max range)
    jitter = rng.uniform(-25.5, 25.5)
    img = np.clip(img + jitter, 0, 255)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_retinal_split(split_dir, data_prop=1.0, image_size=64, seed=0, grayscale=True):
    """Load a class-organized retinal split from disk."""
    entries = _iter_labeled_image_paths(split_dir)
    entries = _subsample_entries(entries, data_prop=data_prop, seed=seed)

    images, labels = [], []
    for image_path, label in entries:
        with Image.open(image_path) as image:
            image = image.convert("L") if grayscale else image.convert("RGB")
            if image_size is not None:
                image = image.resize((image_size, image_size))
            images.append(np.asarray(image, dtype=np.float32))
            labels.append(int(label))

    if not images:
        raise RuntimeError(f"No image files found under {split_dir}.")

    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def load_retinal_dataset(data_prop=1, image_size=64, seed=0):
    """Load the retinal dataset from the APTOS imagefolder splits."""
    dataset_root = resolve_retinal_dataset_root()
    X_train, y_train = load_retinal_split(dataset_root / "train", data_prop=data_prop,
                                           image_size=image_size, seed=seed)
    X_test,  y_test  = load_retinal_split(dataset_root / "val",   data_prop=1.0,
                                           image_size=image_size, seed=seed)

    if X_train.ndim != 3:
        raise ValueError("Expected loaded retinal images to be 3D arrays after grayscale conversion.")

    input_shape = X_train[0].shape
    return X_train, y_train, X_test, y_test, input_shape


# ---------------------------------------------------------------------------
# Oversampling  (image-level, before spike encoding)
# ---------------------------------------------------------------------------

def oversample_to_balanced(X: np.ndarray, y: np.ndarray,
                            strategy: str = "oversample",
                            augment: bool = True,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance class frequencies at the raw-image level before DoG/spike encoding.

    Parameters
    ----------
    X        : (N, H, W) float32 grayscale images
    y        : (N,)      int64  labels
    strategy : "oversample"  – repeat minority samples with augmentation
               "hybrid"      – oversample minorities + undersample majority to 2× median
    augment  : apply random flip/rotate/jitter when generating synthetic copies
    seed     : RNG seed for reproducibility

    Returns
    -------
    X_bal, y_bal  – shuffled balanced arrays
    """
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    classes = sorted(counts.keys())

    if strategy == "oversample":
        target = max(counts.values())
    elif strategy == "hybrid":
        median_count = int(np.median(list(counts.values())))
        target = max(median_count * 2, max(counts.values()) // 2)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    X_parts, y_parts = [X], [y]

    for cls in classes:
        mask = y == cls
        X_cls = X[mask]
        n_have = len(X_cls)
        n_need = target - n_have
        if n_need <= 0:
            continue

        # Draw indices with replacement to fill gap
        idx = rng.choice(n_have, size=n_need, replace=True)
        X_extra = X_cls[idx].copy()

        if augment:
            X_extra = np.stack([_augment_image(x, rng) for x in X_extra])

        X_parts.append(X_extra)
        y_parts.append(np.full(n_need, cls, dtype=np.int64))

    X_bal = np.concatenate(X_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def get_class_weights(y: np.ndarray) -> dict:
    """
    Compute per-class inverse-frequency weights (sklearn-style 'balanced').
    Returns {class_id: weight}.
    """
    counts = Counter(y.tolist())
    n_total = len(y)
    n_classes = len(counts)
    return {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}


# ---------------------------------------------------------------------------
# DoG + spike encoding
# ---------------------------------------------------------------------------

def spike_encoding(img, nb_timesteps):
    """
    Encode an image into spikes using temporal coding based on pixel intensity.

    Args:
        img         : ndarray (H, W)
        nb_timesteps: int – number of spike bins
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        I, lat = np.argsort(1 / img.flatten()), np.sort(1 / img.flatten())
    I = np.delete(I, np.where(lat == np.inf))
    II = np.unravel_index(I, img.shape)
    t_step = np.ceil(np.arange(I.size) / (I.size / (nb_timesteps - 1))).astype(np.uint8)
    II = (t_step,) + II
    spike_times = np.zeros((nb_timesteps, img.shape[0], img.shape[1]), dtype=np.uint8)
    spike_times[II] = 1
    return spike_times


def DoG_filter(img, filt, threshold):
    """
    Apply a DoG filter on the given image.

    Args:
        img       : ndarray (H, W)
        filt      : ndarray – DoG filter kernel
        threshold : int     – contrast threshold
    """
    img = correlate(img, filt, mode='constant')
    border = np.zeros(img.shape)
    border[5:-5, 5:-5] = 1.
    img = img * border
    img = (img >= threshold).astype(int) * img
    img = np.abs(img)
    return img


def DoG(size, s1, s2):
    """
    Create a Difference-of-Gaussians filter.

    Args:
        size : int – filter size
        s1   : int – std of inner Gaussian
        s2   : int – std of outer Gaussian
    """
    r = np.arange(size) + 1
    x = np.tile(r, [size, 1])
    y = x.T
    d2 = (x - size / 2. - 0.5) ** 2 + (y - size / 2. - 0.5) ** 2
    filt = (1 / np.sqrt(2 * np.pi)) * (
        1 / s1 * np.exp(-d2 / (2 * s1 ** 2)) -
        1 / s2 * np.exp(-d2 / (2 * s2 ** 2))
    )
    filt -= np.mean(filt)
    filt /= np.amax(filt)
    return filt


def preprocess_retinal_images(dataset, nb_timesteps, filters, threshold):
    """
    Apply DoG filtering + spike encoding to a batch of grayscale images.

    dataset  : (N, H, W) float32
    Returns  : (N, T, C, H, W) uint8  where C = len(filters)
    """
    nb_channels = len(filters)
    samples, height, width = dataset.shape
    out = np.zeros((samples, nb_timesteps, nb_channels, height, width), dtype=np.uint8)
    for i, img in enumerate(dataset):
        encoded_img = np.zeros((nb_channels, nb_timesteps, height, width))
        for f, filt in enumerate(filters):
            dog_img = DoG_filter(img, filt, threshold)
            encoded_img[f] = spike_encoding(dog_img, nb_timesteps)
        out[i] = np.swapaxes(encoded_img, 0, 1)
    return out


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_encoded_retinal_dataset(
    data_prop=1,
    nb_timesteps=15,
    image_size=64,
    threshold=15,
    filters=None,
    seed=0,
    oversample=True,
    oversample_strategy="oversample",
    augment_minority=True,
):
    """
    Load, optionally balance, and spike-encode the retinal dataset.

    Parameters
    ----------
    data_prop            : fraction of training data to load (1 = all)
    nb_timesteps         : number of spike time bins
    image_size           : resize images to (image_size × image_size)
    threshold            : DoG contrast threshold
    filters              : list of DoG filter kernels (default: on-centre + off-centre)
    seed                 : RNG seed
    oversample           : if True, balance classes before encoding
    oversample_strategy  : "oversample" or "hybrid"
    augment_minority     : apply augmentation when oversampling minority classes

    Returns
    -------
    X_train_enc, y_train, X_test_enc, y_test
        X_*_enc : (N, T, C, H, W) uint8 spike tensors
    """
    if filters is None:
        filters = [DoG(7, 1, 2), DoG(7, 2, 1)]   # on-centre, off-centre

    X_train, y_train, X_test, y_test, _ = load_retinal_dataset(
        data_prop=data_prop, image_size=image_size, seed=seed,
    )

    if oversample:
        print(f"[utils] Class distribution before balancing: {sorted(Counter(y_train.tolist()).items())}")
        X_train, y_train = oversample_to_balanced(
            X_train, y_train,
            strategy=oversample_strategy,
            augment=augment_minority,
            seed=seed,
        )
        print(f"[utils] Class distribution after  balancing: {sorted(Counter(y_train.tolist()).items())}")

    print("[utils] Encoding train spikes …")
    X_train_enc = preprocess_retinal_images(X_train, nb_timesteps, filters, threshold)
    print("[utils] Encoding test  spikes …")
    X_test_enc  = preprocess_retinal_images(X_test,  nb_timesteps, filters, threshold)

    return X_train_enc, y_train, X_test_enc, y_test


# ---------------------------------------------------------------------------
# Few-shot support utilities
# ---------------------------------------------------------------------------

def build_few_shot_support(X_enc: np.ndarray, y: np.ndarray,
                            k_shot: int = 5,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample k_shot examples per class from an encoded dataset to form a
    few-shot support set.

    Parameters
    ----------
    X_enc  : (N, T, C, H, W) uint8 – encoded spike tensors
    y      : (N,) int64 labels
    k_shot : number of examples per class
    seed   : RNG seed

    Returns
    -------
    X_support, y_support  – (n_classes * k_shot, T, C, H, W) and labels
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    X_parts, y_parts = [], []
    for cls in classes:
        mask = np.where(y == cls)[0]
        chosen = rng.choice(mask, size=min(k_shot, len(mask)), replace=False)
        X_parts.append(X_enc[chosen])
        y_parts.append(y[chosen])
    return np.concatenate(X_parts), np.concatenate(y_parts)


def compute_class_prototypes(features: np.ndarray,
                              labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-class mean prototype vectors from extracted SCNN features.

    Parameters
    ----------
    features : (N, D) float – SCNN readout feature vectors
    labels   : (N,)   int   – class labels

    Returns
    -------
    prototypes : (n_classes, D) float
    classes    : (n_classes,)   int   – ordered class ids
    """
    classes = np.unique(labels)
    prototypes = np.stack([features[labels == cls].mean(axis=0) for cls in classes])
    return prototypes, classes


def few_shot_predict(query_features: np.ndarray,
                     prototypes: np.ndarray,
                     classes: np.ndarray) -> np.ndarray:
    """
    Predict labels for query samples using nearest-prototype (Euclidean) rule.

    Parameters
    ----------
    query_features : (Q, D) float
    prototypes     : (n_classes, D) float
    classes        : (n_classes,)   int

    Returns
    -------
    y_pred : (Q,) int  – predicted class ids
    """
    # Squared Euclidean distances  (Q, n_classes)
    diff = query_features[:, np.newaxis, :] - prototypes[np.newaxis, :, :]  # (Q, C, D)
    dists = (diff ** 2).sum(axis=-1)                                          # (Q, C)
    return classes[np.argmin(dists, axis=1)]