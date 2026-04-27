import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.functional import conv2d, max_pool2d

from utils import (
    load_encoded_retinal_dataset,
    get_class_weights,
    build_few_shot_support,
    compute_class_prototypes,
    few_shot_predict,
)

"""
Upgraded STDP-based Spiking Convolutional Neural Network for retinal classification.

Key improvements over the original:
  1. Deeper architecture – 3 conv+pool blocks (was 2) with more filters per layer.
  2. STDP convergence threshold is per-layer configurable and defaults to a
     much lower value so training does NOT stop prematurely.
  3. Balanced SVC readout  (class_weight='balanced') fixes minority-class bias.
  4. Prototypical few-shot readout  – nearest-prototype classifier using the
     SCNN spike features as embeddings. Useful when minority classes are scarce.
  5. Rich feature extraction  – per-channel (max, sum, mean, first-spike-time)
     giving 4× more signal to the readout than max or sum alone.
  6. Training loop shuffles data each epoch and supports class-stratified order
     so rare classes are seen by STDP more uniformly.

Architecture (default, 64×64 input, 2 DoG channels):

  Input  (T=15, C=2, H=64, W=64)
    │
  Conv1  – 32 filters, 5×5, pad=2  → (32, 64, 64)   fires ~T1 spikes
  Pool1  – 2×2 stride-2            → (32, 32, 32)
    │
  Conv2  – 64 filters, 5×5, pad=2  → (64, 32, 32)
  Pool2  – 2×2 stride-2            → (64, 16, 16)
    │
  Conv3  – 128 filters, 3×3, pad=1 → (128, 16, 16)
  Pool3  – 2×2 stride-2            → (128,  8,  8)
    │
  Feature extraction  (max | sum | mean | first-spike per channel per spatial loc)
    └─► flattened feature vector → LinearSVC (balanced) or Prototypical classifier

References:
  [1] Kheradpisheh et al. (2018). STDP-based spiking deep CNNs for object recognition.
      Neural Networks, 99, 56–67. https://doi.org/10.1016/J.NEUNET.2017.12.005
  [2] Mozafari et al. (2019). SpykeTorch. Front. Neurosci., 13, 625.
  [3] Snell et al. (2017). Prototypical Networks for Few-shot Learning. NeurIPS.
"""


# ---------------------------------------------------------------------------
# Spiking primitives
# ---------------------------------------------------------------------------

class SpikingPool:
    """Max-pooling layer where each output neuron fires at most once."""

    def __init__(self, input_shape, kernel_size, stride, padding=0):
        in_channels, in_height, in_width = input_shape
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride      = stride      if isinstance(stride, tuple)      else (stride, stride)
        self.padding     = padding     if isinstance(padding, tuple)     else (padding, padding)
        oh = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        ow = int(((in_width  + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.output_shape   = (in_channels, oh, ow)
        self.active_neurons = np.ones(self.output_shape, dtype=bool)

    def reset(self):
        self.active_neurons[:] = True

    def __call__(self, in_spks):
        in_spks = np.pad(in_spks,
                         ((0,), (self.padding[0],), (self.padding[1],)),
                         mode='constant')
        in_spks  = torch.as_tensor(in_spks, dtype=torch.float32).unsqueeze(0)
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride).numpy()[0]
        out_spks = out_spks * self.active_neurons
        self.active_neurons[out_spks == 1] = False
        return out_spks


class SpikingConv:
    """
    Convolutional layer with integrate-and-fire neurons (fire once) and
    winner-take-all STDP learning.
    """

    def __init__(
        self, input_shape, out_channels, kernel_size, stride, padding=0,
        nb_winners=1, firing_threshold=1, stdp_max_iter=None, adaptive_lr=False,
        stdp_a_plus=0.004, stdp_a_minus=-0.003, stdp_a_max=0.15,
        inhibition_radius=0, update_lr_cnt=500,
        weight_init_mean=0.8, weight_init_std=0.05, v_reset=0,
    ):
        in_channels, in_height, in_width = input_shape
        self.out_channels     = out_channels
        self.kernel_size      = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride           = stride      if isinstance(stride, tuple)      else (stride, stride)
        self.padding          = padding     if isinstance(padding, tuple)     else (padding, padding)
        self.firing_threshold = firing_threshold
        self.v_reset          = v_reset

        self.weights = np.random.normal(
            loc=weight_init_mean, scale=weight_init_std,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
        )

        oh = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        ow = int(((in_width  + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.pot            = np.zeros((out_channels, oh, ow))
        self.active_neurons = np.ones(self.pot.shape, dtype=bool)
        self.output_shape   = self.pot.shape

        # STDP state
        self.recorded_spks  = np.zeros((in_channels,
                                        in_height  + 2 * self.padding[0],
                                        in_width   + 2 * self.padding[1]))
        self.nb_winners        = nb_winners
        self.inhibition_radius = inhibition_radius
        self.adaptive_lr       = adaptive_lr
        self.a_plus            = stdp_a_plus
        self.a_minus           = stdp_a_minus
        self.a_max             = stdp_a_max
        self.stdp_cnt          = 0
        self.update_lr_cnt     = update_lr_cnt
        self.stdp_max_iter     = stdp_max_iter
        self.plasticity        = True
        self.stdp_neurons      = np.ones(self.pot.shape, dtype=bool)

    # ------------------------------------------------------------------
    def get_learning_convergence(self):
        return (self.weights * (1 - self.weights)).sum() / np.prod(self.weights.shape)

    def reset(self):
        self.pot[:]            = self.v_reset
        self.active_neurons[:] = True
        self.stdp_neurons[:]   = True
        self.recorded_spks[:]  = 0

    # ------------------------------------------------------------------
    def get_winners(self):
        winners   = []
        channels  = np.arange(self.pot.shape[0])
        pots_tmp  = np.copy(self.pot) * self.stdp_neurons
        while len(winners) < self.nb_winners:
            winner = np.unravel_index(np.argmax(pots_tmp), pots_tmp.shape)
            if pots_tmp[winner] <= self.firing_threshold:
                break
            winners.append(winner)
            pots_tmp[channels != winner[0],
                     max(0, winner[1] - self.inhibition_radius): winner[1] + self.inhibition_radius + 1,
                     max(0, winner[2] - self.inhibition_radius): winner[2] + self.inhibition_radius + 1
                     ] = self.v_reset
            pots_tmp[winner[0]] = self.v_reset
        return winners

    def lateral_inhibition(self, spks):
        spks_c, spks_h, spks_w = np.where(spks)
        spks_pot = np.array([self.pot[spks_c[i], spks_h[i], spks_w[i]] for i in range(len(spks_c))])
        for ind in np.argsort(spks_pot)[::-1]:
            if spks[spks_c[ind], spks_h[ind], spks_w[ind]] == 1:
                inhib = np.arange(spks.shape[0]) != spks_c[ind]
                spks[inhib, spks_h[ind], spks_w[ind]]               = 0
                self.pot[inhib, spks_h[ind], spks_w[ind]]            = self.v_reset
                self.active_neurons[inhib, spks_h[ind], spks_w[ind]] = False
        return spks

    def get_conv_of(self, input_spks, output_neuron):
        n_c, n_h, n_w = output_neuron
        input_t = torch.as_tensor(input_spks, dtype=torch.float32).unsqueeze(0)
        convs   = torch.nn.functional.unfold(
            input_t, kernel_size=self.kernel_size, stride=self.stride
        )[0].numpy()
        return convs[:, n_h * self.pot.shape[2] + n_w]

    def stdp(self, winner):
        if not self.stdp_neurons[winner]:
            return
        if not self.plasticity:
            return
        self.stdp_cnt += 1
        winner_c, winner_h, winner_w = winner
        conv    = self.get_conv_of(self.recorded_spks, winner).flatten()
        w       = self.weights[winner_c].flatten() * (1 - self.weights[winner_c]).flatten()
        w_plus  = conv > 0
        w_minus = conv == 0
        dW      = (w_plus * w * self.a_plus) + (w_minus * w * self.a_minus)
        self.weights[winner_c] += dW.reshape(self.weights[winner_c].shape)

        channels = np.arange(self.pot.shape[0])
        self.stdp_neurons[
            channels != winner_c,
            max(0, winner_h - self.inhibition_radius): winner_h + self.inhibition_radius + 1,
            max(0, winner_w - self.inhibition_radius): winner_w + self.inhibition_radius + 1,
        ] = False
        self.stdp_neurons[winner_c] = False

        if self.adaptive_lr and self.stdp_cnt % self.update_lr_cnt == 0:
            self.a_plus  = min(2 * self.a_plus, self.a_max)
            self.a_minus = -0.75 * self.a_plus

        if self.stdp_max_iter is not None and self.stdp_cnt > self.stdp_max_iter:
            self.plasticity = False

    # ------------------------------------------------------------------
    def __call__(self, spk_in, train=False):
        spk_in = np.pad(spk_in,
                        ((0,), (self.padding[0],), (self.padding[1],)),
                        mode='constant')
        self.recorded_spks += spk_in
        spk_out = np.zeros(self.pot.shape)

        x       = torch.as_tensor(spk_in, dtype=torch.float32).unsqueeze(0)
        weights = torch.as_tensor(self.weights, dtype=torch.float32)
        out_conv = conv2d(x, weights, stride=self.stride).numpy()[0]

        self.pot[self.active_neurons] += out_conv[self.active_neurons]
        output_spikes = self.pot > self.firing_threshold

        if np.any(output_spikes):
            spk_out[output_spikes] = 1
            spk_out = self.lateral_inhibition(spk_out)
            if train and self.plasticity:
                for winner in self.get_winners():
                    self.stdp(winner)
            self.pot[spk_out == 1]            = self.v_reset
            self.active_neurons[spk_out == 1] = False

        return spk_out


# ---------------------------------------------------------------------------
# Deeper SNN  (3 conv + pool blocks)
# ---------------------------------------------------------------------------

class SNN:
    """
    3-block STDP Spiking CNN for retinal classification.

    Block layout
    ────────────
    Conv1 (32 ch, 5×5) → Pool1 (2×2)
    Conv2 (64 ch, 5×5) → Pool2 (2×2)
    Conv3 (128 ch, 3×3) → Pool3 (2×2)

    For a 64×64 input this gives a (128, 8, 8) = 8 192-dim feature map.
    With 4 readout statistics (max, sum, mean, first-spike-time) the final
    feature vector is 4 × 8 192 = 32 768 dims.
    """

    def __init__(self, input_shape):
        # ── Block 1 ──────────────────────────────────────────────────────────
        conv1 = SpikingConv(
            input_shape,
            out_channels=32, kernel_size=5, stride=1, padding=2,
            nb_winners=1, firing_threshold=10,
            adaptive_lr=True, inhibition_radius=2,
            stdp_a_plus=0.004, stdp_a_minus=-0.003,
        )
        pool1 = SpikingPool(conv1.output_shape, kernel_size=2, stride=2)

        # ── Block 2 ──────────────────────────────────────────────────────────
        conv2 = SpikingConv(
            pool1.output_shape,
            out_channels=64, kernel_size=5, stride=1, padding=2,
            nb_winners=1, firing_threshold=1,
            adaptive_lr=True, inhibition_radius=1,
            stdp_a_plus=0.004, stdp_a_minus=-0.003,
        )
        pool2 = SpikingPool(conv2.output_shape, kernel_size=2, stride=2)

        # ── Block 3 ──────────────────────────────────────────────────────────
        conv3 = SpikingConv(
            pool2.output_shape,
            out_channels=128, kernel_size=3, stride=1, padding=1,
            nb_winners=2, firing_threshold=1,
            adaptive_lr=True, inhibition_radius=1,
            stdp_a_plus=0.002, stdp_a_minus=-0.0015,
        )
        pool3 = SpikingPool(conv3.output_shape, kernel_size=2, stride=2)

        self.conv_layers        = [conv1, conv2, conv3]
        self.pool_layers        = [pool1, pool2, pool3]
        self.output_shape       = pool3.output_shape
        self.nb_trainable_layers = len(self.conv_layers)
        self.recorded_sum_spks  = []

    # ------------------------------------------------------------------
    def reset(self):
        for layer in self.conv_layers:
            layer.reset()
        for layer in self.pool_layers:
            layer.reset()

    # ------------------------------------------------------------------
    def __call__(self, x, train_layer=None):
        """
        Forward pass through the 3-block SCNN.

        x           : (T, C, H, W) uint8 spike tensor
        train_layer : int or None – which conv layer to apply STDP on
        """
        self.reset()
        nb_timesteps   = x.shape[0]
        output_spikes  = np.zeros((nb_timesteps,) + self.output_shape)
        sum_spks       = 0

        for t in range(nb_timesteps):
            spk = x[t].astype(np.float64)
            sum_spks += spk.sum()

            spk = self.conv_layers[0](spk, train=(train_layer == 0))
            sum_spks += spk.sum()
            spk = self.pool_layers[0](spk)

            spk = self.conv_layers[1](spk, train=(train_layer == 1))
            sum_spks += spk.sum()
            spk = self.pool_layers[1](spk)

            spk = self.conv_layers[2](spk, train=(train_layer == 2))
            sum_spks += spk.sum()
            spk_out = self.pool_layers[2](spk)
            sum_spks += spk_out.sum()

            output_spikes[t] = spk_out

        if train_layer is None:
            self.recorded_sum_spks.append(sum_spks)

        if output_spikes.sum() == 0:
            print("[WARNING] No output spike recorded.")

        return output_spikes


# ---------------------------------------------------------------------------
# Rich feature extraction
# ---------------------------------------------------------------------------

def extract_features(spike_tensor: np.ndarray) -> np.ndarray:
    """
    Extract 4 complementary statistics from the output spike tensor
    (T, C, H, W) → flat feature vector of length 4 × C × H × W.

    Statistics per spatial location per channel:
      • max           – whether the neuron ever fired (binary-like)
      • sum           – total spike count (firing rate proxy)
      • mean          – mean activity across time
      • first_spike   – normalised latency of first spike (0 = early, 1 = late/never)
    """
    T = spike_tensor.shape[0]

    feat_max  = spike_tensor.max(axis=0).flatten()
    feat_sum  = spike_tensor.sum(axis=0).flatten()
    feat_mean = spike_tensor.mean(axis=0).flatten()

    # First spike time: index of first non-zero timestep, normalised to [0,1].
    # Neurons that never fire get value 1.0.
    first = np.full(spike_tensor.shape[1:], T, dtype=np.float32)
    for t in range(T):
        fired = (spike_tensor[t] > 0) & (first == T)
        first[fired] = t
    feat_first = (first / T).flatten()

    return np.concatenate([feat_max, feat_sum, feat_mean, feat_first])


def extract_all_features(snn: SNN, X_enc: np.ndarray,
                          desc: str = "Extracting") -> np.ndarray:
    """
    Run the full SCNN forward pass on every sample and extract rich features.

    Returns : (N, D) float32 feature matrix
    """
    features = []
    for x in tqdm(X_enc, desc=desc):
        spk = snn(x)
        features.append(extract_features(spk))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Balanced SVC readout
# ---------------------------------------------------------------------------

def train_balanced_svc(X_train: np.ndarray, y_train: np.ndarray,
                        seed: int = 0) -> LinearSVC:
    """
    Train a class-balanced LinearSVC readout.
    class_weight='balanced' automatically up-weights minority classes.
    """
    clf = LinearSVC(max_iter=5000, random_state=seed, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Prototypical (few-shot) readout
# ---------------------------------------------------------------------------

class PrototypicalReadout:
    """
    Nearest-prototype classifier built from SCNN spike feature embeddings.

    After STDP training, compute one prototype per class as the mean feature
    vector over all (or k-shot) support examples. At inference, classify by
    Euclidean distance to the nearest prototype.

    This is especially useful when minority classes (3, 4) have very few
    samples and the SVC decision boundary collapses toward the majority class.
    """

    def __init__(self):
        self.prototypes: np.ndarray | None = None
        self.classes:    np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Compute per-class mean prototypes from feature matrix."""
        self.prototypes, self.classes = compute_class_prototypes(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.prototypes is not None, "Call .fit() before .predict()"
        return few_shot_predict(features, self.prototypes, self.classes)

    def fit_k_shot(self, X_enc: np.ndarray, y: np.ndarray,
                   snn: SNN, k_shot: int = 5, seed: int = 0):
        """
        Build prototypes from only k_shot examples per class.
        Useful for the truly few-shot regime (minority classes 3 and 4).
        """
        X_sup, y_sup = build_few_shot_support(X_enc, y, k_shot=k_shot, seed=seed)
        features_sup = extract_all_features(snn, X_sup, desc=f"Prototypes ({k_shot}-shot)")
        self.fit(features_sup, y_sup)
        return self


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_snn(
    snn: SNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: list[int] | None = None,
    convergence_threshold: float = 1e-4,
    seed: int = 0,
):
    """
    Layer-wise STDP training with class-stratified shuffling per epoch.

    Parameters
    ----------
    snn                   : SNN instance
    X_train               : (N, T, C, H, W) encoded spike data
    y_train               : (N,) labels
    epochs                : list of epoch counts per conv layer (default [5, 5, 3])
    convergence_threshold : stop layer training when convergence < this value.
                            Set to 0 to always run all epochs (recommended).
    seed                  : RNG seed for shuffling
    """
    if epochs is None:
        epochs = [5, 5, 3]

    rng = np.random.default_rng(seed)

    for layer_idx in range(snn.nb_trainable_layers):
        n_epochs = epochs[layer_idx] if layer_idx < len(epochs) else epochs[-1]
        print(f"\n[STDP] Training Conv Layer {layer_idx + 1} / {snn.nb_trainable_layers}  "
              f"({n_epochs} epoch(s), convergence_threshold={convergence_threshold})")

        for epoch in range(n_epochs):
            # Stratified shuffle: interleave classes so rare classes aren't starved
            indices = _stratified_shuffle(y_train, rng)
            converged = False

            for i in tqdm(indices, desc=f"  Layer {layer_idx+1} epoch {epoch+1}/{n_epochs}"):
                snn(X_train[i], train_layer=layer_idx)
                conv_val = snn.conv_layers[layer_idx].get_learning_convergence()
                if convergence_threshold > 0 and conv_val < convergence_threshold:
                    print(f"  [early stop] convergence={conv_val:.6f} < {convergence_threshold}")
                    converged = True
                    break

            print(f"  epoch {epoch+1} done | convergence="
                  f"{snn.conv_layers[layer_idx].get_learning_convergence():.6f}")
            if converged:
                break


def _stratified_shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return index permutation that interleaves classes in round-robin fashion.
    This ensures each STDP mini-batch sees every class, even rare ones.
    """
    classes = np.unique(y)
    class_indices = {cls: rng.permutation(np.where(y == cls)[0]).tolist() for cls in classes}
    interleaved = []
    while any(class_indices[c] for c in classes):
        for cls in classes:
            if class_indices[cls]:
                interleaved.append(class_indices[cls].pop(0))
    return np.array(interleaved)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main(
    seed: int = 1,
    data_prop: float = 1.0,
    nb_timesteps: int = 15,
    epochs: list[int] | None = None,
    convergence_threshold: float = 0.0,   # 0 = run all epochs, no early stop
    oversample: bool = True,
    k_shot: int = 5,                       # shots per class for few-shot readout
):
    """
    End-to-end SCNN training + evaluation with three readout modes:
      1. Balanced LinearSVC  on rich (max+sum+mean+first-spike) features
      2. Prototypical classifier  on full training set
      3. Prototypical classifier  in k-shot mode (minority-class friendly)
    """
    if epochs is None:
        epochs = [5, 5, 3]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("=== Loading dataset ===")
    X_train, y_train, X_test, y_test = load_encoded_retinal_dataset(
        data_prop=data_prop,
        nb_timesteps=nb_timesteps,
        oversample=oversample,
        augment_minority=True,
        seed=seed,
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Model ─────────────────────────────────────────────────────────────────
    input_shape = X_train[0][0].shape  # (C, H, W) at timestep 0
    snn = SNN(input_shape)

    print(f"\nInput  shape : {X_train[0].shape}")
    print(f"Output shape : {snn.output_shape}  ({np.prod(snn.output_shape)} values)")

    # ── STDP Training ─────────────────────────────────────────────────────────
    print("\n=== STDP Training ===")
    train_snn(snn, X_train, y_train,
              epochs=epochs,
              convergence_threshold=convergence_threshold,
              seed=seed)

    # ── Feature Extraction ────────────────────────────────────────────────────
    print("\n=== Feature Extraction ===")
    F_train = extract_all_features(snn, X_train, desc="Train features")
    F_test  = extract_all_features(snn, X_test,  desc="Test  features")
    print(f"Feature vector size: {F_train.shape[1]}")

    # ── Readout 1 : Balanced LinearSVC ───────────────────────────────────────
    print("\n=== Readout: Balanced LinearSVC ===")
    clf = train_balanced_svc(F_train, y_train, seed=seed)
    y_pred_svc = clf.predict(F_test)
    acc_svc    = accuracy_score(y_test, y_pred_svc)
    print(f"Test accuracy (Balanced SVC) : {acc_svc:.4f}")
    print(classification_report(y_test, y_pred_svc,
                                 target_names=[str(i) for i in range(5)],
                                 zero_division=0))

    # ── Readout 2 : Prototypical (full training set) ──────────────────────────
    print("\n=== Readout: Prototypical (full support) ===")
    proto_full = PrototypicalReadout().fit(F_train, y_train)
    y_pred_proto = proto_full.predict(F_test)
    acc_proto    = accuracy_score(y_test, y_pred_proto)
    print(f"Test accuracy (Prototypical) : {acc_proto:.4f}")
    print(classification_report(y_test, y_pred_proto,
                                 target_names=[str(i) for i in range(5)],
                                 zero_division=0))

    # ── Readout 3 : Prototypical k-shot ──────────────────────────────────────
    print(f"\n=== Readout: Prototypical ({k_shot}-shot) ===")
    proto_kshot = PrototypicalReadout().fit_k_shot(
        X_train, y_train, snn, k_shot=k_shot, seed=seed
    )
    y_pred_kshot = proto_kshot.predict(F_test)
    acc_kshot    = accuracy_score(y_test, y_pred_kshot)
    print(f"Test accuracy (Prototypical {k_shot}-shot) : {acc_kshot:.4f}")
    print(classification_report(y_test, y_pred_kshot,
                                 target_names=[str(i) for i in range(5)],
                                 zero_division=0))

    return snn, clf, proto_full, proto_kshot


if __name__ == "__main__":
    main()