# -*- coding: utf-8 -*-
import mne
import numpy as np
import pandas as pd
import re
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pywt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from library import ParallelEncoder, DecisionLayer

def apply_bipolar_reference(labeled_signal):
    """Apply a bipolar reference to the signal.
    The bipolar reference is defined as the difference between two adjacent channels.

    pre : labeled_signal (dict): dictionary of the form {channel_name: signal}
                                 where channel_name is a string and signal is a 1D numpy array.
    post: data_out (dict): dictionary of the form {channel_name: signal}
                           where channel_name is a string and signal is a 1D numpy array.
                           The channel names are of the form "ch1-ch2".
    """
    data_out = {}
    for i in range(len(labeled_signal) - 1):
        ch1 = list(labeled_signal.keys())[i]
        ch2 = list(labeled_signal.keys())[i + 1]
        data_out[f"{ch1}-{ch2}"] = labeled_signal[ch1] - labeled_signal[ch2]
    return data_out

def apply_car_reference(labeled_signal):
    """Apply a common average reference (CAR) to the signal.
    The CAR is defined as the difference between each channel and the average of all channels.

    pre : labeled_signal (dict): dictionary of the form {channel_name: signal}
                                 where channel_name is a string and signal is a 1D numpy array.
    post: data_out (dict): dictionary of the form {channel_name: signal}
                           where channel_name is a string and signal is a 1D numpy array.
                           The channel names are the same as in the input dictionary.
    """
    data_out = {}
    avg = np.mean(list(labeled_signal.values()), axis=0)
    for ch in labeled_signal.keys():
        data_out[ch] = labeled_signal[ch] - avg
    return data_out

def apply_laplacian_reference(labeled_signal):
    """Apply a Laplacian reference to the signal.
    The Laplacian reference is defined as the difference between each channel and the average of its neighbors.
    Edges are treated as if they had only one neighbor.

    pre : labeled_signal (dict): dictionary of the form {channel_name: signal}
                                 where channel_name is a string and signal is a 1D numpy array.
    post: data_out (dict): dictionary of the form {channel_name: signal}
                           where channel_name is a string and signal is a 1D numpy array.
                           The channel names are the same as in the input dictionary.
    """
    data_out = {}
    for i in range(len(labeled_signal)):
        ch = list(labeled_signal.keys())[i]
        if i == 0:
            data_out[ch] = labeled_signal[ch] - labeled_signal[list(labeled_signal.keys())[i + 1]]
        elif i == len(labeled_signal) - 1:
            data_out[ch] = labeled_signal[ch] - labeled_signal[list(labeled_signal.keys())[i - 1]]
        else:
            data_out[ch] = labeled_signal[ch] - (
                labeled_signal[list(labeled_signal.keys())[i - 1]] +
                labeled_signal[list(labeled_signal.keys())[i + 1]]
            ) / 2
    return data_out


def edf_to_csv(file, reference_type, output_dir, segment_duration, downsample_fs):
    """Extract the iEEG signal from [file], apply the selected reference,
    segment it, and return the segments directly in memory.
    """
    # Load data from the .edf file
    raw = mne.io.read_raw_edf(file, preload=False, encoding="latin1", verbose=True)

    # Keep only EEG channels
    raw.pick([ch for ch in raw.ch_names if ch.startswith("EEG")])
    print("OK pick")
    raw.rename_channels({ch: ch.replace("EEG ", "").strip() for ch in raw.ch_names})

    sfreq = raw.info["sfreq"]
    print(f"Original sampling frequency: {sfreq} Hz")

    # Downsample
    if downsample_fs is not None and downsample_fs < sfreq:
        raw.resample(sfreq=downsample_fs)
        print(f"Signal downsampled to {downsample_fs} Hz")
        sfreq = raw.info["sfreq"]
    elif downsample_fs is not None and downsample_fs >= sfreq:
        print(f"downsample_fs ({downsample_fs} Hz) >= original sfreq ({sfreq} Hz)")

    eeg_channels = raw.ch_names
    signal = raw.get_data()
    labeled_signal = dict(zip(eeg_channels, signal))

    # Apply the selected reference
    ref = reference_type.lower()

    if ref in ["none", "monopolar"]:
        data_out = labeled_signal.copy()
    elif ref == "bipolar":
        print("OK test")
        data_out = apply_bipolar_reference(labeled_signal)
        print("OK bipolar")
    elif ref == "laplacian":
        data_out = apply_laplacian_reference(labeled_signal)
    elif ref == "car":
        data_out = apply_car_reference(labeled_signal)
    else:
        raise ValueError(
            "reference_type must be one of: "
            "'none', 'monopolar', 'bipolar', 'laplacian', 'car'"
        )

    # Convert to DataFrame
    if isinstance(data_out, dict):
        df = pd.DataFrame(data_out)
    else:
        df = pd.DataFrame(data_out.T)
    print("OK conversion into dataframe")

    # Segmentation
    sfreq = raw.info["sfreq"]
    samples_per_segment = int(segment_duration * sfreq)
    n_samples = df.shape[0]
    n_segments = n_samples // samples_per_segment

    segments = []
    for i in range(n_segments):
        start = i * samples_per_segment
        stop = start + samples_per_segment
        segment_df = df.iloc[start:stop].reset_index(drop=True)
        segments.append(segment_df)

    print(f"{n_segments} segments of {segment_duration} seconds created in memory")

    file_name = os.path.basename(file)
    match = re.search(r"(COG_\d+)", file_name)
    if match:
        patient_id = match.group(1)
    else:
        patient_id = "unknown_patient"

    return segments, patient_id


def scalogram_generation(segments, wavelet='morl', sfreq=1024, upper_freq=500):
    """Generate scalograms for all segments directly in memory.

    Returns:
        all_segments_scalograms = [
            [seg0_ch1_scalo, seg0_ch2_scalo, ..., seg0_chN_scalo],
            [seg1_ch1_scalo, seg1_ch2_scalo, ..., seg1_chN_scalo],
            ...
        ]
    """
    all_segments_scalograms = []

    for i, df in enumerate(segments):
        segment_scalograms = []
        for ch_name in df.columns:
            signal = df[ch_name].values.astype(float)
            dt = 1 / sfreq
            freqs = np.linspace(1, upper_freq, sfreq)
            scales = pywt.frequency2scale(wavelet, freqs * dt)

            coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
            power = np.abs(coef) ** 2
            segment_scalograms.append(power.astype(np.float32))

        all_segments_scalograms.append(segment_scalograms)
        print(f"Scalograms generated in memory for segment {i}")

    print("All scalograms generated in memory")
    return all_segments_scalograms


class ScalogramDataset(Dataset):
    def __init__(self, segment_scalograms, labels, log_transform=True):
        """
        pre : segment_scalograms : list of list of 2D np.ndarray
              [
                [seg0_ch1_scalo, seg0_ch2_scalo, ..., seg0_chN_scalo],
                [seg1_ch1_scalo, seg1_ch2_scalo, ..., seg1_chN_scalo],
                ...
              ]
              labels (list of int) : list of labels corresponding to each segment.
              log_transform (bool) : whether to apply log transformation to the scalograms.
        """
        self.segment_scalograms = segment_scalograms
        self.labels = labels
        self.log_transform = log_transform

    def __len__(self):
        return len(self.segment_scalograms)

    def normalize(self, scalogram):
        """Normalize the scalogram to have zero mean and unit variance."""
        mean = np.mean(scalogram)
        std = np.std(scalogram)
        normalized_scalogram = (scalogram - mean) / (std + 1e-8)
        return normalized_scalogram

    def resize(self, array, target_size):
        """Resize using torch interpolate to avoid cv2 dependency."""
        array_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_tensor = nn.functional.interpolate(
            array_tensor, size=target_size, mode='bilinear', align_corners=False
        )
        resized_array = resized_tensor.squeeze().numpy()
        return resized_array

    def visualize_scalogram(self, x, y, cmap="viridis"):
        """Visualize the scalogram using matplotlib."""
        x = x.squeeze(0).numpy()
        plt.imshow(x, aspect="auto", origin="lower", cmap=cmap)
        plt.title(f"Label = {y.item()}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.show()

    def __getitem__(self, idx, visualize=False):
        segment_scalograms_raw = self.segment_scalograms[idx]
        label = self.labels[idx]
        segment_scalograms = []

        for scalogram in segment_scalograms_raw:
            scalogram = scalogram.astype(np.float32)

            if self.log_transform:
                scalogram = np.log(scalogram + 1e-8)

            scalogram = self.normalize(scalogram)
            scalogram = self.resize(scalogram, target_size=(80, 80))
            segment_scalograms.append(scalogram)

        x = np.stack(segment_scalograms, axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label).long()

        if visualize:
            self.visualize_scalogram(x, y)

        return x, y


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(80, 80), patch_size=(16, 16), embed_dim=40):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        img_h, img_w = img_size
        patch_h, patch_w = patch_size

        if img_h % patch_h != 0 or img_w % patch_w != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")

        self.num_patches = (img_h // patch_h) * (img_w // patch_w)

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def visualize_all_patches(self, x, cmap="viridis"):
        """
        x must have shape (B, C, H, W)
        Display the real 16x16 patches of the first element in the batch.
        """
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B, C, H, W), got {x.shape}")

        img = x[0, 0].detach().cpu().numpy()

        H, W = img.shape
        ph, pw = self.patch_size
        n_h = H // ph
        n_w = W // pw

        fig, axes = plt.subplots(n_h, n_w, figsize=(2 * n_w, 2 * n_h))
        axes = np.array(axes).reshape(n_h, n_w)

        patch_idx = 0
        for i in range(n_h):
            for j in range(n_w):
                patch = img[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                ax = axes[i, j]
                ax.imshow(patch, aspect="auto", origin="lower", cmap=cmap)
                ax.set_title(f"{patch_idx}", fontsize=8)
                ax.axis("off")
                patch_idx += 1

        plt.suptitle("Real 16x16 patches of the scalogram", fontsize=14)
        plt.tight_layout()
        plt.show()

    def forward(self, x, visualize=False):
        # x : (B, N_channels, H, W)
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B, N_channels, H, W), got {x.shape}")

        if visualize:
            self.visualize_all_patches(x)

        B, N_channels, H, W = x.shape

        # reshape to (B*N_channels, 1, H, W)
        x = x.reshape(B * N_channels, 1, H, W)

        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x + self.pos_embed

        # reshape back to regroup channels
        x = x.reshape(B, N_channels, self.num_patches, self.embed_dim)
        return x


def build_tusz_model(channels):
    # TUSZ hyperparameters
    n_channels = 20
    n_layers = 8
    num_heads = 4
    projection_dim = 40
    transformer_units = [80, 40]
    mlp_head_units = [512, 256]
    n_classes = 1

    # Model input
    inputs = tf.keras.layers.Input(shape=(channels, 25, 40), name="input_tusz")

    # Parallel encoder
    enc = ParallelEncoder(
        n_channels=n_channels,
        n_layers=n_layers,
        num_heads=num_heads,
        projection_dim=projection_dim,
        transformer_units=transformer_units
    )(inputs)

    # Decision head
    logits = DecisionLayer(
        mlp_units=mlp_head_units,
        n_classes=n_classes
    )(enc)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="TUSZ_model")
    return model


# =========================================================
# FULL PIPELINE WITHOUT INTERMEDIATE SAVES
# =========================================================

edf_file = "/auto/home/users/m/d/mdegand/data/COG_011_BodyBc241123_5min.edf"

segments, patient_id = edf_to_csv(
    edf_file,
    reference_type="bipolar",
    output_dir="",
    segment_duration=5,
    downsample_fs=1024
)

all_segment_scalograms = scalogram_generation(
    segments,
    wavelet='morl',
    sfreq=1024,
    upper_freq=500
)

labels = [0] * len(all_segment_scalograms)

dataset = ScalogramDataset(all_segment_scalograms, labels)
print("Number of segments:", len(dataset))
print("OK scalogram dataset")

batch_size = 8
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
patch = PatchEmbedding(embed_dim=40)
X_list = []
y_list = []
print("OK data loader")

for x_batch, y_batch in loader:
    patch_embeddings_torch = patch(x_batch)
    patch_embeddings_tf = tf.convert_to_tensor(
        patch_embeddings_torch.detach().cpu().numpy(),
        dtype=tf.float32
    )

    X_list.append(patch_embeddings_tf)
    y_list.append(y_batch.detach().cpu().numpy())

X_all = np.concatenate(X_list, axis=0).astype(np.float32)
y_all = np.concatenate(y_list, axis=0)

print("\nFinal batch shape:", X_all.shape)
print("Label shape:", y_all.shape)

print("OK patch")

model = build_tusz_model(channels=122)
model.summary()
print("OK built model")

dummy_input = tf.zeros((1,) + X_all.shape[1:], dtype=tf.float32)
_ = model(dummy_input)

try:
    model.load_weights("/auto/home/users/m/d/mdegand/data/checkpoint_fine_tuning_no_pt")
    print("Weights loaded successfully.")
except Exception as e:
    print("Error while loading weights:", e)

logits = model.predict(X_all, verbose=0)
proba = tf.sigmoid(logits).numpy()
pred_class = (proba > 0.5).astype(int)

df_pred = pd.DataFrame({
    "segment_id": np.arange(len(X_all)),
    "true_label": y_all,
    "logit": logits.squeeze(),
    "probability": proba.squeeze(),
    "predicted_class": pred_class.squeeze()
})

final_csv_path = f"/auto/home/users/m/d/mdegand/{patient_id}_predictions_segments.csv"
df_pred.to_csv(final_csv_path, index=False)
print(f"CSV saved: {final_csv_path}")
print("Done")