# -*- coding: utf-8 -*-
import mne
import numpy as np
import pandas as pd
import re
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pywt
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from collections import defaultdict
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from library import ParallelEncoder, DecisionLayer

def apply_bipolar_reference(labeled_signal):
    """Apply a bipolar reference to the signal. The bipolar reference is defined as the difference between two adjacent channels.
    pre : labeled_signal (dict): dictionary of the form {channel_name: signal} where channel_name is a string and signal is a 1D numpy array.
    post: data_out (dict): dictionary of the form {channel_name: signal} where channel_name is a string and signal is a 1D numpy array. 
                           The channel names are of the form "ch1-ch2" where ch1 and ch2 are the names of the two channels used to compute the bipolar reference.
    """
    data_out = {}
    for i in range(len(labeled_signal)-1):
        ch1 = list(labeled_signal.keys())[i]
        ch2 = list(labeled_signal.keys())[i+1]
        data_out[f"{ch1}-{ch2}"] = labeled_signal[ch1] - labeled_signal[ch2]
    return data_out

def apply_car_reference(labeled_signal):
    """" Apply a common average refernce (CAR) to the signal. The CAR is defined as the difference between each channel and the average of all channels.
    pre : labeled_signal (dict): dictionary of the form {channel_name: signal} where channel_name is a string and signal is a 1D numpy array.
    post: data_out (dict): dictionary of the form {channel_name: signal} where channel_name is a string and signal is a 1D numpy array. 
                           The channel names are the same as in the input dictionary.
    """
    data_out = {}
    avg = np.mean(list(labeled_signal.values()), axis=0)
    for ch in labeled_signal.keys():
        data_out[ch] = labeled_signal[ch] - avg
    return data_out

def apply_laplacian_reference(labeled_signal):
    """" Apply a Laplacian reference to the signal. The Laplacian reference is defined as the difference between each channel and the average of its neighbors (contac[i-1] + contact[i+1]).
        Edges are treated as if they had only one neighbor (contact[i-1] for the first channel and contact[i+1] for the last channel).
    pre : labeled_signal (dict): dictionary of the form {channel_name: signal} where channel_name is a string and signal is a 1D numpy array.
    post: data_out (dict): dictionary of the form {channel_name: signal} where channel_name is a string and signal is a 1D numpy array. 
                           The channel names are the same as in the input dictionary.
    """
    data_out = {}
    for i in range(len(labeled_signal)):
        ch = list(labeled_signal.keys())[i]
        if i == 0:
            data_out[ch] = labeled_signal[ch] - labeled_signal[list(labeled_signal.keys())[i+1]]
        elif i == len(labeled_signal)-1:
            data_out[ch] = labeled_signal[ch] - labeled_signal[list(labeled_signal.keys())[i-1]]
        else:
            data_out[ch] = labeled_signal[ch] - (labeled_signal[list(labeled_signal.keys())[i-1]] + labeled_signal[list(labeled_signal.keys())[i+1]])/2
    return data_out




def edf_to_csv(file, reference_type, output_dir, segment_duration, downsample_fs):
    """Extract the iEEG signal from [file] (which is a .edf file) transfers it to a common bipolar montage and saves
    it in a temporary .csv file
    pre : file (pyedflib.EdfReader): open reader of a .edf file.
          reference_type (string): type of reference you want to use. Possibilities are 'none', 'monopolar', 'bipolar', 'laplacian', and 'car'
    post: data (np.ndarray - 2D) : the data extracted from the .edf file and stored in the .csv file.
    """
    # Load the data from the .edf file
    raw = mne.io.read_raw_edf(file, preload=False, encoding="latin1", verbose=True)
    # Keep only EEG channels
    raw.pick([ch for ch in raw.ch_names if ch.startswith("EEG")])
    print("ok pick")
    raw.rename_channels({ch: ch.replace("EEG ", "").strip() for ch in raw.ch_names})
    

    sfreq = raw.info["sfreq"]
    print(f"Original sampling frequency: {sfreq} Hz")
    
    # Downsample 
    if downsample_fs is not None and downsample_fs < sfreq : 
      raw.resample (sfreq = downsample_fs)
      print(f"Signal downsampled to {downsample_fs} Hz")
      sfreq = raw.info["sfreq"]
    elif downsample_fs is not None and downsample_fs >= sfreq:
      print(f"downsample_fs ({downsample_fs} Hz) >= original sfreq ({sfreq} Hz)")  
    
    eeg_channels = raw.ch_names
    signal = raw.get_data()
    labeled_signal = dict(zip(eeg_channels, signal))
    
    # Transform the data to a common bipolar montage from 4 possible unipolar montages
    ref = reference_type.lower()

    if ref in ["none", "monopolar"]:
        data_out = labeled_signal.copy()
    elif ref == "bipolar":
        print("ok test")
        data_out = apply_bipolar_reference(labeled_signal)
        print("ok bipo")
    elif ref == "laplacian":
        data_out = apply_laplacian_reference(labeled_signal)
    elif ref == "car":
        data_out = apply_car_reference(labeled_signal)
    else:
        raise ValueError(
            "reference_type must be one of: "
            "'none', 'monopolar', 'bipolar', 'laplacian', 'car'"
        )
        

    # conversion into DataFrame
    if isinstance(data_out, dict):
        df = pd.DataFrame(data_out)
    else:
        df = pd.DataFrame(data_out.T)
    print("ok conversion into dataframe")
    # segmentation 
    sfreq = raw.info["sfreq"]
    samples_per_segment = int(segment_duration * sfreq)
    n_samples = df.shape[0]
    n_segments = n_samples // samples_per_segment
    segment_paths =  []
    for i in range (n_segments): 
      start = i * samples_per_segment
      stop = start + samples_per_segment
      segment_df = df.iloc[start:stop].reset_index(drop=True)
      out_path = os.path.join(output_dir, f"segment_{i:05d}.csv")
      segment_df.to_csv(out_path, index=False)
      segment_paths.append(out_path)
    print(f"{n_segments} segments of {segment_duration} seconds saved in {output_dir}")
    
    file_name = os.path.basename(file)
    match = re.search(r"(COG_\d+)", file_name)
    patient_id = match.group(1) if match else os.path.splitext(file_name)[0]

    save_path = os.path.join(output_dir, f"{patient_id}_segment_paths.npy")
    np.save(save_path, segment_paths)
    print(f"Segment paths saved in: {save_path}")
    return save_path
    
#data_out = edf_to_csv("/auto/home/users/m/d/mdegand/data/COG_011_BodyBc241123_10min.edf", reference_type = "bipolar", output_dir = "/auto/globalscratch/users/m/d/mdegand/temp_segment_5s_10min/", segment_duration = 5, downsample_fs = 1024)

def scalogram_generation(segments_path, output_dir, wavelet='morl', sfreq=1024, upper_freq=500):
    """Generate scalograms for the segment files in [segments_path] and save them in [output_dir].
    pre : segment_paths (list of strings) : path to the .npy file containing a list of paths to the segment .csv files.
          output_dir (string) : path to the directory where the scalograms will be saved.
          wavelet (string) : name of the wavelet to use for CWT. Default is 'morl'.
    """
    segments = np.load(segments_path, allow_pickle=True).tolist()
    for seg_path in segments:
        df = pd.read_csv(seg_path)
        for ch_name in df.columns:
            signal = df[ch_name].values.astype(float)
            dt = 1 / sfreq
            freqs = np.linspace(1,upper_freq,sfreq)
            scales = pywt.frequency2scale(wavelet, freqs * dt)
            # Compute the scalogram using CWT

            coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
            power = np.abs(coef) ** 2
            # Save the scalogram as a .mat file
            scalogram_path = os.path.join(output_dir, f"{os.path.basename(seg_path).replace('.csv', '')}_{ch_name}_scalogram.mat")
            savemat(scalogram_path, {"scalogram": power})
            print(f"Scalogram for {ch_name} saved in {scalogram_path}")
            print("Scalogram shape:", power.shape)
    print(f"Scalograms saved in {output_dir}")
    return output_dir
#segment_paths = "/auto/home/users/m/d/mdegand/segments_115_119.npy"
#scalo_output_dir = scalogram_generation(segment_paths,"/auto/globalscratch/users/m/d/mdegand/temp_scalogram_5s_10min_115/")

def group_scalogram_by_segment(scalogram_paths):
    grouped = defaultdict(list)

    for path in scalogram_paths:
        filename = os.path.basename(path)

        # segment_00012_Fp1-Fp2_scalogram.mat
        parts = filename.split("_")
        segment_index = int(parts[1])

        grouped[segment_index].append(path)

    # Sort channels within each segment
    grouped_sorted = []
    for seg_idx in sorted(grouped.keys()):
        # Sort by channel name to ensure consistent ordering
        paths_sorted = sorted(grouped[seg_idx])
        grouped_sorted.append(paths_sorted)
    np.save("/auto/home/users/m/d/mdegand/scalo_per_seg_paths2.npy",grouped_sorted)
    return grouped_sorted
import os


#scalogram_dir = "/auto/globalscratch/users/m/d/mdegand/temp_scalogram_5s_10min/"
#scalogram_paths = [
#    os.path.join(scalogram_dir, f)
#    for f in os.listdir(scalogram_dir)
#    if f.endswith(".mat")
#]
#grouped_sorted = group_scalogram_by_segment(scalogram_paths)


class ScalogramDataset(Dataset):
    def __init__(self, segment_scalogram_paths, labels, log_transform=True):
        """
        pre : segment_scalogram_paths (list of strings) : list of paths to the scalogram .mat files.
                                                            [
                                                            [seg0_ch1.mat, seg0_ch2.mat, ..., seg0_chN.mat],
                                                            [seg1_ch1.mat, seg1_ch2.mat, ..., seg1_chN.mat],
                                                            ...
                                                            ]
              labels (list of int) : list of labels corresponding to each scalogram.
              log_transform (bool) : whether to apply log transformation to the scalograms. Default is True.
        """
        self.segment_scalogram_paths = segment_scalogram_paths
        self.labels = labels
        self.log_transform = log_transform

    def __len__(self):
        return len(self.segment_scalogram_paths)
    
    def normalize(self, scalogram):
        """Normalize the scalogram to have zero mean and unit variance.
        pre : scalogram (np.ndarray) : 2D array of shape (F, T) representing the scalogram.
        post: normalized_scalogram (np.ndarray) : 2D array of shape (F, T) representing the normalized scalogram.
        """
        mean = np.mean(scalogram)
        std = np.std(scalogram)
        normalized_scalogram = (scalogram - mean) / (std + 1e-8)  # Add small constant to avoid division by zero
        return normalized_scalogram
    
    def resize(self, array, target_size):
        """"Resize with torch interpolate to avoid cv2 dependency.
        array (np.ndarray) : 2D array of shape (F, T) representing the scalogram.
        target_size (tuple) : tuple of the form (F_target, T_target) representing the target size.
        """
        array_tensor = torch.tensor(array,dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        resized_tensor = nn.functional.interpolate(array_tensor, size=target_size, mode='bilinear', align_corners=False)
        resized_array = resized_tensor.squeeze().numpy()  # Remove batch and channel dimensions
        return resized_array
    
    def visualize_scalogram(self, x, y, cmap="viridis"):
        """Visualize the scalogram using matplotlib.
        pre: x (torch.Tensor) : tensor of shape (1, F, T) representing the scalogram.
            y (int) : label corresponding to the scalogram.
        """
        x = x.squeeze(0).numpy()
        plt.imshow(x, aspect="auto", origin="lower", cmap=cmap)
        plt.title(f"Label = {y.item()}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.show()


    def __getitem__(self, idx, visualize=False):
        segment_paths = self.segment_scalogram_paths[idx]
        label = self.labels[idx]
        segment_scalograms= []
        # Load the scalogram from a .mat file
        for scalogram_path in segment_paths:
            try:
                scalogram = loadmat(scalogram_path)["scalogram"].astype(np.float32) # shape (F, T)
                #print("Scalogram shape :", scalogram.shape)
            except Exception as e:
                raise RuntimeError(f"error loading {scalogram_path}:{e}")
    
            # Apply log transformation if specified
            if self.log_transform:
                scalogram = np.log(scalogram + 1e-8)  # Add small constant to avoid log(0)
            # Normalize the scalogram
            scalogram = self.normalize(scalogram)
            # Resize the scalogram to a fixed size (e.g., 128x256)
            scalogram = self.resize(scalogram, target_size=(80, 80))

            segment_scalograms.append(scalogram)
            #print("Resized scalogram shape :", scalogram.shape)
        # PyTorch expects the input to be in the shape (C, F, T), where C is the number of channels (1 in this case)
        x = np.stack(segment_scalograms, axis=0)  # shape (N_channels, F, T)
        x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor
        y = torch.tensor(label).long()

        if visualize:
            self.visualize_scalogram(x, y)

        return x, y

#grouped_paths = np.load("/auto/home/users/m/d/mdegand/scalo_per_seg_paths_out.npy", allow_pickle=True).tolist()

#labels = [0]*len(grouped_paths)


print("OK scalogram Dataset")

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
        x doit avoir la forme (B, C, H, W)
        Affiche les vrais patchs 16x16 du 1er element du batch.
        """
        if x.ndim != 4:
            raise ValueError(f"x doit avoir la forme (B, C, H, W), recu {x.shape}")

        img = x[0, 0].detach().cpu().numpy()   # (H, W)

        H, W = img.shape
        ph, pw = self.patch_size
        n_h = H // ph
        n_w = W // pw

        fig, axes = plt.subplots(n_h, n_w, figsize=(2 * n_w, 2 * n_h))
        axes = np.array(axes).reshape(n_h, n_w)

        patch_idx = 0
        for i in range(n_h):
            for j in range(n_w):
                patch = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
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
            raise ValueError(f"x doit avoir la forme (B, N_channels, H, W), recu {x.shape}")
            
        if visualize:
          self.visualize_all_patches(x)
          
        B, N_channels, H, W = x.shape

        # on transforme en (B*N_channels, 1, H, W)
        x = x.reshape(B * N_channels, 1, H, W)

        x = self.proj(x)                  # (B*N_channels, embed_dim, H', W')
        x = x.flatten(2)                  # (B*N_channels, embed_dim, num_patches)
        x = x.transpose(1, 2)             # (B*N_channels, num_patches, embed_dim)
        x = x + self.pos_embed

        # on remet les channels ensemble
        x = x.reshape(B, N_channels, self.num_patches, self.embed_dim)
        return x
        
        

def build_tusz_model(channels):
    # Hyperparametres TUSZ
    n_channels = 20
    n_layers = 8
    num_heads = 4
    projection_dim = 40
    transformer_units = [80, 40]
    mlp_head_units = [512, 256]
    n_classes = 1

    # Entre attendue par le modele TUSZ
    inputs = tf.keras.layers.Input(shape=(channels, 25, 40), name="input_tusz")

    # Encodeur parallele
    enc = ParallelEncoder(
        n_channels=n_channels,
        n_layers=n_layers,
        num_heads=num_heads,
        projection_dim=projection_dim,
        transformer_units=transformer_units
    )(inputs)

    # Tete de deision
    logits = DecisionLayer(
        mlp_units=mlp_head_units,
        n_classes=n_classes
    )(enc)

    # Modee final
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="TUSZ_model")

    return model
