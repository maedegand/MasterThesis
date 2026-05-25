from library_SEEG import *
import time
import os
import argparse
import numpy as np
import pandas as pd
import shutil
from pathlib import Path


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "iEEG scalogram preprocessing and seizure prediction pipeline.\n\n"
            "Usage:\n"
            "  python main.py -i iEEG.edf -o output_dir/\n\n"
            "With custom temp dirs:\n"
            "  python main.py -i iEEG.edf -o output_dir/ \\\n"
            "      --segment-dir /scratch/segs/ --scalogram-dir /scratch/scalo/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        metavar="EDF_PATH",
        required=True,
        help="iEEG signal file (.edf).",
    )

    parser.add_argument(
        "-o", "--output",
        metavar="OUTPUT_DIR",
        required=True,
        help="Output directory for all results (X_all.npy, predictions.csv, model/, ...).",
    )

    # ── Optional temp dirs (default: inside output_dir) ───────────────────────
    parser.add_argument(
        "--segment-dir",
        metavar="DIR",
        default=None,
        help="Where to store temp segments. Default: <OUTPUT_DIR>/segments/",
    )
    parser.add_argument(
        "--scalogram-dir",
        metavar="DIR",
        default=None,
        help="Where to store temp scalograms. Default: <OUTPUT_DIR>/scalograms/",
    )

    # ── Optional model/pipeline params ───────────────────────────────────────
    parser.add_argument(
        "--weights",
        metavar="WEIGHTS_PATH",
        default=None,
        help="Path to pretrained model weights (optional).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=122,
        help="Number of iEEG channels (default: 122).",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=5,
        help="Segment duration in seconds (default: 5).",
    )
    parser.add_argument(
        "--downsample-fs",
        type=int,
        default=1024,
        help="Downsampling frequency in Hz (default: 1024).",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def reset_folder(folder_path):
    """Clear folder contents without deleting the folder itself.
    Avoids PermissionError on Windows (OneDrive, locked handles, etc.)."""
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Warning: could not delete {item_path}: {e}")
    else:
        os.makedirs(folder_path, exist_ok=True)


# ============================================================
# Main pipeline
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    edf_path   = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Temp dirs: use custom if provided, otherwise nest inside output_dir
    segment_dir   = args.segment_dir   or os.path.join(output_dir, "segments")
    scalogram_dir = args.scalogram_dir or os.path.join(output_dir, "scalograms")

    # Derived output paths
    x_all_path      = os.path.join(output_dir, "X_all.npy")
    y_all_path      = os.path.join(output_dir, "y_all.npy")
    pred_path       = os.path.join(output_dir, "predictions.csv")
    step_count_path = os.path.join(output_dir, "preprocessing_step_counts.csv")
    model_save_path = os.path.join(output_dir, "model")

    print("=" * 60)
    print("Pipeline inputs")
    print("=" * 60)
    print(f"  EDF file      : {edf_path}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Segment dir   : {segment_dir}")
    print(f"  Scalogram dir : {scalogram_dir}")
    if args.weights:
        print(f"  Weights       : {args.weights}")
    print("=" * 60)
    print()

    reset_folder(segment_dir)
    reset_folder(scalogram_dir)

    # ── Step counter ──────────────────────────────────────────────────────────
    step_counts = []

    def add_step_count(step_name, count):
        step_counts.append({"step": step_name, "n_windows": int(count)})
        print(f"[STEP COUNT] {step_name}: {count}")

    start_time = time.perf_counter()

    # ============================================================
    # 1. Expected number of windows from EDF duration
    # ============================================================
    try:
        raw_info = mne.io.read_raw_edf(edf_path, preload=False, encoding="latin1", verbose=False)
        duration_sec = raw_info.n_times / raw_info.info["sfreq"]
        n_expected_windows = int(duration_sec // args.segment_duration)

        add_step_count(f"Raw EDF - expected {args.segment_duration}s windows", n_expected_windows)

        print(f"EDF duration: {duration_sec:.2f} s")
        print(f"Sampling frequency: {raw_info.info['sfreq']} Hz")
        print(f"Expected number of {args.segment_duration}s windows: {n_expected_windows}")

    except Exception as e:
        print("Could not read EDF duration:", e)
        add_step_count(f"Raw EDF - expected {args.segment_duration}s windows", 0)

    # ============================================================
    # 2. iEEG signal preprocessing and segmentation
    # ============================================================
    segments_list_npy = edf_to_csv(
        edf_path,
        reference_type="bipolar",
        output_dir=segment_dir,
        segment_duration=args.segment_duration,
        downsample_fs=args.downsample_fs
    )
    segment_paths = np.load(segments_list_npy, allow_pickle=True).tolist()

    add_step_count("Valid segments", len(segment_paths))

    print("Segments returned:", segments_list_npy)
    print("Files in segment folder:", os.listdir(segment_dir)[:10])

    # ============================================================
    # 3. Scalogram generation
    # ============================================================
    scalogram_mat = scalogram_generation(segments_list_npy, scalogram_dir)

    scalogram_paths = [
        os.path.join(scalogram_mat, f)
        for f in os.listdir(scalogram_mat)
        if f.endswith(".mat")
    ]

    add_step_count("Scalogram files generated", len(scalogram_paths))

    scalogram_sorted = group_scalogram_by_segment(scalogram_paths)

    add_step_count("Segments with grouped scalograms", len(scalogram_sorted))

    print("Number of grouped scalogram segments:", len(scalogram_sorted))

    # ============================================================
    # 4. Transformation into PyTorch dataset
    # ============================================================
    labels = [0] * len(scalogram_sorted)
    dataset = ScalogramDataset(scalogram_sorted, labels)

    add_step_count("Dataset samples", len(dataset))

    # ============================================================
    # 5. Patch embedding and positional encoding
    # ============================================================
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    patch = PatchEmbedding(embed_dim=40)

    X_list = []
    y_list = []

    n_tokenized_windows = 0
    token_shape_example = None

    for x_batch, y_batch in loader:
        print("Input batch shape:", x_batch.shape)

        patch_embeddings_torch = patch(x_batch)
        print("Torch patch embeddings shape:", patch_embeddings_torch.shape)

        n_tokenized_windows += patch_embeddings_torch.shape[0]

        if token_shape_example is None:
            token_shape_example = tuple(patch_embeddings_torch.shape[1:])

        patch_embeddings_tf = tf.convert_to_tensor(
            patch_embeddings_torch.detach().cpu().numpy(),
            dtype=tf.float32
        )

        print("TensorFlow patch embeddings shape:", patch_embeddings_tf.shape)

        X_list.append(patch_embeddings_tf)
        y_list.append(y_batch.detach().cpu().numpy())

    add_step_count("Windows converted to tokens", n_tokenized_windows)

    if token_shape_example is not None:
        print("Example token shape per window:", token_shape_example)

    # ============================================================
    # 6. Concatenate final model-ready inputs
    # ============================================================
    if len(X_list) == 0:
        raise ValueError(
            "No tokenized windows were generated. "
            "Check segmentation, scalogram generation, and dataset creation."
        )

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0)

    add_step_count("Model-ready inputs", X_all.shape[0])

    np.save(x_all_path, X_all)
    np.save(y_all_path, y_all)

    print("\nBatch final shape:", X_all.shape)
    print("Label shape:", y_all.shape)

    # ============================================================
    # 7. Save preprocessing step counts
    # ============================================================
    df_steps = pd.DataFrame(step_counts)
    df_steps.to_csv(step_count_path, index=False)

    print("\nPreprocessing step counts:")
    print(df_steps)
    print("Step counts CSV saved:", step_count_path)

    # ============================================================
    # 8. Model loading and prediction
    # ============================================================
    model = build_tusz_model(channels=args.channels)
    model.summary()
    model.save(model_save_path)
    print("Model saved to:", model_save_path)

    dummy_input = tf.zeros((1,) + X_all.shape[1:], dtype=tf.float32)
    _ = model(dummy_input)

    if args.weights:
        try:
            model.load_weights(args.weights)
            print("Weights loaded successfully.")
        except Exception as e:
            print("Error loading weights:", e)
    else:
        print("No weights provided — running with untrained model.")

    logits = model.predict(X_all, verbose=0)
    proba = tf.sigmoid(logits).numpy()
    pred_class = (proba > 0.5).astype(int)

    df_pred = pd.DataFrame({
        "segment_id":      np.arange(len(X_all)),
        "true_label":      y_all,
        "logit":           logits.squeeze(),
        "probability":     proba.squeeze(),
        "predicted_class": pred_class.squeeze()
    })

    df_pred.to_csv(pred_path, index=False)
    print("Predictions saved:", pred_path)
    print("Fini")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} s  ({elapsed/60:.2f} min)")
