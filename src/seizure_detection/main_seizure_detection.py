from library_SEEG import *
import time
import os
import argparse
import numpy as np
import pandas as pd
import shutil
from pathlib import Path


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

    # Resolve weights path: explicit --weights > default location in repo
    if args.weights:
        weights_path = args.weights
    else:
        # Default: Data/transformer_model/ relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root  = os.path.dirname(script_dir)  # one level up from seizure_detection/
        weights_path = os.path.join(
            repo_root, "Data", "transformer_model", "checkpoint_fine_tuning_no_pt"
        )

    if os.path.exists(weights_path + ".index"):
        try:
            model.load_weights(weights_path)
            print(f"Weights loaded from: {weights_path}")
        except Exception as e:
            print(f"Error loading weights from {weights_path}: {e}")
    else:
        print(f"Warning: weights not found at {weights_path} — running with untrained model.")

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
