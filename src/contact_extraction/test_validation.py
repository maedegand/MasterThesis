# -*- coding: utf-8 -*-

import nibabel as nib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import atlas as atlas_prep
import labeling as label
from nilearn.image import resample_to_img
import matplotlib.patches as mpatches


# ============================================================
# Paths
# ============================================================

CT_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\CT\51138313_Volume_TF_EB2_20231121142028_302_strip.nii.gz"
mask_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\mask_generated\51138313_Volume_TF_EB2_20231121142028_302_strip.nii.gz_electrode_mask.nii.gz"
#contact_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\contacts_reordered.csv"
contact_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\contacts_shifted_corrected.csv"
entry_points_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\input_ElectroLoc\input_ElectroLoc_voxel.csv"
entry_points_path_world = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\input_ElectroLoc\input_ElectroLoc_world.csv"
#contact_path =  r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\output_ElectroLoc\ouput_manual_test.csv"
contact_col = "c_id"      # change to "contact_id" if needed
electrode_col = "electrode_id"


# ============================================================
# Validation Contact Localization 
# ============================================================

def validate_contacts_with_electrode_mask(mask_path, contact_path, output_path):
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()

    df = pd.read_csv(contact_path)

    mask_bin = (mask > 0).astype(np.uint8)
    voxel_sizes = mask_img.header.get_zooms()[:3]

    dist_map_mm = distance_transform_edt(
        1 - mask_bin,
        sampling=voxel_sizes
    )

    inside_list = []
    dist_list = []

    for _, row in df.iterrows():
        x = int(round(row["vox_x"]))
        y = int(round(row["vox_y"]))
        z = int(round(row["vox_z"]))

        if 0 <= x < mask_bin.shape[0] and 0 <= y < mask_bin.shape[1] and 0 <= z < mask_bin.shape[2]:
            inside = mask_bin[x, y, z] == 1
            dist = dist_map_mm[x, y, z]
        else:
            inside = False
            dist = np.nan

        inside_list.append(inside)
        dist_list.append(dist)

    df["inside_mask"] = inside_list
    df["distance_to_mask_mm"] = dist_list

    df.to_csv(output_path, index=False)
    print(f"Saved contact validation to: {output_path}")

    return output_path, df

def validation_contact_loc_with_ground_truth(
    ground_truth_path,
    tested_output_path,
    validation_output_path,
    threshold=2.0,
    electrode_col="electrode_id"
):

    # =========================================================
    # LOAD CSV
    # =========================================================
    ground_truth = pd.read_csv(ground_truth_path)
    tested_output = pd.read_csv(tested_output_path)

    results = []

    TP = 0
    FP = 0
    FN = 0
    No_GT = 0

    # =========================================================
    # LOOP OVER ELECTRODES
    # =========================================================
    ground_truth[electrode_col] = ground_truth[electrode_col].astype(str).str.strip().str.upper()
    tested_output["electrode_id"] = tested_output["electrode_id"].astype(str).str.strip().str.upper()
    electrodes = ground_truth[electrode_col].unique()
    
    for electrode in electrodes:

        gt_elec = ground_truth[
            ground_truth[electrode_col] == electrode
        ]

        pred_elec = tested_output[
            tested_output["electrode_id"] == electrode
        ]

        # Tous les c_id de cette électrode
        c_ids = gt_elec["c_id"].unique()

        # =====================================================
        # LOOP OVER CONTACTS
        # =====================================================
        for c_id in c_ids:

            gt_contact = gt_elec[
                gt_elec["c_id"] == c_id
            ]

            pred_contact = pred_elec[
                pred_elec["c_id"] == c_id
            ]

            # -------------------------------------------------
            # Contact absent dans la prédiction
            # -------------------------------------------------
            if len(pred_contact) == 0:

                results.append({
                    "electrode_id": electrode,
                    "c_id": c_id,
                    "distance_mm": np.nan,
                    "status": "FN"
                })

                FN += 1
                continue

            # -------------------------------------------------
            # Coordonnées
            # -------------------------------------------------
            gt_coords = gt_contact[
                ["world_x", "world_y", "world_z"]
            ].values[0]

            pred_coords = pred_contact[
                ["world_x", "world_y", "world_z"]
            ].values[0]

            # -------------------------------------------------
            # Distance euclidienne
            # -------------------------------------------------
            distance = np.linalg.norm(pred_coords - gt_coords)

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------
            if distance < threshold:

                status = "TP"
                TP += 1
            elif np.isnan(distance):

                status = "No GT available"
                No_GT +=1
            else:

                status = "FP"
                FP += 1

            # -------------------------------------------------
            # Save result
            # -------------------------------------------------
            results.append({
                "electrode_id": electrode,
                "c_id": c_id,
                "distance_mm": distance,
                "status": status
            })

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    df_results = pd.DataFrame(results)

    df_results.to_csv(validation_output_path, index=False)

    # =========================================================
    # METRICS
    # =========================================================
    total = TP + FP + FN

    accuracy = TP / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    distances = df_results["distance_mm"].dropna()

    # =========================================================
    # PRINT
    # =========================================================
    print(f"Validation results saved to: {validation_output_path}")

    print("\n===== Classification metrics =====")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"No GT available: {No_GT}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    print("\n===== Distance metrics =====")

    if len(distances) > 0:
        print(f"Mean distance   : {distances.mean():.2f} mm")
        print(f"Median distance : {distances.median():.2f} mm")
        print(f"Max distance    : {distances.max():.2f} mm")

        success_rate = 100 * (distances < threshold).sum() / len(distances)

        print(f"Success rate @{threshold} mm : {success_rate:.1f}%")

    return df_results, distances.median()



def compute_mean_intercontact_distance(ground_truth_path):
    """
    Calcule la distance moyenne entre contacts consécutifs
    pour chaque électrode, à partir des coordonnées world_x, world_y, world_z.

    Returns
    -------
    mean_distance : float
        Distance moyenne inter-contact globale.
    distances_df : pd.DataFrame
        Tableau contenant toutes les distances inter-contact calculées.
    """

    df = pd.read_csv(ground_truth_path)

    # Nettoyage des lignes sans coordonnées
    df = df.dropna(subset=["world_x", "world_y", "world_z"])

    # Harmoniser les types
    df["electrode_id"] = df["electrode_id"].astype(str)
    df["c_id"] = df["c_id"].astype(int)

    all_distances = []

    # Calcul électrode par électrode
    for electrode_id, group in df.groupby("electrode_id"):

        # Trier les contacts selon leur index
        group = group.sort_values("c_id").reset_index(drop=True)

        coords = group[["world_x", "world_y", "world_z"]].values
        c_ids = group["c_id"].values

        # Distance entre contacts consécutifs
        for i in range(len(coords) - 1):

            dist = np.linalg.norm(coords[i + 1] - coords[i])

            all_distances.append({
                "electrode_id": electrode_id,
                "c_id_1": c_ids[i],
                "c_id_2": c_ids[i + 1],
                "distance_mm": dist
            })

    distances_df = pd.DataFrame(all_distances)

    mean_distance = distances_df["distance_mm"].mean()

    print(f"Mean inter-contact distance: {mean_distance:.3f} mm")

    return mean_distance, distances_df


# ============================================================
# Validation entry points
# ============================================================

def validate_entry_points(contact_path, entry_points_path, output_path, mask_path=None):

    contacts = pd.read_csv(contact_path)
    entry_points = pd.read_csv(entry_points_path)

    if mask_path is not None:
        mask_img = nib.load(mask_path)
        voxel_sizes = np.array(mask_img.header.get_zooms()[:3])
    else:
        voxel_sizes = np.array([1, 1, 1])

    results = []

    electrode_ids = contacts["electrode_id"].drop_duplicates().tolist()

    for i, elec_id in enumerate(electrode_ids):

        group = contacts[contacts["electrode_id"] == elec_id]

        superficial_contact = group.loc[group["c_id"].idxmax()]

        contact_world = np.array([
            superficial_contact["world_x"],
            superficial_contact["world_y"],
            superficial_contact["world_z"]
        ], dtype=float)

        entry_row = entry_points.iloc[i]

        entry_world = np.array([
            entry_row["world_x"],
            entry_row["world_y"],
            entry_row["world_z"]
        ], dtype=float)

        dist_mm = np.linalg.norm(contact_world - entry_world)

        results.append({
            "electrode_id": elec_id,
            "c_id_superficial": superficial_contact["c_id"],
            "contact_world_x": contact_world[0],
            "contact_world_y": contact_world[1],
            "contact_world_z": contact_world[2],
            "entry_world_x": entry_world[0],
            "entry_world_y": entry_world[1],
            "entry_world_z": entry_world[2],
            "distance_mm": dist_mm
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"Saved entry point validation to: {output_path}")

    return output_path




def validate_entry_points_with_ground_truth(
    ground_truth_path,
    entry_points_path,
    output_path,
    threshold=1,
    coord_cols=("world_x", "world_y", "world_z")
):
    """
    Validate extracted entry points by comparing electrodes with the same electrode_id.

    For each extracted electrode entry point:
    - find the matching ground truth electrode_id
    - compute the Euclidean distance between coordinates
    - consider the entry point correct if distance <= threshold

    Args:
        ground_truth_path (str): Path to the ground truth CSV file.
        entry_points_path (str): Path to the extracted entry points CSV file.
        output_path (str): Path where the validation CSV will be saved.
        threshold (float): Distance threshold in mm.
        coord_cols (tuple): Coordinate columns used for distance computation.

    Returns:
        pd.DataFrame: Validation results.
    """

    ground_truth = pd.read_csv(ground_truth_path, sep=",", encoding="utf-8-sig", decimal='.')
    entry_points = pd.read_csv(entry_points_path, sep=",")

    # =========================
    # Check required columns
    # =========================
    required_cols = list(coord_cols) + ["electrode_id"]

    for col in required_cols:
        if col not in ground_truth.columns:
            raise ValueError(f"Column '{col}' not found in ground truth file.")

        if col not in entry_points.columns:
            raise ValueError(f"Column '{col}' not found in entry points file.")

    # Clean electrode IDs
    ground_truth["electrode_id"] = (
        ground_truth["electrode_id"].astype(str).str.strip()
    )

    entry_points["electrode_id"] = (
        entry_points["electrode_id"].astype(str).str.strip()
    )

    results = []

    # =========================
    # Compare same electrode_id
    # =========================
    for i, row in entry_points.iterrows():

        electrode_id = row["electrode_id"]

        gt_match = ground_truth[
            ground_truth["electrode_id"] == electrode_id
        ]

        # No matching electrode in GT
        if gt_match.empty:

            result = {
                "entry_point_index": i,
                "electrode_id": electrode_id,
                "distance_mm": np.nan,
                "correct": False,
                "threshold_mm": threshold,
                "match_found": False,
            }

            for col in coord_cols:
                result[f"extracted_{col}"] = row[col]

            results.append(result)
            continue

        # Take first matching row
        gt_row = gt_match.iloc[0]

        ep_coord = row[list(coord_cols)].to_numpy(dtype=float)
        gt_coord = gt_row[list(coord_cols)].to_numpy(dtype=float)

        # Euclidean distance
        distance = np.linalg.norm(ep_coord - gt_coord)

        is_correct = distance <= threshold

        result = {
            "entry_point_index": i,
            "electrode_id": electrode_id,
            "distance_mm": distance,
            "correct": is_correct,
            "threshold_mm": threshold,
            "match_found": True,
        }

        # Extracted coordinates
        for col in coord_cols:
            result[f"extracted_{col}"] = row[col]

        # Ground truth coordinates
        for col in coord_cols:
            result[f"ground_truth_{col}"] = gt_row[col]

        results.append(result)

    results_df = pd.DataFrame(results)

    # =========================
    # Global metrics
    # =========================
    valid_results = results_df[results_df["match_found"] == True]

    accuracy = valid_results["correct"].mean()
    mean_distance = valid_results["distance_mm"].mean()
    median_distance = valid_results["distance_mm"].median()

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Mean distance: {mean_distance:.3f} mm")
    print(f"Median distance: {median_distance:.3f} mm")

    results_df.to_csv(output_path, index=False)

    return results_df

# ============================================================
# Plot validation barplot (inside vs outside <1mm vs outside >=1mm)
# ============================================================

def plot_validation(df):
    df_valid = df.dropna(subset=["distance_to_mask_mm"])

    inside = df_valid["inside_mask"].sum()

    outside_close = df_valid[
        (df_valid["inside_mask"] == False) &
        (df_valid["distance_to_mask_mm"] < 1)
    ].shape[0]

    outside_far = df_valid[
        (df_valid["inside_mask"] == False) &
        (df_valid["distance_to_mask_mm"] >= 1)
    ].shape[0]

    labels = ["Inside mask", "Outside < 1 mm", "Outside ≥ 1 mm"]
    values = [inside, outside_close, outside_far]
    colors = ["green", "orange", "red"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            int(bar.get_height()),
            ha="center",
            va="bottom"
        )

    plt.title("Electrode Localization Validation")
    plt.ylabel("Number of contacts")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_3d(ct, mask, df, step_ct=6, step_mask=1, show_ct=True, show_labels=False):
    fig = go.Figure()

    # ============================================================
    # CT volume, très transparent et filtré
    # ============================================================
    if show_ct:
        ct_small = ct[::step_ct, ::step_ct, ::step_ct]

        X, Y, Z = np.mgrid[
            0:ct.shape[0]:step_ct,
            0:ct.shape[1]:step_ct,
            0:ct.shape[2]:step_ct
        ]

        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=ct_small.flatten(),

            # Important : ne pas afficher tout le fond CT
            isomin=np.percentile(ct_small, 70),
            isomax=np.percentile(ct_small, 99),

            opacity=0.03,
            surface_count=3,
            colorscale="Greys",
            showscale=False,
            name="CT"
        ))

    # ============================================================
    # Electrode mask : pas ou peu de sous-échantillonnage
    # ============================================================
    mask_small = mask[::step_mask, ::step_mask, ::step_mask]

    Xm, Ym, Zm = np.mgrid[
        0:mask.shape[0]:step_mask,
        0:mask.shape[1]:step_mask,
        0:mask.shape[2]:step_mask
    ]

    fig.add_trace(go.Isosurface(
        x=Xm.flatten(),
        y=Ym.flatten(),
        z=Zm.flatten(),
        value=mask_small.flatten(),
        isomin=0.5,
        isomax=1,
        surface_count=1,
        opacity=1.0,
        colorscale=[[0, "red"], [1, "red"]],
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        name="Electrode mask"
    ))

    # ============================================================
    # Contacts
    # ============================================================
    mode = "markers+text" if show_labels else "markers"

    fig.add_trace(go.Scatter3d(
        x=df["vox_x"],
        y=df["vox_y"],
        z=df["vox_z"],
        mode=mode,
        marker=dict(size=4, color="cyan"),
        text=df[contact_col].astype(str) if show_labels else None,
        textposition="top center",
        name="Contacts"
    ))

    # ============================================================
    # Lines per electrode
    # ============================================================
    for elec_id, group in df.groupby(electrode_col):
        group = group.sort_values(contact_col)

        fig.add_trace(go.Scatter3d(
            x=group["vox_x"],
            y=group["vox_y"],
            z=group["vox_z"],
            mode="lines",
            line=dict(width=4, color="blue"),
            name=f"Electrode {elec_id}"
        ))

    fig.update_layout(
        title="CT + Electrode Mask + Contacts",
        scene=dict(
            xaxis_title="vox_x",
            yaxis_title="vox_y",
            zaxis_title="vox_z",
            aspectmode="data"
        ),
        showlegend=True
    )

    fig.show()
# ============================================================
# Validation Contact labeling
# ============================================================

def validation_dilatation_atlas(dilatation_width, atlas, contact, brain_mask, output_dir_coregistration, AAL_txt_path, mapping_CT_to_MRI, mri_img, output_dir_COG_011):
    plot_labeled = []
    plot_unlabeled = []
    plot_diff_label = []
    for width in dilatation_width:
        dilated_atlas = atlas_prep.dilate_atlas_labels(atlas, brain_mask, width, output_dir_coregistration)
        label_output = label.label_contacts(contact, dilated_atlas, AAL_txt_path, mapping_CT_to_MRI, mri_img.affine, output_dir_COG_011)

        df_label = pd.read_csv(label_output)
        labeled = []
        unlabeled = []
        diff_label = []
        diff_label = set()

        for _, row in df_label.iterrows():
            if row["label"] == "Unknown 0" or row["label"] == "out_of_bounds":
                unlabeled.append(row)
            else:
                labeled.append(row)
                diff_label.add(row["label"])

        valid_labels = df_label[~df_label["label"].isin(["Unknown 0", "out_of_bounds"])]
        plot_diff_label.append(valid_labels["label"].nunique())
        plot_labeled.append(len(labeled))
        plot_unlabeled.append(len(unlabeled))
    print(f"unique labels: {plot_diff_label}")
    print(f"Labeled: {plot_labeled}")


    plt.figure(figsize=(6, 5))

    plt.scatter(dilatation_width, plot_labeled, color="green", label="Labeled")
    plt.scatter(dilatation_width, plot_unlabeled, color="orange", label="Unlabeled")
    plt.scatter(dilatation_width, plot_diff_label, color="blue", label="Unique labels")

    plt.xlabel("Dilatation width")
    plt.ylabel("Number of contacts")
    plt.title("Atlas Dilatation")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_electrode_label_bars(
    label_csv_path,
    electrode_col="electrode_id",
    contact_col="c_id",
    label_col="label",
    figsize=(10, 6),
    save_path=None
):
    """
    Plot one horizontal stacked bar per electrode.
    Each segment corresponds to consecutive contacts with the same anatomical label.

    Parameters
    ----------
    label_csv_path : str
        Path to the CSV containing electrode_id, c_id and label columns.

    electrode_col : str
        Column containing electrode identifiers.

    contact_col : str
        Column containing contact indices.

    label_col : str
        Column containing anatomical labels.

    exclude_labels : tuple
        Labels considered as unlabeled / invalid.

    figsize : tuple
        Figure size.

    save_path : str or None
        If provided, saves the figure.
    """

    df = pd.read_csv(label_csv_path)

    # Sort contacts along each electrode
    df = df.sort_values([electrode_col, contact_col])

    # Unique labels and colors
    unique_labels = sorted(df[label_col].dropna().unique())

    cmap = plt.get_cmap("tab20")
    color_map = {
        lab: cmap(i % 20)
        for i, lab in enumerate(unique_labels)
    }

    fig, ax = plt.subplots(figsize=figsize)

    electrodes = list(df[electrode_col].unique())

    for y_pos, electrode in enumerate(electrodes):
        df_elec = df[df[electrode_col] == electrode].sort_values(contact_col)

        labels = df_elec[label_col].tolist()

        start = 0
        current_label = labels[0]

        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != current_label:
                length = i - start

                ax.barh(
                    y=y_pos,
                    width=length,
                    left=start,
                    color=color_map[current_label],
                    edgecolor="black",
                    linewidth=0.5
                )

                # Optional text inside segment
                if length >= 2:
                    ax.text(
                        start + length / 2,
                        y_pos,
                        current_label,
                        ha="center",
                        va="center",
                        fontsize=7
                    )

                start = i
                if i < len(labels):
                    current_label = labels[i]

    ax.set_yticks(range(len(electrodes)))
    ax.set_yticklabels(df[electrode_col].unique())
    max_length = df.groupby(electrode_col)[contact_col].max().max()

    ax.set_xticks(range(int(max_length) + 2))
    ax.set_xlabel("Number of contacts along electrode")
    ax.set_ylabel("Electrode")
    ax.set_title("Anatomical labeling along each electrode")

    # Legend
    patches = [
        mpatches.Patch(color=color_map[label], label=label)
        for label in unique_labels
    ]

    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8
    )

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# ==========================================
# Validation co-registration
# ==========================================
def apply_ct_to_mri_mapping(ct_mask_path, mri_mask_path, mapping_path):
    ct_img = nib.load(ct_mask_path)
    mri_img = nib.load(mri_mask_path)

    ct_mask = (ct_img.get_fdata() > 0).astype(np.float32)
    mri_mask_data = mri_img.get_fdata() > 0

    # Load DIPY AffineMap
    mapping_obj = np.load(mapping_path, allow_pickle=True)

    if mapping_obj.shape == ():
        mapping_obj = mapping_obj.item()

    # Apply CT -> MRI transform directly
    ct_mask_in_mri = mapping_obj.transform(
        ct_mask,
        interpolation="nearest"
    )

    ct_mask_in_mri = ct_mask_in_mri > 0.5

    out_img = nib.Nifti1Image(
        ct_mask_in_mri.astype(np.uint8),
        mri_img.affine,
        header=mri_img.header
    )

    #nib.save(out_img, output_path)

    return ct_mask_in_mri, mri_mask_data

def apply_dipy_mapping(source_mask_path, target_mask_path, mapping_path):
    source_img = nib.load(source_mask_path)
    target_img = nib.load(target_mask_path)

    source_mask = (source_img.get_fdata() > 0).astype(np.float32)
    target_mask = target_img.get_fdata() > 0

    mapping_obj = np.load(mapping_path, allow_pickle=True)
    if mapping_obj.shape == ():
        mapping_obj = mapping_obj.item()

    source_mask_in_target = mapping_obj.transform(
        source_mask,
        interpolation="nearest"
    ) > 0.5

    return source_mask_in_target, target_mask

def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    volume_sum = mask1.sum() + mask2.sum()

    if volume_sum == 0:
        return np.nan

    return 2 * intersection / volume_sum



dilatation_width = [0, 1, 2, 3, 4, 5,6,7,8,9,10]
atlas = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\coregistration\aal_coregistered_to_MNI152_T1_1mm_brain_coregistered_to_75158154_t1_mprage_sag_p2_iso_new_20230424160641_3_strip.nii.gz"
contact = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\contacts_shifted_corrected.csv"
brain_mask = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\coregistration\MNI152_T1_2mm_brain_mask_coregistered_to_75158154_t1_mprage_sag_p2_iso_new_20230424160641_3_strip.nii.gz"
output_dir_coregistration = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\coregistration"
AAL_txt_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\Atlas\AAL_atlas.txt"
mapping_CT_to_MRI = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\coregistration\33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip_mapping.npy"
mri_img = nib.load(r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\MRI\95545094_t1_mprage_sag_p3_iso_20210930101225_5_strip.nii.gz")
output_dir_COG_011 = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out"
#validation_dilatation_atlas(dilatation_width, atlas, contact, brain_mask, output_dir_coregistration, AAL_txt_path, mapping_CT_to_MRI, mri_img, output_dir_COG_011)

