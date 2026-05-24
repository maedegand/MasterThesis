import pandas as pd
import nibabel as nib
import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import test_validation as test_val 

def selection_of_potential_shifted_electrode(reordered_contact, electrode_mask, electrode_info_world_path):
    dist_entry_point_path = test_val.validate_entry_points(reordered_contact, electrode_info_world_path, output_path=r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation\entry_point_validation.csv")
    dist_mask_path, df = test_val.validate_contacts_with_electrode_mask(electrode_mask, reordered_contact,output_path=r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation\contacts_validation.csv")
    dist_entry_point = pd.read_csv(dist_entry_point_path)
    dist_mask = pd.read_csv(dist_mask_path)


    # ------------------------------------------------------------
    # Condition 1: if distance between entry point and c_id = max is greater than 5mm
    # ------------------------------------------------------------
    electrodes_entry_point_issue = dist_entry_point.loc[
        dist_entry_point["distance_mm"] > 5,
        "electrode_id"
    ].unique()

    # ------------------------------------------------------------
    # Condition 2: if distance between electrode mask and c_id = 0 is greater than 2mm
    # ------------------------------------------------------------
    electrodes_mask_issue = dist_mask.loc[
        (dist_mask["c_id"] == 0) &
        (dist_mask["distance_to_mask_mm"] > 2),
        "electrode_id"
    ].unique()

    # ------------------------------------------------------------
    # Potential shifted electrodes list
    # ------------------------------------------------------------
    #shifted_electrode_ids = sorted(set(electrodes_entry_point_issue).union(set(electrodes_mask_issue)))
    shifted_electrode_ids = electrodes_mask_issue

    print("Electrode IDs with entry point issue:", electrodes_entry_point_issue)
    print("Electrode IDs with mask issue at c_id=0:", electrodes_mask_issue)
    print("Electrode IDs with potential shift issues:", shifted_electrode_ids)
    print("Electrode IDs with potential shift issues:", shifted_electrode_ids)
    return shifted_electrode_ids
def shift_selected_electrodes_with_mask(
    contacts_path,
    entry_point_path,
    electrode_mask_path,
    electrode_ids_to_shift,
    output_path,
    max_shift_mm=20,
    step_mm=0.2,
    lambda_EP=0.5,
    c0_mask_threshold_mm=1.0
):
    import pandas as pd
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt

    df_contacts = pd.read_csv(contacts_path)
    df_out = df_contacts.copy()

    df_entry_val = pd.read_csv(entry_point_path)

    mask_img = nib.load(electrode_mask_path)
    mask = mask_img.get_fdata()
    mask_bin = (mask > 0).astype(np.uint8)

    voxel_sizes = mask_img.header.get_zooms()[:3]
    inv_affine = np.linalg.inv(mask_img.affine)

    dist_map_mm = distance_transform_edt(
        1 - mask_bin,
        sampling=voxel_sizes
    )

    for elec_id in electrode_ids_to_shift:

        idx = df_out[df_out["electrode_id"] == elec_id].index

        if len(idx) == 0:
            print(f"Warning: electrode {elec_id} not found.")
            continue

        group = df_out.loc[idx].sort_values("c_id").copy()
        coords_world = group[["world_x", "world_y", "world_z"]].to_numpy(dtype=float)

        entry_row = df_entry_val[df_entry_val["electrode_id"] == elec_id]

        if entry_row.empty:
            print(f"Warning: no entry point found for electrode {elec_id}.")
            continue

        entry_row = entry_row.iloc[0]
        entry_world = entry_row[["world_x", "world_y", "world_z"]].to_numpy(dtype=float)

        diffs = np.diff(coords_world, axis=0)

        if len(diffs) == 0:
            print(f"Warning: electrode {elec_id} has only one contact.")
            continue

        # ============================================================
        # Electrode direction
        # ============================================================
        step_vector = np.median(diffs, axis=0)
        axis = step_vector / np.linalg.norm(step_vector)

        # ============================================================
        # Current distances to mask
        # ============================================================
        coords_h_current = np.c_[coords_world, np.ones(len(coords_world))]
        coords_vox_current = (inv_affine @ coords_h_current.T).T[:, :3]

        current_mask_distances = []

        for v in coords_vox_current:
            x, y, z = np.round(v).astype(int)

            if (
                0 <= x < dist_map_mm.shape[0]
                and 0 <= y < dist_map_mm.shape[1]
                and 0 <= z < dist_map_mm.shape[2]
            ):
                current_mask_distances.append(dist_map_mm[x, y, z])
            else:
                current_mask_distances.append(1000.0)

        current_mask_distances = np.array(current_mask_distances)
        initial_c0_mask_dist = current_mask_distances[0]
        n_outside_mask = np.sum(current_mask_distances > 2.0)

        # ============================================================
        # Detection of contact-less shaft
        # ============================================================
        dists_to_entry = np.linalg.norm(coords_world - entry_world, axis=1)
        closest_idx = int(np.argmin(dists_to_entry))

        if (
            closest_idx == len(coords_world) - 1
            and n_outside_mask == 0
        ):
            print(f"{elec_id}: Contact-less shaft → no shift")

            df_out.loc[group.index, "applied_shift_mm"] = 0.0
            df_out.loc[group.index, "shift_status"] = "Contact-less shaft → no shift"
            df_out.loc[group.index, "entry_distance_after_shift_mm"] = dists_to_entry[-1]
            df_out.loc[group.index, "mean_mask_distance_after_shift_mm"] = np.mean(current_mask_distances)
            df_out.loc[group.index, "c0_mask_distance_after_shift_mm"] = initial_c0_mask_dist

            continue

        # ============================================================
        # Detection of partially implanted electrode
        # ============================================================
        is_partially_implanted = initial_c0_mask_dist > c0_mask_threshold_mm

        if is_partially_implanted:
            print(
                f"{elec_id}: partially implanted electrode detected "
                f"(initial c_id=0 mask distance = {initial_c0_mask_dist:.2f} mm)"
            )

        # ============================================================
        # Search for best shift
        # ============================================================
        best_score = np.inf
        best_shift_mm = 0.0
        best_coords_world = coords_world.copy()
        best_entry_dist = None
        best_mask_dist = None
        best_c0_mask_dist = None

        shift_values = np.arange(-max_shift_mm, max_shift_mm + step_mm, step_mm)

        for shift_mm in shift_values:

            candidate_world = coords_world + shift_mm * axis

            superficial_world = candidate_world[-1]
            entry_dist = np.linalg.norm(superficial_world - entry_world)

            coords_h = np.c_[candidate_world, np.ones(len(candidate_world))]
            coords_vox = (inv_affine @ coords_h.T).T[:, :3]

            mask_distances = []

            for v in coords_vox:
                x, y, z = np.round(v).astype(int)

                if (
                    0 <= x < dist_map_mm.shape[0]
                    and 0 <= y < dist_map_mm.shape[1]
                    and 0 <= z < dist_map_mm.shape[2]
                ):
                    mask_distances.append(dist_map_mm[x, y, z])
                else:
                    mask_distances.append(1000.0)

            mask_distances = np.array(mask_distances)

            mean_mask_dist = np.mean(mask_distances)
            c0_mask_dist = mask_distances[0]

            if is_partially_implanted:
                # For partially implanted electrodes, the entry point is not reliable.
                # The shift is selected by minimizing the distance between c_id=0 and the mask.
                lambda_current = 0.0
                
            else:
                # Standard case: balance entry-point alignment and mask consistency.
                lambda_current = lambda_EP
            score = lambda_current * entry_dist +  mean_mask_dist

            if score < best_score:
                best_score = score
                best_shift_mm = shift_mm
                best_coords_world = candidate_world.copy()
                best_entry_dist = entry_dist
                best_mask_dist = mean_mask_dist
                best_c0_mask_dist = c0_mask_dist

        # ============================================================
        # Updating
        # ============================================================
        df_out.loc[group.index, ["world_x", "world_y", "world_z"]] = best_coords_world

        coords_h = np.c_[best_coords_world, np.ones(len(best_coords_world))]
        best_coords_vox = (inv_affine @ coords_h.T).T[:, :3]

        df_out.loc[group.index, "vox_x"] = best_coords_vox[:, 0]
        df_out.loc[group.index, "vox_y"] = best_coords_vox[:, 1]
        df_out.loc[group.index, "vox_z"] = best_coords_vox[:, 2]

        if is_partially_implanted:
            shift_status = "Partially implanted --> shift based on mask only"
        else:
            shift_status = "Standard case --> shift based on entry point + mask"

        df_out.loc[group.index, "applied_shift_mm"] = np.round(best_shift_mm, 1)
        df_out.loc[group.index, "entry_distance_after_shift_mm"] = best_entry_dist
        df_out.loc[group.index, "mean_mask_distance_after_shift_mm"] = best_mask_dist
        df_out.loc[group.index, "c0_mask_distance_after_shift_mm"] = best_c0_mask_dist
        df_out.loc[group.index, "shift_status"] = shift_status

        print(
            f"{elec_id}: shift = {best_shift_mm:.1f} mm | "
            f"entry dist = {best_entry_dist:.2f} mm | "
            f"mask dist = {best_mask_dist:.2f} mm | "
            f"c0 mask dist = {best_c0_mask_dist:.2f} mm | "
            f"status = {shift_status}"
        )

    df_out.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return output_path, df_out


reordered_contact = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\contacts_reordered.csv"
electrode_info_world_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\input_ElectroLoc\input_ElectroLoc_world.csv"
electrode_mask = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\mask_generated\33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip.nii.gz_electrode_mask.nii.gz"


