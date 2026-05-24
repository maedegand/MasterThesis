import os
from nibabel.affines import apply_affine
import pandas as pd
import nibabel  as nib
import numpy as np


def label_contacts (contact_list_path, atlas_path,atlas_txt_path, mapping_path_CT_to_MRI,mri_affine, output_dir):
    """Label contacts based on their coordinates and the atlas.
    Args:
        contact_list_path (str): Path to the CSV file containing contact coordinates.
        atlas_path (str): Path to the atlas NIfTI file.
        atlas_txt_path (str): Path to the atlas text file.
        mapping_path (str): Path to the mapping file.
        mri_affine (numpy.ndarray): Affine matrix for the MRI space.
        output_dir (str): Directory to save the labeled contacts file   .
    Returns:
        str: Path to the labeled contacts file.
    """
     # ===================
    # Load Args
    # ===================
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine

    contact_list = pd.read_csv(contact_list_path)
    coords_world_CT = contact_list[["world_x", "world_y", "world_z"]].values

    mapping_CT_to_MRI = np.load(mapping_path_CT_to_MRI, allow_pickle=True).item()

    # ===================
    # Transform contact coordinates from CT space to atlas space
    # world CT --> world MRI --> voxel MRI --> voxel Atlas
    # ===================
    coords_world_MRI = apply_affine(np.linalg.inv(mapping_CT_to_MRI.affine), coords_world_CT)
    coords_world_MRI_csv = pd.DataFrame(coords_world_MRI, columns=["x_world_MRI", "y_world_MRI", "z_world_MRI"])
    output_path_csv = output_dir + "coords_world_MRI.csv"
    coords_world_MRI_csv.to_csv(output_path_csv, index=False)

    print(f"CSV saved: {output_path_csv}")
    coords_voxel_MRI = apply_affine(np.linalg.inv(mri_affine), coords_world_MRI)
    coords_voxel_atlas = np.round(coords_voxel_MRI).astype(int)
    #coords_voxel_atlas = np.round(apply_affine(np.linalg.inv(atlas_affine), coords_world_MRI)).astype(int)
    
    # ==============
    # Load atlas labels 
    # ==============
    def load_labels(atlas_txt_path):
        atlas_labels = {}
        with open(atlas_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip()
                if not line: 
                    continue 
                parts = line.split()
                try:
                    idx = int(parts[1])
                    name = "".join(parts[3:])
                    atlas_labels[idx] = name
                except ValueError:
                    print(f"Skipping line due to parsing error: {line}")
                    continue
        return atlas_labels
    
    atlas_labels = load_labels(atlas_txt_path)
    
    # ===============
    # Get labels for each contact
    # ===============
    labels = []
    for coord in coords_voxel_atlas:
        x, y, z = np.round(coord).astype(int)
        if (0 <= x < atlas_img.shape[0]) and (0 <= y < atlas_img.shape[1]) and (0 <= z < atlas_img.shape[2]):
            label_value = int(round(atlas_data[x,y,z]))
            labels.append(atlas_labels.get(label_value, f"Unknown {label_value}")) #check si c'est 0 ou + grand que 116
        else:
            labels.append("out_of_bounds")  # Out of bounds

    output_df = pd.DataFrame({
    "electrode_id": contact_list["electrode_id"].astype(str),
    "contact_id": contact_list["c_id"],
    "vox_x": coords_voxel_MRI[:,0],
    "vox_y": coords_voxel_MRI[:,1],
    "vox_z": coords_voxel_MRI[:,2],
    "world_x": coords_world_MRI[:,0],
    "world_y": coords_world_MRI[:,1],
    "world_z": coords_world_MRI[:,2],
    "label": labels
    })
    output_path = os.path.join(output_dir, "labeled_contacts.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Labeled contacts saved at: {output_path}")  
    return output_path



def numbering_contact(contact_list_path, entry_point_world_list_path, output_dir):
    """
    Rename electrode_id using entry point list and reorder contacts inside each electrode.

    Main idea:
    - Contact groups are already in the same order as entry points.
    - Entry point is the superficial reference point.
    - Contacts are ordered using their signed projection on the electrode axis.
    - Deepest contact gets c_id = 0.
    - Contacts outside the skull are kept.
    """

    os.makedirs(output_dir, exist_ok=True)

    entry_df = pd.read_csv(entry_point_world_list_path).copy()
    contact_df = pd.read_csv(contact_list_path).copy()

    grouped_contacts = list(contact_df.groupby("electrode_id", sort=False))

    if len(grouped_contacts) != len(entry_df):
        print(
            f"Warning: {len(grouped_contacts)} contact groups found, "
            f"but {len(entry_df)} entry points provided."
        )

    ordered_groups = []

    for i, ((old_electrode_id, group), (_, entry_row)) in enumerate(
        zip(grouped_contacts, entry_df.iterrows())
    ):
        group = group.copy()

        entry_point = entry_row[["world_x", "world_y", "world_z"]].to_numpy(dtype=float)
        new_electrode_id = entry_row["electrode_id"]
        nb_expected = int(entry_row["nb_contacts"])

        coords = group[["world_x", "world_y", "world_z"]].to_numpy(dtype=float)

        # Vectors from entry point to contacts
        vectors = coords - entry_point

        # Estimate the electrode axis using PCA/SVD
        _, _, vh = np.linalg.svd(vectors, full_matrices=False)
        axis = vh[0]

        # Project contacts onto the estimated electrode axis
        projections = vectors @ axis

        # Orient the axis so that most contacts are in the positive direction
        if np.mean(projections) < 0:
            axis = -axis
            projections = -projections

        group["signed_dist_to_entry"] = projections

        # Sort from deepest to most superficial
        # Deepest contact = largest signed distance from entry point
        # Most superficial/outside contact = smallest signed distance
        group = group.sort_values(
            "signed_dist_to_entry",
            ascending=False
        ).reset_index(drop=True)

        # Rename electrode
        group["electrode_id"] = new_electrode_id

        # Re-index contacts: deepest contact gets c_id = 0
        group["c_id"] = np.arange(len(group))

        if len(group) != nb_expected:
            print(
                f"Warning: electrode {new_electrode_id} has {len(group)} contacts, "
                f"expected {nb_expected}"
            )

        ordered_groups.append(group)

    contact_final = pd.concat(ordered_groups, ignore_index=True)

    contact_final = contact_final[
        [
            "vox_x", "vox_y", "vox_z",
            "world_x", "world_y", "world_z",
            "electrode_id", "c_id"
        ]
    ]

    output_path = os.path.join(output_dir, "contacts_reordered.csv")
    contact_final.to_csv(output_path, index=False)

    print(f"Saved reordered contacts to: {output_path}")

    return output_path, contact_final


