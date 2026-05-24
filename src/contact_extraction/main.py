import nii_preprocessing as nii_prep    
import atlas as atlas_prep
import edf_preprocessing as edf_prep
import csv_preprocessing as csv_prep
import shift_handling as shift_handling
import test_validation as test_val
import labeling as label

import nibabel as nib
import numpy as np
import time
import pandas as pd
import subprocess
import os
import sys
from pathlib import Path


# ============================================================
# Patient configuration
# ============================================================

BASE_DIR = Path(r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data")
PROJECT_DIR = Path(r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE")
ELECTROLOC_PARENT = PROJECT_DIR / "MasterThesis_ElectroLoc"

PATIENT_ID = "COG_011"   # <-- Change only this line: "COG_011" or "COG_004"


PATIENTS = {
    "COG_011": {
        "edf": BASE_DIR / "COG_011" / "iEEG" / "COG_011_BodyBc241123.edf",
        "ct": BASE_DIR / "COG_011" / "CT" / "51138313_Volume_TF_EB2_20231121142028_302_strip.nii.gz",
        "mri": BASE_DIR / "COG_011" / "MRI" / "95545094_t1_mprage_sag_p3_iso_20210930101225_5_strip.nii.gz",
        "brain_mask_ct": BASE_DIR / "COG_011" / "CT" / "51138313_Volume_TF_EB2_20231121142028_302_brain_mask.nii.gz",
    },

    "COG_004": {
        "edf": BASE_DIR / "COG_004" / "edf" / "COG_004_BodyA130423.edf",
        "ct": BASE_DIR / "COG_004" / "CT" / "33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip.nii.gz",
        "mri": BASE_DIR / "COG_004" / "IRM" / "75158154_t1_mprage_sag_p2_iso_new_20230424160641_3_strip.nii.gz",
        "brain_mask_ct": BASE_DIR / "COG_004" / "CT" / "33312121_CT_cerebrum_uden_kontrast_20230501091320_5brain_mask.nii.gz",
    },
}


# ============================================================
# Atlas paths
# ============================================================

MNI_path = BASE_DIR / "Atlas_Maps" / "MNI152_T1_1mm_brain.nii.gz"
AAL_path = BASE_DIR / "Atlas" / "aal.nii.gz"
AAL_txt_path = BASE_DIR / "Atlas" / "AAL_atlas.txt"
brain_mask_MNI_path = BASE_DIR / "Atlas_Maps" / "MNI152_T1_2mm_brain_mask.nii.gz"


# ============================================================
# Utility functions
# ============================================================

def get_patient_paths(patient_id):
    if patient_id not in PATIENTS:
        raise ValueError(f"Unknown patient ID: {patient_id}")

    paths = PATIENTS[patient_id].copy()

    patient_dir = BASE_DIR / patient_id
    output_dir = patient_dir / "out"

    paths["patient_dir"] = patient_dir
    paths["output_dir"] = output_dir
    paths["output_dir_coregistration"] = output_dir / "coregistration"
    paths["output_dir_input_ElectroLoc"] = output_dir / "input_ElectroLoc"
    paths["output_dir_mask"] = output_dir / "mask_generated"

    paths["output_csv_ElectroLoc"] = output_dir / "output_ElectroLoc.csv"
    paths["contacts_shifted_corrected"] = output_dir / "contacts_shifted_corrected.csv"
    paths["contact_shifted_corrected_mask"] = output_dir / "contact_shifted_corrected.nii.gz"

    return paths


def ensure_dirs(paths):
    dirs_to_create = [
        paths["output_dir"],
        paths["output_dir_coregistration"],
        paths["output_dir_input_ElectroLoc"],
        paths["output_dir_mask"],
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)


def check_existing_paths(paths):
    required = [
        "edf",
        "ct",
        "mri",
        "brain_mask_ct",
    ]

    for key in required:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing file for {key}: {paths[key]}")





if __name__ == "__main__":

    # ============================================================
    # Load patient-specific paths
    # ============================================================

    paths = get_patient_paths(PATIENT_ID)
    ensure_dirs(paths)
    check_existing_paths(paths)

    path_edf = paths["edf"]
    path_CT = paths["ct"]
    path_mri = paths["mri"]
    brain_mask_CT_path = paths["brain_mask_ct"]

    output_dir = paths["output_dir"]
    output_dir_coregistration = paths["output_dir_coregistration"]
    output_dir_input_ElectroLoc = paths["output_dir_input_ElectroLoc"]
    output_dir_mask = paths["output_dir_mask"]

    output_csv_ElectroLoc = paths["output_csv_ElectroLoc"]

    print(f"Running pipeline for patient: {PATIENT_ID}")

    # ============================================================
    # Data Preprocessing
    # ============================================================

    print("Starting data preprocessing...")
    start_data_prep = time.time()

    # Extract information from EDF file
    data_edf, channel_names = edf_prep.load_edf(
        str(path_edf),
        str(output_dir)
    )

    output_electrode_info = edf_prep.extract_electrodes(
        channel_names,
        str(path_edf),
        str(output_dir_input_ElectroLoc)
    )

    # Create masks for electrodes and brain
    mask_dir_electrodes = nii_prep.electrode_mask(
        str(path_CT),
        str(output_dir_mask)
    )

    mask_dir_surface = nii_prep.extract_surface_mask(
        str(brain_mask_CT_path),
        str(output_dir_mask)
    )

    # Find entry points of electrodes and create a mask for them
    entry_point_voxel_path, entry_point_world_path = csv_prep.entry_point_coordinates(
        mask_dir_surface,
        mask_dir_electrodes,
        str(output_dir_input_ElectroLoc)
    )
    
    """mask_entry_dir = nii_prep.create_contact_mask_from_csv(
        entry_point_voxel_path,
        path_CT,
        str(output_dir_input_ElectroLoc),
        radius_vox=1
    )"""

    

    end_data_prep = time.time()
    print("Time taken for data preprocessing:", end_data_prep - start_data_prep, "seconds")
    print("Data preprocessing completed.")

    # ============================================================
    # Manual Electrode Identification
    # ============================================================

    print("Starting manual electrode identification...")

    electrode_info_path, df_electrode_info = csv_prep.get_valid_csv_path(
        prompt="Enter path to electrode_info.csv: ",
        required_columns=["nb_contacts", "vox_x", "vox_y", "vox_z"],
        name="electrode_info.csv"
    )

    electrode_info_world_path, df_electrode_info_world = csv_prep.get_valid_csv_path(
        prompt="Enter path to electrode_info_world.csv: ",
        required_columns=["electrode_id", "nb_contacts", "world_x", "world_y", "world_z"],
        name="electrode_info_world.csv"
    )

    print("Manual electrode identification completed.")

    # ============================================================
    # Electrode Localization with ElectroLoc
    # ============================================================

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ELECTROLOC_PARENT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m", "ElectroLoc",
        str(path_CT),
        str(brain_mask_CT_path),
        str(electrode_info_path),
        "-o",
        str(output_csv_ElectroLoc)
    ]

    print("Running ElectroLoc...")
    subprocess.run(cmd, check=True, env=env)

    print(f"Electrode localization completed. Output saved to: {output_csv_ElectroLoc}")

    nii_prep.create_contact_mask_from_csv(
        str(output_csv_ElectroLoc),
        str(path_CT),
        str(output_dir)
    )

    # ============================================================
    # Contact Labeling
    # ============================================================

    print("Starting contact labeling...")
    start_contact_labeling = time.time()

    # Renumbering contacts
    reordered_contact, contact_final = label.numbering_contact(
        str(output_csv_ElectroLoc),
        str(electrode_info_world_path),
        str(output_dir)
    )

    # Handle potential shift issues
    shifted_electrode_ids = shift_handling.selection_of_potential_shifted_electrode(
        reordered_contact,
        mask_dir_electrodes,
        str(electrode_info_world_path)
    )

    contact_shifted_corrected_path, df_out = shift_handling.shift_selected_electrodes_with_mask(
        reordered_contact,
        str(electrode_info_world_path),
        mask_dir_electrodes,
        shifted_electrode_ids,
        output_path=str(paths["contacts_shifted_corrected"]),
        max_shift_mm=20,
        step_mm=0.1,
        lambda_EP=0.5,
        c0_mask_threshold_mm=0.5
    )

    nii_prep.create_contact_mask_from_csv(
        contact_shifted_corrected_path,
        ct_path=str(path_CT),
        output_dir=str(output_dir)
    )

    # ============================================================
    # Atlas co-registration
    # ============================================================

    # Option 1: run co-registration again
    # Uncomment this block if you want to recompute the co-registrations.

    
    coregistered_MNI_to_mri, mapping_MNI_to_MRI = atlas_prep.coregister_1_to_2(
        str(MNI_path),
        str(path_mri),
        str(output_dir_coregistration)
    )

    coregistered_aal_to_MNI_coregistered, mapping_aal_to_MNI = atlas_prep.coregister_1_to_2(
        str(AAL_path),
        coregistered_MNI_to_mri,
        str(output_dir_coregistration)
    )

    coregistered_brain_mask_MNI_to_MRI, mapping_brain_mask_MNI_to_MRI = atlas_prep.coregister_1_to_2(
        str(brain_mask_MNI_path),
        str(path_mri),
        str(output_dir_coregistration)
    )

    coregistered_CT_to_MRI, mapping_CT_to_MRI = nii_prep.coregistration_CT_to_MRI(
        str(path_CT),
        str(path_mri),
        str(output_dir_coregistration)
    )
    

    # Option 2: use already existing files
    # This assumes the co-registration outputs already exist in the patient's out/coregistration folder.


    #coregistered_aal_to_MNI_coregistered = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\coregistration\aal_coregistered_to_MNI152_T1_1mm_brain_coregistered_to_95545094_t1_mprage_sag_p3_iso_20210930101225_5_strip.nii.gz"
    #mapping_CT_to_MRI = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\coregistration\51138313_Volume_TF_EB2_20231121142028_302_strip_mapping.npy"
    #coregistered_brain_mask_MNI_to_MRI = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\coregistration\MNI152_T1_2mm_brain_mask_coregistered_to_95545094_t1_mprage_sag_p3_iso_20210930101225_5_strip.nii.gz"
    

    

    mri_img = nib.load(str(path_mri))

    # Optional: dilate atlas
    dilate_atlas_path = atlas_prep.dilate_atlas_labels(
         str(coregistered_aal_to_MNI_coregistered),
         str(coregistered_brain_mask_MNI_to_MRI),
         6,
         str(output_dir_coregistration)
    )

    # Label contacts based on their coordinates and the atlas
    label_output = label.label_contacts(
        contact_shifted_corrected_path,
        str(dilate_atlas_path),
        str(AAL_txt_path),
        str(mapping_CT_to_MRI),
        mri_img.affine,
        str(output_dir)
    )

    end_contact_labeling = time.time()
    print("Time taken for contact labeling:", end_contact_labeling - start_contact_labeling, "seconds")
    print("Contact labeling completed.")