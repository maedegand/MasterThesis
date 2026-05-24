import os
from pathlib import Path

import test_validation as test_val
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img


if __name__ == "__main__":

    # ============================================================
    # Base directory
    # ============================================================
    BASE_DIR = Path(r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data")

    # ============================================================
    # Patient paths
    # ============================================================
    PATIENTS = {
        "COG_011": {
            "ct": BASE_DIR / "COG_011" / "CT" / "51138313_Volume_TF_EB2_20231121142028_302_strip.nii.gz",
            "electrode_mask": BASE_DIR / "COG_011" / "out" / "mask_generated" / "51138313_Volume_TF_EB2_20231121142028_302_strip.nii.gz_electrode_mask.nii.gz",
            "ground_truth": BASE_DIR / "COG_011" / "ground_truth" / "groundtruth_COG_011.csv",
            "ground_truth_entry_point": BASE_DIR / "COG_011" / "ground_truth" / "groundtruth_entry_points_COG_011.csv",
            "mri_mask": BASE_DIR / "COG_011" / "MRI" / "95545094_t1_mprage_sag_p3_iso_20210930101225_5_brain_mask.nii.gz",
            "ct_brain_mask": BASE_DIR / "COG_011" / "CT" / "51138313_Volume_TF_EB2_20231121142028_302_brain_mask.nii.gz",
            "mapping_CT_to_MRI": BASE_DIR / "COG_011" / "out" / "coregistration" / "51138313_Volume_TF_EB2_20231121142028_302_strip_mapping.npy",
            "mapping_MNI_to_MRI": BASE_DIR / "COG_011" / "out" / "coregistration" / "MNI152_T1_2mm_brain_mask_mapping.npy"
        },

        "COG_004": {
            "ct": BASE_DIR / "COG_004" / "CT" / "33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip.nii.gz",
            "electrode_mask": BASE_DIR / "COG_004" / "out" / "mask_generated" / "33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip.nii.gz_electrode_mask.nii.gz",
            "ground_truth": BASE_DIR / "COG_004" / "ground_truth" / "groundtruth_COG_004.csv",
            "ground_truth_entry_point": BASE_DIR / "COG_004" / "ground_truth" / "groundtruth_entry_points_COG_004.csv",
            "mri_mask": BASE_DIR / "COG_004" / "IRM" / "75158154_t1_mprage_sag_p2_iso_new_20230424160641_3_brain_mask.nii.gz",
            "ct_brain_mask": BASE_DIR / "COG_004" / "CT" / "33312121_CT_cerebrum_uden_kontrast_20230501091320_5brain_mask.nii.gz",
            "mapping_CT_to_MRI": BASE_DIR / "COG_004" / "out" / "coregistration" / "33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip_mapping.npy",
            "mapping_MNI_to_MRI": BASE_DIR / "COG_004" / "out" / "coregistration" / "MNI152_T1_2mm_brain_mask_mapping.npy"
        },
    }

    # ============================================================
    # Patient selection
    # ============================================================
    PATIENT_ID = "COG_011"
    #PATIENT_ID = "COG_004"

    patient = PATIENTS[PATIENT_ID]

    # ============================================================
    # General directories
    # ============================================================
    patient_dir = BASE_DIR / PATIENT_ID
    out_dir = patient_dir / "out"
    validation_dir = out_dir / "validation"
    input_electroloc_dir = out_dir / "input_ElectroLoc"

    validation_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Input paths
    # ============================================================
    CT_path = patient["ct"]
    mask_path = patient["electrode_mask"]
    ground_truth_path = patient["ground_truth"]
    ground_truth_entry_points_path = patient["ground_truth_entry_point"]

    output_electroLoc = out_dir / "output_ElectroLoc.csv"
    output_indexation = out_dir / "contacts_reordered.csv"
    output_shift = out_dir / "contacts_shifted_corrected.csv"

    input_ElectroLoc_path = input_electroloc_dir / "input_ElectroLoc_world.csv"
    output_data_preprocessing = input_electroloc_dir / "entry_points_world.csv"
  

    # ============================================================
    # Output validation paths
    # ============================================================
    contact_validation_path = validation_dir / "contacts_validation.csv"
    contact_validation_path2 = validation_dir / "contacts_shifted_validation.csv"
    entry_validation_path = validation_dir / "entry_point_validation.csv"
    entry_validation_GT_path = validation_dir / "entry_point_GT.csv"

    validation_output_electroloc = validation_dir / "validation_output_ElectroLoc.csv"
    validation_output_indexation = validation_dir / "validation_output_indexation.csv"
    validation_output_shift = validation_dir / "validation_output_shift.csv"

    # ============================================================
    # Safety checks
    # ============================================================
    required_files = {
        "CT_path": CT_path,
        "mask_path": mask_path,
        "ground_truth_path": ground_truth_path,
        "groundtruth_entry_points_path" : ground_truth_entry_points_path,
        "output_electroLoc": output_electroLoc,
        "output_indexation": output_indexation,
        "output_shift": output_shift,
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # ============================================================
    # Validation contacts with electrode mask
    # ============================================================
    print("\nValidation contacts with electrode mask:")

    test_val.validate_contacts_with_electrode_mask(
        mask_path=str(mask_path),
        contact_path=str(output_electroLoc),
        output_path=str(contact_validation_path)
    )

    test_val.validate_contacts_with_electrode_mask(
        mask_path=str(mask_path),
        contact_path=str(output_shift),
        output_path=str(contact_validation_path2)
    )

    # ============================================================
    # Validation entry points
    # ============================================================
    print("\nValidation entry points:")
    print(ground_truth_entry_points_path)

    test_val.validate_entry_points(
        contact_path=str(output_electroLoc),
        entry_points_path=str(input_ElectroLoc_path),
        output_path=str(entry_validation_path),
        mask_path=str(mask_path)
    )
    # ============================================================
    # Validation entry points with ground truth
    # ============================================================
    print("\nValidation entry points with ground truth:")

    test_val.validate_entry_points_with_ground_truth(
        ground_truth_path=ground_truth_entry_points_path,
        entry_points_path=input_ElectroLoc_path,
        output_path=str(entry_validation_GT_path)
    )
    

    # ============================================================
    # Ground truth-based contact validation
    # ============================================================
    meandistance, df = test_val.compute_mean_intercontact_distance(
        str(ground_truth_path)
    )

    print(f"\nMean inter-contact distance: {meandistance:.3f} mm")

    threshold = meandistance * 0.75
    print(f"Using distance threshold of {threshold:.3f} mm for validation.")

    print("\nValidation ElectroLoc:")
    validate_results_outputElectroloc, median1 = test_val.validation_contact_loc_with_ground_truth(
        str(ground_truth_path),
        str(output_electroLoc),
        str(validation_output_electroloc),
        threshold=threshold,
        electrode_col="electrode_id_nb"
    )

    print("\nValidation Indexation:")
    validate_results_output_indexation, median2 = test_val.validation_contact_loc_with_ground_truth(
        str(ground_truth_path),
        str(output_indexation),
        str(validation_output_indexation),
        threshold=threshold,
        electrode_col="electrode_id"
    )

    print("\nValidation Shift:")
    validate_results_output_shift, median3 = test_val.validation_contact_loc_with_ground_truth(
        str(ground_truth_path),
        str(output_shift),
        str(validation_output_shift),
        threshold=threshold,
        electrode_col="electrode_id"
    )

    # ============================================================
    # Optional plots
    # ============================================================
    # Uncomment if needed
    #
    # ct_img = nib.load(str(CT_path))
    # ct = ct_img.get_fdata()
    #
    # mask_img = nib.load(str(mask_path))
    # mask = mask_img.get_fdata()
    #
    # contacts = pd.read_csv(str(contacts_path))
    #
    # test_val.plot_3d(
    #     ct,
    #     mask,
    #     contacts,
    #     step_ct=6,
    #     step_mask=1,
    #     show_ct=False,
    #     show_labels=False
    # )

    # ============================================================
    # Summary
    # ============================================================
    print("\n====================================")
    print("Validation summary")
    print("====================================")
    print(f"ElectroLoc median distance: {median1:.3f} mm")
    print(f"Indexation median distance: {median2:.3f} mm")
    print(f"Shift median distance:      {median3:.3f} mm")
    print("====================================")

    # ============================================================
    # Co-registration validation with Dice score
    # ============================================================
    print("\nCo-registration validation with Dice score:")

    mni_mask = BASE_DIR / "Atlas_Maps" / "MNI152_T1_2mm_brain_mask.nii.gz"

    ct_mask_coreg, mri_mask_data = test_val.apply_dipy_mapping(
        str(patient["ct_brain_mask"]),
        str(patient["mri_mask"]),
        str(patient["mapping_CT_to_MRI"])
    )

    ct_mask_before = resample_to_img(
        source_img=str(patient["ct_brain_mask"]),
        target_img=str(patient["mri_mask"]),
        interpolation="nearest"
    )

    dice_ct_before = test_val.dice_coefficient(
        mri_mask_data,
        ct_mask_before.get_fdata() > 0
    )

    dice_ct_after = test_val.dice_coefficient(
        mri_mask_data,
        ct_mask_coreg
    )

    mni_mask_coreg, mri_mask_data = test_val.apply_dipy_mapping(
        str(mni_mask),
        str(patient["mri_mask"]),
        str(patient["mapping_MNI_to_MRI"])
    )

    mni_mask_before = resample_to_img(
        source_img=str(mni_mask),
        target_img=str(patient["mri_mask"]),
        interpolation="nearest"
    )

    dice_mni_before = test_val.dice_coefficient(
        mri_mask_data,
        mni_mask_before.get_fdata() > 0
    )

    dice_mni_after = test_val.dice_coefficient(
        mri_mask_data,
        mni_mask_coreg
    )

    print(f"Dice CT to MRI before:  {dice_ct_before:.3f}")
    print(f"Dice CT to MRI after:   {dice_ct_after:.3f}")
    print(f"Dice MNI to MRI before: {dice_mni_before:.3f}")
    print(f"Dice MNI to MRI after:  {dice_mni_after:.3f}")