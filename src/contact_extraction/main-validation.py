import test_validation as test_val
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    #contacts_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\contacts_shifted_corrected.csv"
    contacts_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\output_ElectroLoc.csv"
    contacts_path_2 = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\contacts_shifted_corrected.csv"
    ground_truth_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\ground_truth\groundtruth_COG_004.csv"
    #mask_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\mask_generated\33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip.nii.gz_electrode_mask.nii.gz"
    #CT_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\CT\33312121_CT_cerebrum_uden_kontrast_20230501091320_5_strip.nii.gz"
    output_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation\validation_outputElectroLoc.csv"
    entry_point_path_world = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\input_ElectroLoc\input_ElectroLoc_world.csv"
    entry_point_path_voxel = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\input_ElectroLoc\input_ElectroLoc_voxel.csv"

    contact_validation_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation\contacts_validation.csv"
    contact_validation_path2 = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation\contacts_shifted_validation.csv"
    entry_validation_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation\entry_point_validation.csv"
    output_electroLoc = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\output_ElectroLoc.csv"
    output_indexation = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\contacts_reordered.csv"
    output_shift = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\contacts_shifted_corrected.csv"

     # Validation contacts
    #test_val.validate_contacts_with_electrode_mask(
    #    mask_path=mask_path,
    #    contact_path=contacts_path,
    #    output_path=contact_validation_path
    #)

    #test_val.validate_contacts_with_electrode_mask(
      #  mask_path=mask_path,
     #   contact_path=contacts_path_2,
       # output_path=contact_validation_path2
    #)

    # Validation entry points
    """test_val.validate_entry_points(
        contact_path=contacts_path,
        entry_points_path=entry_point_path_world,
        output_path=entry_validation_path,
        mask_path=mask_path
    )"""

    # Reload for plots
    """df_contacts_validation = pd.read_csv(contact_validation_path)
    df_contacts_validation2 = pd.read_csv(contact_validation_path2)"""

    # Summary
    #errors = df_contacts_validation.loc[
    #    df_contacts_validation["distance_to_mask_mm"] != 0,
    #    "distance_to_mask_mm"
    #].dropna()
    #errors = df_contacts_validation2.loc[
    #    df_contacts_validation2["distance_to_mask_mm"] != 0,
    #    "distance_to_mask_mm"
    #].dropna()

    #print("\nValidation summary")
    #print("------------------")
    #print(f"Total contacts: {len(df_contacts_validation)}")
    #print(f"Inside mask: {df_contacts_validation['inside_mask'].sum()}")
    #print(f"Outside mask: {(df_contacts_validation['inside_mask'] == False).sum()}")

    #if len(errors) > 0:
    #    print(f"Median error: {np.median(errors):.3f} mm")
    #    print(f"Mean error: {np.mean(errors):.3f} mm")
    #    print(f"Max error: {np.max(errors):.3f} mm")
    #else:
    #    print("All contacts are inside the mask.")

    # Plots
    """ct_img = nib.load(CT_path)
    ct = ct_img.get_fdata()

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    contacts = pd.read_csv(contacts_path)
    contacts2 = pd.read_csv(contacts_path_2)
    test_val.plot_validation(df_contacts_validation)
    test_val.plot_validation(df_contacts_validation2)"""

    #test_val.plot_3d(ct, mask, contacts, step_ct=6, step_mask=1, show_ct=False, show_labels=False)
    meandistance, df = test_val.compute_mean_intercontact_distance(ground_truth_path)

    print(f"Mean inter-contact distance: {meandistance:.3f} mm")
    threshold = meandistance * 0.75
    print(f"Using distance threshold of {threshold:.3f} mm for validation.")
    print("Validation ElectroLoc:")
    validate_results_outputElectroloc, median1 = test_val.validation_contact_loc_with_ground_truth(ground_truth_path, output_electroLoc, r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation_output_ElectroLoc.csv", threshold=threshold, electrode_col="electrode_id_nb")
    print("Validation Indexation:")
    validate_results_output_indexation, median2 = test_val.validation_contact_loc_with_ground_truth(ground_truth_path, output_indexation, r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation_output_indexation.csv", threshold=threshold, electrode_col="electrode_id")
    print("Validation Shift:")
    validate_results_output_shift, median3 = test_val.validation_contact_loc_with_ground_truth(ground_truth_path, output_shift, r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_004\out\validation_output_shift.csv", threshold=threshold, electrode_col="electrode_id")



    #ground_truth_entry_points_path = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\groundtruth_entry_points_COG011.csv"
    #entry_points_path_world = r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\input_ElectroLoc\input_ElectroLoc_world.csv"
    #test_val.validate_entry_points_with_ground_truth(ground_truth_entry_points_path, entry_points_path_world, r"C:\Users\maely\OneDrive - UCL\MASTER GBIO\TFE\Data\COG_011\out\validation_entry_points.csv", threshold=threshold, coord_cols=("world_x", "world_y", "world_z"))

