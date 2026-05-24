import os 
from regis.core import find_transform, apply_transform
import nibabel as nib 
import numpy as np
from scipy.ndimage import distance_transform_edt

def load_atlas(path_atlas):
    """Load an atlas NIfTI file and return the data as a NumPy array.
    Args:
        path_atlas (str): The path to the atlas NIfTI file.
    Returns:
        numpy.ndarray: The data from the atlas NIfTI file.
    """
    img = nib.load(path_atlas)
    data_atlas = img.get_fdata()
    return data_atlas

def coregister_1_to_2(path_1, path_2, output_dir):
    """Coregister first file to the second file using FSL's FLIRT tool.
    Args:
        path_1 (str): Path to the first NIfTI file.
        path_2 (str): Path to the second NIfTI file.
        output_dir (str): Directory to save the coregistered file.
    Returns:
        str: Path to the coregistered file.
    """
    mapping = find_transform(path_1, path_2, level_iters = [1000, 100, 10], diffeomorph=False,sanity_check=True,
                             normalize=True)
    output_path_coregistration = os.path.join(output_dir, os.path.basename(path_1)[:-7] + "_coregistered_to_" + os.path.basename(path_2)[:-7] + ".nii.gz")
    apply_transform(path_1, mapping, static_file=path_2, output_path=output_path_coregistration, labels = True)
    output_path_mapping = os.path.join(output_dir, os.path.basename(path_1)[:-7] + "_mapping.npy")
    np.save(output_path_mapping, mapping)
    print(f"Coregistered file saved at: {output_path_coregistration}")
    print(f"Mapping saved at : {output_path_mapping}")
    return output_path_coregistration, mapping




def dilate_atlas_labels(atlas_path, brain_mask_path, dilation_width, output_dir):
    """
    Dilates cortical atlas labels to include nearby unlabeled voxels (0),
    constrained by a brain mask.

    Parameters
    ----------
    atlas_path : str
        Path to atlas NIfTI file.
    brain_mask_path : str
        Path to brain mask NIfTI file.
    dilation_width : float
        Maximum dilation distance in voxels.
    output_dir : str
        Path where the output NIfTI will be saved.

    Returns
    -------
    dilated_atlas : np.ndarray
        Atlas with dilated labels.
    output_path : str
        Path to the saved output file.
    """
    atlas_img = nib.load(atlas_path)
    atlas = atlas_img.get_fdata()

    mask_img = nib.load(brain_mask_path)
    brain_mask = mask_img.get_fdata()

    # Mask unlabeled voxels inside the brain
    unlabeled = np.where(atlas == 0, 1, 0)
    unlabeled *= brain_mask.astype("int32")

    # Distance transform + nearest labeled voxel indices
    distances, nearest_idx = distance_transform_edt(
        unlabeled, return_indices=True
    )

    # Copy atlas to output
    dilated_atlas = atlas.copy()

    # For unlabeled voxels within dilation_width, assign nearest label
    within_dilation = (unlabeled == 1) & (distances <= dilation_width)

    coords = np.argwhere(within_dilation)
    for x, y, z in coords:
        nx, ny, nz = nearest_idx[:, x, y, z]
        dilated_atlas[x, y, z] = atlas[nx, ny, nz]

    out = nib.Nifti1Image(dilated_atlas, atlas_img.affine, atlas_img.header)

    

    output_path = os.path.join(output_dir, "dilated_" + os.path.basename(atlas_path))
    out.to_filename(output_path)

    return output_path