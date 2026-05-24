import nibabel as nib
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os 
from regis.core import find_transform, apply_transform
import pandas as pd
from scipy.ndimage import label
from sklearn.decomposition import PCA

def load_nii(path_nii):
    """Load and visualize NIfTI file.
    Args:
        path_nii (str): Path to the NIfTI file.
        output_dir (str): Directory to save output files.
    Returns:
        Visualize 3D volume and electrodes mask.
    """
    img = nib.load(path_nii)
    data_nii = img.get_fdata()
    #print(f"header : {img.header}")  # Print the header information
    #print(f"Shape volume : {data_nii.shape}")  # Print the shape of the data 
    #print(f"Max/min value : {np.max(data_nii), np.min(data_nii)}")  # Print the data type

    grid = pv.ImageData()
    grid.dimensions = np.array(data_nii.shape) + 1
    grid.cell_data['values'] = data_nii.flatten(order='F')
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap='gray', opacity=[0.0, 0.045], show_scalar_bar=True) # Comment this line to not show the full volume
    plotter.show()


def coregistration_CT_to_MRI(path_ct, path_mri,output_dir):
    """Coregister CT to MRI 
    Args:
        path_ct (str): Path to the CT NIfTI file.
        path_mri (str): Path to the MRI NIfTI file.
        output_dir (str): Directory to save the coregistered CT file.
    Returns:
        str: Path to the coregistered CT NIfTI file.
    """
    mapping = find_transform(path_ct, path_mri, level_iters = [1000, 100, 10], diffeomorph=False)
    output_path_coregistration = os.path.join(output_dir, os.path.basename(path_ct)[:-7] + "_coregistered.nii.gz")
    apply_transform(path_ct, mapping, static_file=path_mri, output_path=output_path_coregistration)
    
    output_path_mapping = os.path.join(output_dir, os.path.basename(path_ct)[:-7] + "_mapping.npy")
    np.save(output_path_mapping, mapping)
    print(f"Coregistered CT saved at: {output_path_coregistration}")
    print(f"Mapping saved at : {output_path_mapping}")
    return output_path_coregistration, output_path_mapping

def electrode_mask(path_nii, output_dir,threshold=2200):
    """Create a mask to visualize deep electrodes.
    Args:
        data_nii (numpy array): The NIfTI data array.
        plotter (pyvista.Plotter): The PyVista plotter object.
        threshold (int): The threshold value to create the mask.
    """
    img = nib.load(path_nii)
    data_nii = img.get_fdata()


    mask_array = np.where(data_nii > threshold, 1, 0).astype(np.uint8)

    affine = img.affine
    header = img.header.copy()
    mask_img = nib.Nifti1Image(mask_array, affine, header)
    output_path = os.path.join(output_dir, os.path.basename(path_nii) + "_electrode_mask.nii.gz")
    nib.save(mask_img, output_path)
    print(f"Electrodes mask saved at: {output_path}")
    return(output_path)

def extract_surface_mask(brain_mask_path, output_dir):
    """Extract the surface of the brain from the brain mask. 
    Args:
        brain_mask_path (str): Path to the brain mask NIfTI file.
        output_dir (str): Directory to save the surface mask file.
    Returns:
        str: Path to the surface mask NIfTI file.
    """
    img = nib.load(brain_mask_path)
    data = img.get_fdata()
    brain = data > 0
    surface_brain = np.zeros_like(data)
    nx, ny, nz = data.shape
    for x in range(1, nx-1):
        for y in range(1, ny-1):
            for z in range(1, nz-1):
                if brain[x, y, z] :
                    neighbors = [
                        brain[x+1, y, z],
                        brain[x-1, y, z],
                        brain[x, y+1, z],
                        brain[x, y-1, z],
                        brain[x, y, z+1],
                        brain[x, y, z-1],
                    ]

                    if not all(neighbors):
                        surface_brain[x, y, z] = 1
    out_img = nib.Nifti1Image(surface_brain, img.affine, img.header)
    nib.save(out_img, os.path.join(output_dir, os.path.basename(brain_mask_path)[:-7] + "_surface_mask.nii.gz"))
    print(f"Surface mask saved at {os.path.join(output_dir, os.path.basename(brain_mask_path)[:-7] + '_surface_mask.nii.gz')}")
    return os.path.join(output_dir, os.path.basename(brain_mask_path)[:-7] + "_surface_mask.nii.gz")

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation


def dilate_surface_mask(surface_mask_path, output_dir, dilation_vox=2):
    """
    Dilate an existing surface mask to make the surface line thicker.

    Args:
        surface_mask_path (str): Path to the surface mask NIfTI file.
        output_dir (str): Directory where the dilated surface mask is saved.
        dilation_vox (int): Number of voxels used to thicken the surface.

    Returns:
        str: Path to the dilated surface mask.
    """

    os.makedirs(output_dir, exist_ok=True)

    img = nib.load(surface_mask_path)
    data = img.get_fdata()

    surface = data > 0

    structure = np.ones((3, 3, 3), dtype=bool)

    surface_dilated = binary_dilation(
        surface,
        structure=structure,
        iterations=dilation_vox
    )

    surface_dilated = surface_dilated.astype(np.uint8)

    out_img = nib.Nifti1Image(surface_dilated, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)

    nib.save(out_img, os.path.join(output_dir, os.path.basename(surface_mask_path)[:-7] + "_dilated.nii.gz"))

    print(f"Dilated surface mask saved at: {os.path.join(output_dir, os.path.basename(surface_mask_path)[:-7] + '_dilated.nii.gz')}")

    return os.path.join(output_dir, os.path.basename(surface_mask_path)[:-7] + '_dilated.nii.gz')

def visualize_mask(mask_path, ct_path):
    """
    Visualize CT and its binary mask in 3D.

    Args:
        mask_path (str): Path to mask .nii.gz
        ct_path (str): Path to CT .nii.gz
    """

    # 🔹 Load files
    mask_nii = nib.load(mask_path)
    ct_nii = nib.load(ct_path)

    mask = mask_nii.get_fdata()
    ct_data = ct_nii.get_fdata()

    # 🔹 Safety check
    if mask.shape != ct_data.shape:
        raise ValueError(f"Mask shape {mask.shape} != CT shape {ct_data.shape}")

    spacing = ct_nii.header.get_zooms()[:3]

    # --- CT grid ---
    ct_grid = pv.ImageData()
    ct_grid.dimensions = np.array(ct_data.shape) + 1
    ct_grid.spacing = spacing
    ct_grid.cell_data["values"] = ct_data.flatten(order="F")

    # --- Mask grid ---
    mask_grid = pv.ImageData()
    mask_grid.dimensions = np.array(mask.shape) + 1
    mask_grid.spacing = spacing
    mask_grid.cell_data["values"] = mask.flatten(order="F")

    # 🔹 Plot
    plotter = pv.Plotter()

    # CT en gris
    """plotter.add_volume(
        ct_grid,
        cmap="gray",
        opacity=[0.0, 0.1, 0.3, 0.6],
        show_scalar_bar=False
    )"""

    # Mask en rouge
    plotter.add_volume(
        mask_grid,
        cmap="Reds",
        opacity=[0.0, 0.8],
        show_scalar_bar=False
    )

    plotter.add_axes()
    plotter.show()

def create_contact_mask_from_csv(
    contact_shifted_corrected_path,
    ct_path,
    output_dir,
    radius_vox=1
):
    """
    Create a NIfTI mask from shifted contact voxel coordinates.

    The output mask has:
    - same shape as the CT
    - same affine as the CT
    - same space as the CT

    Parameters
    ----------
    contact_shifted_corrected_path : str
        CSV containing shifted contacts with vox_x, vox_y, vox_z.

    ct_path : str
        Reference CT NIfTI path.

    output_path : str
        Output NIfTI mask path.

    radius_vox : int
        Radius around each contact in voxel units.
        radius_vox=0 gives one voxel per contact.
        radius_vox=1 gives a small 3D blob around each contact.
    """

    # Load CT as reference
    ct_img = nib.load(ct_path)
    ct_shape = ct_img.shape
    ct_affine = ct_img.affine
    ct_header = ct_img.header.copy()

    # Load contacts
    df = pd.read_csv(contact_shifted_corrected_path)

    # Empty mask with same size as CT
    mask = np.zeros(ct_shape, dtype=np.uint8)

    for _, row in df.iterrows():
        x = int(round(row["vox_x"]))
        y = int(round(row["vox_y"]))
        z = int(round(row["vox_z"]))

        for dx in range(-radius_vox, radius_vox + 1):
            for dy in range(-radius_vox, radius_vox + 1):
                for dz in range(-radius_vox, radius_vox + 1):

                    xx = x + dx
                    yy = y + dy
                    zz = z + dz

                    if (
                        0 <= xx < ct_shape[0]
                        and 0 <= yy < ct_shape[1]
                        and 0 <= zz < ct_shape[2]
                    ):
                        mask[xx, yy, zz] = 1

    # Save as NIfTI in CT space
    out_img = nib.Nifti1Image(mask, affine=ct_affine, header=ct_header)
    out_img.set_data_dtype(np.uint8)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    nib.save(out_img, os.path.join(output_dir, os.path.basename(contact_shifted_corrected_path)[:-4] + ".nii.gz"))

    print(f"Saved contact mask to: { os.path.join(output_dir, os.path.basename(contact_shifted_corrected_path)[:-7] + '.nii.gz')}")

    return  os.path.join(output_dir, os.path.basename(contact_shifted_corrected_path)[:-4] + '.nii.gz')


