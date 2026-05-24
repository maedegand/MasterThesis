import nibabel as nib
import numpy as np
import os
import pyvista as pv
from scipy.ndimage import label, center_of_mass
import pandas as pd

def entry_point_coordinates(surface_mask_path, electrode_mask_path, output_dir):
    """ Find the voxel coordinates of the entry points of the electrodes
    Args:
        brain_mask_path (str): Path to the brain mask NIfTI file.
        electrode_mask_path (str): Path to the electrode mask NIfTI file.
    Returns:
        list of tuples: A list of tuples containing the voxel coordinates of the entry points (nb_contacts, vox_x, vox_y, vox_z).
            nb_contacts = number of contacts per electrodes.
            vox_x, vox_y, vox_z = coordinates of the electrode's entry point, expressed in voxel coordinates. 
        Nifti file: A NIfTI file containing the entry points of the electrodes.
    """
    
    surface_img = nib.load(surface_mask_path)
    electrode_img = nib.load(electrode_mask_path)
    surface_data = surface_img.get_fdata()
    electrode_data = electrode_img.get_fdata()
    
    intersection = np.logical_and(surface_data, electrode_data)
    entry_points = np.argwhere(intersection)
    
    # ==============
    # Groups neighboring voxels into clusters (one cluster per electrode)
    # ==============
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_array, num_groups = label(intersection, structure=structure)
    print(f"Number of electrode entry points found: {num_groups}")

    # ==============
    # Find the center of mass for each cluster to get a single coordinate entry point per electrode
    # ==============
    centroids = center_of_mass(intersection, labeled_array, range(1, num_groups + 1))

    entry_points = [(
        int(centroid[0]),
        int(centroid[1]),
        int(centroid[2])
    ) for centroid in centroids
    ]
    # ==============
    # Convert voxel -> world space
    # ==============
    affine = surface_img.affine
    entry_point_world = []
    for point in entry_points:
        world_coords = nib.affines.apply_affine(affine, point)
        entry_point_world.append((world_coords[0], world_coords[1], world_coords[2]))

    # ==============
    # Save to CSV file (voxel space)
    # ==============
    output_path_voxel = os.path.join(output_dir, "entry_points_voxel.csv")
    with open(output_path_voxel, "w") as f:
        f.write("vox_x,vox_y,vox_z\n")
        for point in entry_points:
            f.write(f"{point[0]},{point[1]},{point[2]}\n")
    print(f"Entry points in voxel spacesaved at {output_path_voxel}")

    # ==============
    # Save to CSV file (world space)
    # ==============
    output_path_world = os.path.join(output_dir, "entry_points_world.csv")
    with open(output_path_world, "w") as f:
        f.write("world_x,world_y,world_z\n")
        for point in entry_point_world:
            f.write(f"{point[0]},{point[1]},{point[2]}\n")
    print(f"Entry points saved in world space at {output_path_world}")

    return output_path_voxel, output_path_world 

def get_valid_csv_path(prompt, required_columns, name="file"):
    
    while True:
        path = input(prompt).strip()
        path = path.strip('"').strip("'")

        if not os.path.exists(path):
            print(f"❌ {name} not found. Try again.\n")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f" Error reading {name}: {e}\n")
            continue

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            print(f" Missing columns in {name}: {missing}")
            print(f"Available columns: {list(df.columns)}\n")
            continue

        print(f"✔ {name} OK\n")
        return path, df
    
