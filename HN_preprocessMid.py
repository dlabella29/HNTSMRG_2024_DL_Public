import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

def expand_mask(mask, expansion_mm, voxel_spacing):
    """
    Expand mask values of 1 (tumor) and 2 (node) by the given expansion in mm.
    """
    # Calculate expansion in voxels for each axis
    expansion_voxels = [int(expansion_mm / spacing) for spacing in voxel_spacing]

    # Create binary masks for tumor and node
    tumor_mask = mask == 1
    node_mask = mask == 2

    # Expand the tumor and node masks
    expanded_tumor_mask = binary_dilation(tumor_mask, iterations=expansion_voxels[0])
    expanded_node_mask = binary_dilation(node_mask, iterations=expansion_voxels[0])

    # Combine expanded masks back into the original mask
    expanded_mask = np.zeros_like(mask)
    expanded_mask[expanded_tumor_mask] = 1
    expanded_mask[expanded_node_mask] = 2

    return expanded_mask

def compute_bounding_box(mask, margin_mm, voxel_spacing):
    """
    Compute the bounding box of the mask plus a margin.
    """
    indices = np.where(mask > 0)
    min_coords = np.min(indices, axis=1)
    max_coords = np.max(indices, axis=1)

    # Convert margin to voxel space
    margin_voxels = [int(margin_mm / spacing) for spacing in voxel_spacing]

    # Add margin
    min_coords = np.maximum(0, min_coords - margin_voxels)
    max_coords = np.minimum(mask.shape, max_coords + margin_voxels)

    return tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))

def process_case(case_path, output_path):
    """
    Process each MRI case to modify images and masks.
    """
    # Load images and masks
    preRT_image_path = os.path.join(case_path,"midRT", case_path.split('/')[-1] + "_preRT_T2_registered.nii.gz")
    midRT_image_path = os.path.join(case_path,"midRT", case_path.split('/')[-1] + "_midRT_T2.nii.gz")
    preRT_mask_path = os.path.join(case_path, "midRT",case_path.split('/')[-1] + "_preRT_mask_registered.nii.gz")
    midRT_mask_path = os.path.join(case_path, "midRT",case_path.split('/')[-1] + "_midRT_mask.nii.gz")

    preRT_img = nib.load(preRT_image_path)
    midRT_img = nib.load(midRT_image_path)
    preRT_mask_img = nib.load(preRT_mask_path)
    midRT_mask_img = nib.load(midRT_mask_path)

    preRT_data = preRT_img.get_fdata()
    midRT_data = midRT_img.get_fdata()
    preRT_mask = preRT_mask_img.get_fdata()
    midRT_mask = midRT_mask_img.get_fdata()

    voxel_spacing = preRT_img.header.get_zooms()

    # Expand the preRT mask
    expanded_preRT_mask = expand_mask(preRT_mask, 50, voxel_spacing)

    # Make pixel intensities outside of the expanded masks equal to 0
    preRT_data[expanded_preRT_mask == 0] = 0
    midRT_data[expanded_preRT_mask == 0] = 0

    # Compute bounding box
    bounding_box = compute_bounding_box(expanded_preRT_mask, 0, voxel_spacing)

    # Crop images and masks to the bounding box
    #cropped_preRT_data = preRT_data[bounding_box]
    cropped_midRT_data = midRT_data[bounding_box]
    cropped_preRT_mask = preRT_mask[bounding_box]
    cropped_midRT_mask = midRT_mask[bounding_box]

    # Save the modified files
    #nib.save(nib.Nifti1Image(cropped_preRT_data, preRT_img.affine), os.path.join(output_path, f'modified2_{case_path.split("/")[-1]}_preRT_T2_registered.nii.gz'))
    nib.save(nib.Nifti1Image(cropped_midRT_data, midRT_img.affine), os.path.join(output_path, f'modified2_{case_path.split("/")[-1]}_midRT_T2.nii.gz'))
    nib.save(nib.Nifti1Image(cropped_preRT_mask, preRT_mask_img.affine), os.path.join(output_path, f'modified2_{case_path.split("/")[-1]}_preRT_mask_registered.nii.gz'))
    nib.save(nib.Nifti1Image(cropped_midRT_mask, midRT_mask_img.affine), os.path.join(output_path, f'modified2_{case_path.split("/")[-1]}_midRT_mask_registered.nii.gz'))

def main():
    # Set input and output directories
    input_dir = '/home/curasight/HNTSMRG24_train/'
    output_dir = ('/home/curasight/modified2_HNTSMRG24_Train/')

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all cases in the input directory
    for case_folder in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_folder)
        if os.path.isdir(case_path):
            process_case(case_path, output_dir)

if __name__ == "__main__":
    main()
