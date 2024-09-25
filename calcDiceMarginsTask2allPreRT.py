import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion


def dice_coefficient(y_true, y_pred, label):
    """
    Calculate the Dice coefficient for a specific label.
    """
    y_true_label = (y_true == label)
    y_pred_label = (y_pred == label)
    intersection = np.sum(y_true_label & y_pred_label)
    size_y_true = np.sum(y_true_label)
    size_y_pred = np.sum(y_pred_label)
    if size_y_true + size_y_pred == 0:
        return 1.0  # Both segmentations are empty for this label
    dice = 2.0 * intersection / (size_y_true + size_y_pred)
    return dice


def get_structuring_element(voxel_sizes, radius_mm):
    """
    Create a spherical structuring element with given radius in mm.
    """
    radius_voxels = [radius_mm / vs for vs in voxel_sizes]
    # Create a grid of coordinates
    ranges = [np.arange(-np.ceil(r), np.ceil(r) + 1) for r in radius_voxels]
    X, Y, Z = np.meshgrid(*ranges, indexing='ij')
    distances = np.sqrt(
        (X * voxel_sizes[0]) ** 2 +
        (Y * voxel_sizes[1]) ** 2 +
        (Z * voxel_sizes[2]) ** 2
    )
    structuring_element = distances <= radius_mm
    return structuring_element


def erode_mask(mask_data, voxel_sizes, radius_mm):
    """
    Erodes the mask data by a spherical structuring element of given radius in mm.
    """
    structuring_element = get_structuring_element(voxel_sizes, radius_mm)
    eroded_mask = np.zeros_like(mask_data)
    for label in [1, 2]:
        mask_label = (mask_data == label)
        if np.any(mask_label):
            eroded_mask_label = binary_erosion(mask_label, structure=structuring_element)
            eroded_mask[eroded_mask_label] = label
    return eroded_mask


def main():
    base_dir = '/home/curasight/HNTSMRG24_train/'
    margins_mm = [1, 2, 3, 4, 5]

    # Initialize dictionaries to store results
    margins_results = {margin: [] for margin in margins_mm}
    pre_mid_results = []

    # Define output directories
    output_dir_margins = '/home/curasight/Documents/dice_scores_midRT_vs_preRTminus_mm/'
    output_dir_pre_mid = '/home/curasight/Documents/dice_scores_preRT_vs_midRT/'

    # Create output directories if they don't exist
    os.makedirs(output_dir_margins, exist_ok=True)
    os.makedirs(output_dir_pre_mid, exist_ok=True)

    # Get a list of all subject directories
    subjects = sorted(glob.glob(os.path.join(base_dir, '*')))
    for subject_path in subjects:
        if not os.path.isdir(subject_path):
            continue  # Skip if not a directory
        subject_id = os.path.basename(subject_path)
        midRT_dir = os.path.join(subject_path, 'midRT')

        if not os.path.exists(midRT_dir):
            print(f'MidRT directory not found for subject {subject_id}')
            continue

        # Find the preRT mask registered to midRT
        preRT_mask_registered_files = glob.glob(os.path.join(midRT_dir, '*_preRT_mask_registered.nii.gz'))
        if not preRT_mask_registered_files:
            print(f'preRT mask registered files not found for subject {subject_id}')
            continue
        preRT_mask_registered_path = preRT_mask_registered_files[0]

        # Find the midRT mask
        midRT_mask_files = glob.glob(os.path.join(midRT_dir, '*_midRT_mask.nii.gz'))
        if not midRT_mask_files:
            print(f'MidRT mask file not found for subject {subject_id}')
            continue
        midRT_mask_path = midRT_mask_files[0]

        # Load the preRT mask
        preRT_img = nib.load(preRT_mask_registered_path)
        preRT_mask_data = preRT_img.get_fdata().astype(np.int32)
        voxel_sizes = preRT_img.header.get_zooms()

        # Load the midRT mask
        midRT_img = nib.load(midRT_mask_path)
        midRT_mask_data = midRT_img.get_fdata().astype(np.int32)

        # Compute Dice coefficients between original preRT and midRT masks
        dice_label1_pre_mid = dice_coefficient(preRT_mask_data, midRT_mask_data, label=1)
        dice_label2_pre_mid = dice_coefficient(preRT_mask_data, midRT_mask_data, label=2)

        print(
            f"Subject: {subject_id} | PreRT vs MidRT | Dice Label1: {dice_label1_pre_mid:.4f}, Dice Label2: {dice_label2_pre_mid:.4f}")

        # Store the preRT vs midRT results
        pre_mid_results.append({
            'SubjectID': subject_id,
            'Dice_Label1': dice_label1_pre_mid,
            'Dice_Label2': dice_label2_pre_mid
        })

        # For each margin, erode preRT mask and compute Dice with midRT
        for margin in margins_mm:
            eroded_mask_data = erode_mask(preRT_mask_data, voxel_sizes, radius_mm=margin)

            # Compute Dice coefficients between eroded preRT mask and midRT mask
            dice_label1_margin = dice_coefficient(eroded_mask_data, midRT_mask_data, label=1)
            dice_label2_margin = dice_coefficient(eroded_mask_data, midRT_mask_data, label=2)

            print(
                f"Subject: {subject_id} | PreRT_minus{margin}mm vs MidRT | Dice Label1: {dice_label1_margin:.4f}, Dice Label2: {dice_label2_margin:.4f}")

            # Store the margin-specific results
            margins_results[margin].append({
                'SubjectID': subject_id,
                'Dice_Label1': dice_label1_margin,
                'Dice_Label2': dice_label2_margin
            })

            # Optionally, save the eroded mask as a NIfTI file
            # Uncomment the following lines if you wish to save the eroded masks
            '''
            eroded_mask_img = nib.Nifti1Image(eroded_mask_data, affine=preRT_img.affine, header=preRT_img.header)
            eroded_mask_filename = f"{os.path.splitext(os.path.splitext(os.path.basename(preRT_mask_registered_path))[0])[0]}_minus{margin}mm.nii.gz"
            eroded_mask_path = os.path.join(midRT_dir, eroded_mask_filename)
            nib.save(eroded_mask_img, eroded_mask_path)
            '''

    # Save the preRT vs midRT results to a single CSV file
    pre_mid_csv_path = os.path.join(output_dir_pre_mid, 'dice_scores_preRT_vs_midRT.csv')
    df_pre_mid = pd.DataFrame(pre_mid_results)
    df_pre_mid.to_csv(pre_mid_csv_path, index=False)
    print(f'Dice scores for PreRT vs MidRT saved to {pre_mid_csv_path}')

    # Save the midRT vs preRTminus{1-5}mm results to separate CSV files
    for margin in margins_mm:
        csv_path = os.path.join(output_dir_margins, f'dice_scores_midRT_vs_preRTminus_{margin}mm.csv')
        df_margin = pd.DataFrame(margins_results[margin])
        df_margin.to_csv(csv_path, index=False)
        print(f'Dice scores for MidRT vs PreRT_minus{margin}mm saved to {csv_path}')


if __name__ == '__main__':
    main()
