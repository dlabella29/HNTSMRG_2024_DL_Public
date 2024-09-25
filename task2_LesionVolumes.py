import os
import nibabel as nib
import numpy as np
import scipy.ndimage
import csv

# Paths
master_folder = '/home/curasight/HNTSMRG24_train/'
output_folder = '/home/curasight/Documents'

# Output CSV files
midRT_csv = os.path.join(output_folder, 'midRT_lesion_volumes.csv')
preRT_csv = os.path.join(output_folder, 'preRT_registered_lesion_volumes.csv')

# Initialize lists to store data
midRT_data = []
preRT_data = []

# Traverse all subfolders
for root, dirs, files in os.walk(master_folder):
    for file in files:
        if file.endswith('_midRT_mask.nii.gz') or file.endswith('_preRT_mask_registered.nii.gz'):
            # Determine the scan type
            if file.endswith('_midRT_mask.nii.gz'):
                scan_type = 'midRT'
            elif file.endswith('_preRT_mask_registered.nii.gz'):
                scan_type = 'preRT_registered'
            # Get full path of the mask file
            mask_path = os.path.join(root, file)
            # Load the mask image
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            # Get voxel dimensions
            voxel_dims = mask_img.header.get_zooms()
            voxel_volume = np.prod(voxel_dims)  # in mm^3
            # Extract case number from filename
            case_number = file.split('_')[0]
            # Process tumor lesions (mask value == 1)
            tumor_mask = (mask_data == 1)
            if np.any(tumor_mask):
                structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
                labeled_mask, num_features = scipy.ndimage.label(tumor_mask, structure=structure)
                for lesion_num in range(1, num_features + 1):
                    lesion_voxels = np.sum(labeled_mask == lesion_num)
                    lesion_volume = lesion_voxels * voxel_volume
                    data_row = [case_number, scan_type, 'Tumor', lesion_num, lesion_volume]
                    # Print lesion info
                    print(f"Processing {scan_type} lesion: Case {case_number}, Tumor, Lesion {lesion_num}, Volume {lesion_volume} mm³")
                    if scan_type == 'midRT':
                        midRT_data.append(data_row)
                    elif scan_type == 'preRT_registered':
                        preRT_data.append(data_row)
            # Process node lesions (mask value == 2)
            node_mask = (mask_data == 2)
            if np.any(node_mask):
                structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
                labeled_mask, num_features = scipy.ndimage.label(node_mask, structure=structure)
                for lesion_num in range(1, num_features + 1):
                    lesion_voxels = np.sum(labeled_mask == lesion_num)
                    lesion_volume = lesion_voxels * voxel_volume
                    data_row = [case_number, scan_type, 'Node', lesion_num, lesion_volume]
                    # Print lesion info
                    print(f"Processing {scan_type} lesion: Case {case_number}, Node, Lesion {lesion_num}, Volume {lesion_volume} mm³")
                    if scan_type == 'midRT':
                        midRT_data.append(data_row)
                    elif scan_type == 'preRT_registered':
                        preRT_data.append(data_row)

# Write midRT data to CSV
with open(midRT_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write header
    writer.writerow(['Case_Number', 'Scan_Type', 'Lesion_Type', 'Lesion_Number', 'Lesion_Volume_mm3'])
    # Write data rows
    writer.writerows(midRT_data)

# Write preRT data to CSV
with open(preRT_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write header
    writer.writerow(['Case_Number', 'Scan_Type', 'Lesion_Type', 'Lesion_Number', 'Lesion_Volume_mm3'])
    # Write data rows
    writer.writerows(preRT_data)

print('Lesion volumes calculated and saved to CSV files.')
