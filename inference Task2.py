import os
import sys
import logging
import json
import subprocess
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, label
import torch
import fire
import yaml
from scipy.stats import mode
import shutil  # Added for file operations

# Ensure that the module paths are correct
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
basepath = "/"
if __package__ in (None, ""):
    from segmenter import Segmenter, dist_launched, run_segmenter
else:
    from .segmenter import Segmenter, dist_launched, run_segmenter

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
    max_coords = np.minimum(np.array(mask.shape), max_coords + margin_voxels)

    return tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords)), min_coords, max_coords

def remove_small_lesions(mask_array, min_size_tumor=200, min_size_node=60):
    """
    Remove small lesions from the mask array.
    - Remove tumor lesions smaller than min_size_tumor voxels.
    - Remove node lesions smaller than min_size_node voxels.
    """
    # Remove small tumor lesions
    tumor_mask = (mask_array == 1)
    labeled_tumor, num_features_tumor = label(tumor_mask)
    for i in range(1, num_features_tumor + 1):
        lesion_voxels = np.sum(labeled_tumor == i)
        if lesion_voxels < min_size_tumor:
            mask_array[labeled_tumor == i] = 0  # Set to background

    # Remove small node lesions
    node_mask = (mask_array == 2)
    labeled_node, num_features_node = label(node_mask)
    for i in range(1, num_features_node + 1):
        lesion_voxels = np.sum(labeled_node == i)
        if lesion_voxels < min_size_node:
            mask_array[labeled_node == i] = 0  # Set to background

    return mask_array

def process_cases(preRT_image_dir, midRT_image_dir, preRT_mask_dir):
    """
    Process all MRI cases to modify images and masks.
    """

    preRT_image_files = [f for f in os.listdir(preRT_image_dir) if f.endswith('.mha')]
    midRT_image_files = [f for f in os.listdir(midRT_image_dir) if f.endswith('.mha')]
    preRT_mask_files = [f for f in os.listdir(preRT_mask_dir) if f.endswith('.mha')]

    # Sort the lists to ensure corresponding files match
    preRT_image_files.sort()
    midRT_image_files.sort()
    preRT_mask_files.sort()

    for preRT_image_file, midRT_image_file, preRT_mask_file in zip(preRT_image_files, midRT_image_files, preRT_mask_files):
        preRT_image_path = os.path.join(preRT_image_dir, preRT_image_file)
        midRT_image_path = os.path.join(midRT_image_dir, midRT_image_file)
        preRT_mask_path = os.path.join(preRT_mask_dir, preRT_mask_file)

        # Load images and masks using SimpleITK
        preRT_img = sitk.ReadImage(preRT_image_path)
        midRT_img = sitk.ReadImage(midRT_image_path)
        preRT_mask_img = sitk.ReadImage(preRT_mask_path)

        preRT_data = sitk.GetArrayFromImage(preRT_img)
        midRT_data = sitk.GetArrayFromImage(midRT_img)
        preRT_mask = sitk.GetArrayFromImage(preRT_mask_img)

        # Get voxel spacing and reverse to match numpy array ordering (z, y, x)
        voxel_spacing = preRT_img.GetSpacing()[::-1]

        # Expand the preRT mask by 10mm for tumor and node
        expanded_preRT_mask_10mm = expand_mask(preRT_mask, 10, voxel_spacing)

        # Create the directory if it doesn't exist
        preRT_expansion_dir = os.path.join(basepath, 'tmp/preRT_expansion')
        os.makedirs(preRT_expansion_dir, exist_ok=True)

        # Save the expanded preRT mask
        preRT_expansion_filename = os.path.basename(midRT_image_file)
        preRT_expansion_path = os.path.join(preRT_expansion_dir, preRT_expansion_filename)
        expanded_preRT_mask_img_10mm = sitk.GetImageFromArray(expanded_preRT_mask_10mm)
        expanded_preRT_mask_img_10mm.SetOrigin(preRT_mask_img.GetOrigin())
        expanded_preRT_mask_img_10mm.SetSpacing(preRT_mask_img.GetSpacing())
        expanded_preRT_mask_img_10mm.SetDirection(preRT_mask_img.GetDirection())
        sitk.WriteImage(expanded_preRT_mask_img_10mm, preRT_expansion_path)

        # Expand the preRT mask by 35mm for processing
        expanded_preRT_mask = expand_mask(preRT_mask, 35, voxel_spacing)

        # Make pixel intensities outside of the expanded masks equal to 0
        preRT_data[expanded_preRT_mask == 0] = 0
        midRT_data[expanded_preRT_mask == 0] = 0

        # Compute bounding box
        bounding_box, min_coords, max_coords = compute_bounding_box(expanded_preRT_mask, 0, voxel_spacing)

        # Crop images and masks to the bounding box
        cropped_preRT_data = preRT_data[bounding_box]
        cropped_midRT_data = midRT_data[bounding_box]
        cropped_preRT_mask = preRT_mask[bounding_box]

        # Save the bounding box coordinates for later use
        bbox_info = {
            'min_coords': [int(c) for c in min_coords.tolist()],
            'max_coords': [int(c) for c in max_coords.tolist()],
            'original_shape': [int(s) for s in preRT_data.shape]
        }

        # Convert numpy arrays back to SimpleITK images
        cropped_preRT_img = sitk.GetImageFromArray(cropped_preRT_data)
        cropped_midRT_img = sitk.GetImageFromArray(cropped_midRT_data)
        cropped_preRT_mask_img = sitk.GetImageFromArray(cropped_preRT_mask)

        # Set the origin, spacing, and direction from the original images
        # Adjust the origin based on the cropping
        origin = preRT_img.GetOrigin()
        spacing = preRT_img.GetSpacing()
        direction = preRT_img.GetDirection()

        # Compute new origin
        start_indices = [s.start for s in bounding_box]
        new_origin = [origin[i] + spacing[i] * start_indices[::-1][i] for i in range(3)]

        # Set new origin, spacing, and direction
        for img in [cropped_preRT_img, cropped_midRT_img, cropped_preRT_mask_img]:
            img.SetOrigin(new_origin)
            img.SetSpacing(spacing)
            img.SetDirection(direction)

        # Determine output paths by replacing 'input' with 'tmp' in the input paths
        preRT_output_path = preRT_image_path.replace('/input/', '/tmp/')
        midRT_output_path = midRT_image_path.replace('/input/', '/tmp/')
        preRT_mask_output_path = preRT_mask_path.replace('/input/', '/tmp/')

        # Ensure the output directories exist
        os.makedirs(os.path.dirname(preRT_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(midRT_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(preRT_mask_output_path), exist_ok=True)

        # Save the modified files
        sitk.WriteImage(cropped_preRT_img, preRT_output_path)
        sitk.WriteImage(cropped_midRT_img, midRT_output_path)
        sitk.WriteImage(cropped_preRT_mask_img, preRT_mask_output_path)

        # Save the bounding box info with a unique name
        base_filename = os.path.splitext(midRT_image_file)[0]
        bbox_info_filename = f"{base_filename}_bbox_info.json"
        bbox_info_path = os.path.join(os.path.dirname(midRT_output_path), bbox_info_filename)
        with open(bbox_info_path, 'w') as f:
            json.dump(bbox_info, f)

def create_json_file(midrt_dir, prertr_dir, output_json_file):
    """
    Create a JSON file listing pairs of images for testing.
    """
    midrt_images = sorted([f for f in os.listdir(midrt_dir) if f.endswith('.mha')])
    prertr_images = sorted([f for f in os.listdir(prertr_dir) if f.endswith('.mha')])

    # Assuming that the filenames correspond directly
    if len(midrt_images) != len(prertr_images):
        raise ValueError("The number of mid-RT and pre-RT images do not match.")

    testing_list = []
    for midrt_img, prertr_img in zip(midrt_images, prertr_images):
        testing_list.append({
            "image": [
                os.path.join(midrt_dir, midrt_img),
                os.path.join(prertr_dir, prertr_img)
            ]
        })

    json_data = {
        "testing": testing_list
    }

    with open(output_json_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"JSON file has been created at {output_json_file}.")

def run_infer_script():
    """
    Run the infer.py script for each segresnet_* folder with their respective configuration files.
    """

    for i in range(5):  # Loop over segresnet_0 to segresnet_4
        model_folder = f"segresnet_{i}"
        infer_script_path = os.path.join(basepath, f"HN_task2_workdir/{model_folder}/scripts/infer.py")
        config_file_path = os.path.join(basepath, f"HN_task2_workdir/{model_folder}/configs/hyper_parameters.yaml")

        # Define the command and its arguments
        command = [
            'python',
            infer_script_path,
            'run',
            f'--config_file={config_file_path}'
        ]

        try:
            # Run the command and capture output
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Print the standard output
            print(f"Inference for {model_folder} executed successfully. Output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            # Handle execution errors
            print(f"An error occurred while executing inference for {model_folder}: {e}")
            print(f"Standard Error Output:\n{e.stderr}")

def reinsert_predicted_masks():
    """
    Reinsert the predicted masks back into the original full-size images,
    apply postprocessing on the cropped masks, and modify them based on preRT expansion masks.
    """

    # Paths
    original_midRT_dir = os.path.join(basepath, 'input/images/mid-rt-t2w-head-neck/')
    tmp_midRT_dir = os.path.join(basepath, 'tmp/images/mid-rt-t2w-head-neck/')
    preRT_expansion_dir = os.path.join(basepath, 'tmp/preRT_expansion/')
    output_dir = os.path.join(basepath, 'output/images/mri-head-neck-segmentation/')  # Changed to final output directory

    midRT_image_files = [f for f in os.listdir(original_midRT_dir) if f.endswith('.mha')]
    midRT_image_files.sort()

    for midRT_image_file in midRT_image_files:
        # List to store predicted masks from each model
        model_pred_arrays = []

        for i in range(5):  # Loop over segresnet_0 to segresnet_4
            model_folder = f"segresnet_{i}"
            prediction_dir = os.path.join(basepath, f"HN_task2_workdir/{model_folder}/prediction_testing/tmp/images/mid-rt-t2w-head-neck")
            pred_mask_filename = midRT_image_file.replace('.mha', '.nii.gz')  # Assuming the predicted masks are saved as .nii.gz
            pred_mask_path = os.path.join(prediction_dir, pred_mask_filename)
            pred_mask_img = sitk.ReadImage(pred_mask_path)
            pred_mask_array = sitk.GetArrayFromImage(pred_mask_img)
            model_pred_arrays.append(pred_mask_array)

        # Stack predictions along a new axis
        stacked_preds = np.stack(model_pred_arrays, axis=-1)

        # Ensemble predictions using majority voting
        # For each voxel, find the most common label among the models
        ensembled_pred_array, _ = mode(stacked_preds, axis=-1)
        ensembled_pred_array = np.squeeze(ensembled_pred_array, axis=-1).astype(np.uint8)

        # Apply postprocessing to remove small lesions on the cropped mask
        ensembled_pred_array = remove_small_lesions(ensembled_pred_array, min_size_tumor=200, min_size_node=60)

        # Convert back to SimpleITK image
        ensembled_pred_img = sitk.GetImageFromArray(ensembled_pred_array)
        ensembled_pred_img.CopyInformation(pred_mask_img)

        # Load the bounding box info
        base_filename = os.path.splitext(midRT_image_file)[0]
        bbox_info_filename = f"{base_filename}_bbox_info.json"
        bbox_info_path = os.path.join(tmp_midRT_dir, bbox_info_filename)
        if not os.path.exists(bbox_info_path):
            raise FileNotFoundError(f"Bounding box info file not found: {bbox_info_path}")

        with open(bbox_info_path, 'r') as f:
            bbox_info = json.load(f)

        min_coords = bbox_info['min_coords']
        max_coords = bbox_info['max_coords']
        original_shape = bbox_info['original_shape']

        # Compute the destination index for the PasteImageFilter
        # SimpleITK uses (x, y, z), so reverse the order of min_coords
        destination_index = [int(idx) for idx in min_coords[::-1]]  # x, y, z

        # Resample the ensembled mask to match the cropped midRT image if necessary
        cropped_midRT_path = os.path.join(tmp_midRT_dir, midRT_image_file)
        cropped_midRT_img = sitk.ReadImage(cropped_midRT_path)

        if ensembled_pred_img.GetSize() != cropped_midRT_img.GetSize() or ensembled_pred_img.GetSpacing() != cropped_midRT_img.GetSpacing():
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetReferenceImage(cropped_midRT_img)
            resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
            ensembled_pred_img = resample_filter.Execute(ensembled_pred_img)
            print(f"Resampled ensembled mask to match the cropped midRT image size and spacing.")

        # Load the preRT expansion mask corresponding to this case
        preRT_expansion_filename = midRT_image_file  # Assuming filenames correspond
        preRT_expansion_path = os.path.join(preRT_expansion_dir, preRT_expansion_filename)
        if not os.path.exists(preRT_expansion_path):
            raise FileNotFoundError(f"PreRT expansion mask file not found: {preRT_expansion_path}")

        preRT_expansion_img = sitk.ReadImage(preRT_expansion_path)

        # Resample preRT expansion mask to match ensembled mask if necessary
        if preRT_expansion_img.GetSize() != ensembled_pred_img.GetSize() or preRT_expansion_img.GetSpacing() != ensembled_pred_img.GetSpacing():
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetReferenceImage(ensembled_pred_img)
            resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
            preRT_expansion_img = resample_filter.Execute(preRT_expansion_img)
            print(f"Resampled preRT expansion mask to match the ensembled mask image size and spacing.")

        # Convert images to numpy arrays
        ensembled_pred_array = sitk.GetArrayFromImage(ensembled_pred_img)
        preRT_expansion_array = sitk.GetArrayFromImage(preRT_expansion_img)

        # Perform the voxel-wise comparison and overwriting
        # If preRT_expansion has value 1 (tumor) where ensembled mask has value 2 (node), overwrite ensembled mask's value with 1 (tumor)
        condition_tumor = (preRT_expansion_array == 1) & (ensembled_pred_array == 2)
        ensembled_pred_array[condition_tumor] = 1

        # Similarly for nodes
        condition_node = (preRT_expansion_array == 2) & (ensembled_pred_array == 1)
        ensembled_pred_array[condition_node] = 2

        # Convert back to SimpleITK image
        modified_pred_mask_img = sitk.GetImageFromArray(ensembled_pred_array)
        modified_pred_mask_img.CopyInformation(ensembled_pred_img)

        # Create an empty image with the same size and spacing as the original midRT image
        original_midRT_path = os.path.join(original_midRT_dir, midRT_image_file)
        original_midRT_img = sitk.ReadImage(original_midRT_path)
        full_size_pred_mask_img = sitk.Image(original_midRT_img.GetSize(), modified_pred_mask_img.GetPixelID())
        full_size_pred_mask_img.SetOrigin(original_midRT_img.GetOrigin())
        full_size_pred_mask_img.SetSpacing(original_midRT_img.GetSpacing())
        full_size_pred_mask_img.SetDirection(original_midRT_img.GetDirection())

        # Paste the modified predicted mask into the full-size image
        full_size_pred_mask_img = sitk.Paste(
            destinationImage=full_size_pred_mask_img,
            sourceImage=modified_pred_mask_img,
            sourceSize=modified_pred_mask_img.GetSize(),
            sourceIndex=[0, 0, 0],
            destinationIndex=destination_index
        )

        # Save the final output
        os.makedirs(output_dir, exist_ok=True)
        final_output_path = os.path.join(output_dir, 'output.mha')  # Assuming single output file
        sitk.WriteImage(full_size_pred_mask_img, final_output_path)
        print(f"Final output saved at {final_output_path}")

def main():
    # Step 1: Process the cases

    # Define the input directories
    preRT_image_dir = os.path.join(basepath, "input/images/registered-pre-rt-head-neck/")
    midRT_image_dir = os.path.join(basepath, "input/images/mid-rt-t2w-head-neck/")
    preRT_mask_dir = os.path.join(basepath, "input/images/registered-pre-rt-head-neck-segmentation/")

    # Check if directories exist
    if not os.path.isdir(preRT_image_dir):
        raise FileNotFoundError(f"Directory not found: {preRT_image_dir}")
    if not os.path.isdir(midRT_image_dir):
        raise FileNotFoundError(f"Directory not found: {midRT_image_dir}")
    if not os.path.isdir(preRT_mask_dir):
        raise FileNotFoundError(f"Directory not found: {preRT_mask_dir}")

    process_cases(preRT_image_dir, midRT_image_dir, preRT_mask_dir)

    # Step 2: Create the JSON file
    midrt_dir = os.path.join(basepath, "tmp/images/mid-rt-t2w-head-neck/")
    prertr_dir = os.path.join(basepath, "tmp/images/registered-pre-rt-head-neck/")
    output_json_file = os.path.join(basepath, "tmp/HN_preRT_data.json")

    # Create the JSON file
    create_json_file(midrt_dir, prertr_dir, output_json_file)

    # Step 3: Run the infer script for each model
    run_infer_script()

    # Step 4: Reinsert predicted masks into full-size images, apply postprocessing
    reinsert_predicted_masks()

if __name__ == "__main__":
    main()

