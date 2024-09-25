import os
import json
import random
from glob import glob

# Define the master folder where the data is stored
master_folder = "/media/dlabella29/Extreme Pro/HN_Data/HNTSMRG24_train"
output_file_path = "/media/dlabella29/Extreme Pro/HNImageTrLabelTr/HN_all_3_RT_data.json"

# Define the list to store the training data
training_data = []

# Traverse through all case folders in the master folder
case_folders = [f for f in os.listdir(master_folder) if os.path.isdir(os.path.join(master_folder, f)) and f.isdigit()]
case_folders.sort(key=lambda x: int(x))  # Sort by case number

# Iterate through each case folder to gather image and mask pairs
for case_folder in case_folders:
    case_number = case_folder
    midRT_folder = os.path.join(master_folder, case_number, "midRT")
    preRT_folder = os.path.join(master_folder, case_number, "preRT")

    # Get all midRT T2 and mask files
    midRT_T2_files = glob(os.path.join(midRT_folder, "*_midRT_T2.nii.gz"))
    midRT_mask_files = glob(os.path.join(midRT_folder, "*_midRT_mask.nii.gz"))

    # Get all preRT T2 and mask files (registered and unregistered)
    preRT_T2_registered_files = glob(os.path.join(midRT_folder, "*_preRT_T2_registered.nii.gz"))
    preRT_mask_registered_files = glob(os.path.join(midRT_folder, "*_preRT_mask_registered.nii.gz"))
    preRT_T2_files = glob(os.path.join(preRT_folder, "*_preRT_T2.nii.gz"))
    preRT_mask_files = glob(os.path.join(preRT_folder, "*_preRT_mask.nii.gz"))

    # Map T2 files to their corresponding mask files for midRT cases
    for t2_file in midRT_T2_files:
        case_id = os.path.basename(t2_file).split("_")[0]
        mask_file = os.path.join(midRT_folder, f"{case_id}_midRT_mask.nii.gz")

        if os.path.exists(mask_file):
            training_data.append({
                "fold": 0,  # Fold assignment will be done later
                "image": [t2_file],
                "label": mask_file
            })

    # Map T2 files to their corresponding mask files for preRT-registered cases
    for t2_file in preRT_T2_registered_files:
        case_id = os.path.basename(t2_file).split("_")[0]
        mask_file = os.path.join(midRT_folder, f"{case_id}_preRT_mask_registered.nii.gz")

        if os.path.exists(mask_file):
            training_data.append({
                "fold": 0,  # Fold assignment will be done later
                "image": [t2_file],
                "label": mask_file
            })

    # Map T2 files to their corresponding mask files for preRT cases
    for t2_file in preRT_T2_files:
        case_id = os.path.basename(t2_file).split("_")[0]
        mask_file = os.path.join(preRT_folder, f"{case_id}_preRT_mask.nii.gz")

        if os.path.exists(mask_file):
            training_data.append({
                "fold": 0,  # Fold assignment will be done later
                "image": [t2_file],
                "label": mask_file
            })
# Assign folds to the training data (8 folds in total)
fold_count = 8
cases_per_fold = len(training_data) // fold_count

for i, entry in enumerate(training_data):
    entry['fold'] = i // cases_per_fold  # Assign fold number

# If there are any remaining cases after dividing equally, assign them to the last fold
for i in range(len(training_data) % fold_count):
    training_data[-(i + 1)]['fold'] = fold_count - 1

# Prepare the final JSON structure
json_data = {
    "training": training_data,
    "testing": []  # No testing cases as specified
}

# Save the JSON file to the specified output path
with open(output_file_path, 'w') as outfile:
    json.dump(json_data, outfile, indent=4)

print(f"JSON file saved to {output_file_path}")
