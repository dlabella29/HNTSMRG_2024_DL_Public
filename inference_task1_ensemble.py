# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
import sys
import logging
from typing import Optional, Sequence, Union
import SimpleITK as sitk
import yaml
import numpy as np
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __package__ in (None, ""):
    from segmenter import Segmenter, dist_launched, run_segmenter
else:
    from .segmenter import Segmenter, dist_launched, run_segmenter

# Define the base path
BASE_PATH = "/home/dlabella29/Auto3DSegDL/Auto3DSegDL/"

# Define the paths under the base path
INPUT_PATH = os.path.join(BASE_PATH, "input")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
WORKDIR_PATH = os.path.join(BASE_PATH, "HN_pre_3.14")
TMP_PATH = os.path.join(BASE_PATH, "tmp")

# Ensure the necessary directories exist
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Define the input directory and the output paths
input_directory = os.path.join(INPUT_PATH, 'images', 'pre-rt-t2w-head-neck')
output_json_file = os.path.join(TMP_PATH, "HN_preRT_data.json")

# Get the filename from the input directory (assumes there's only one .mha file)
filename_mha = next((f for f in os.listdir(input_directory) if f.endswith('.mha')), None)

if filename_mha is None:
    raise FileNotFoundError("No .mha file found in the input directory.")

# Define the full path for the input .mha file and the output .nii.gz file
input_file_mha = os.path.join(input_directory, filename_mha)


# Define the output file path
output_file = os.path.join(OUTPUT_PATH, 'images', 'mri-head-neck-segmentation', 'output.mha')

print(f"Output file: {output_file}")

# Create the JSON structure
json_data = {
    "testing": [
        {
            "image": [
                filename_mha
            ]
        }
    ]
}

# Write the JSON data to the output file
with open(output_json_file, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file has been created at {output_json_file} with the .nii.gz filename.")

# Create a list of config files for the 5 models
config_files = [
    os.path.join(WORKDIR_PATH, f"segresnet_{i}", "configs", "hyper_parameters.yaml") for i in range(5)
]

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)


def run(**override):
    # For each config file, perform inference
    predictions = []
    for i, config_file in enumerate(config_files):
        # Check if the config file exists and is valid YAML
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r') as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format in {config_file}: {e}")

        # Override the output directory to be unique for each model to avoid conflicts
        model_output_dir = os.path.join(TMP_PATH, f"predictions_{i}")
        os.makedirs(model_output_dir, exist_ok=True)
        override["trainer#output_dir"] = model_output_dir
        override["infer#enabled"] = True
        override["dataset#data_list_file_path"] = output_json_file

        # Run the segmenter with the current config file
        run_segmenter(config_file=config_file, **override)
        print(f"Ran segmenter on config file {config_file}, saved to {model_output_dir}")

        # The predictions are saved in model_output_dir/prediction_testing
        prediction_folder = os.path.join(model_output_dir, 'prediction_testing')
        if not os.path.isdir(prediction_folder):
            print(f"Prediction folder not found: {prediction_folder}")
            continue

        # Collect the prediction files from prediction_folder
        pred_files = [
            os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith('.nii.gz')
        ]
        if pred_files:
            predictions.extend(pred_files)
        else:
            print(f"No prediction files found in {prediction_folder}")

    if not predictions:
        raise RuntimeError("No predictions found from any model.")

    # After collecting all predictions, perform ensembling
    # Assuming all predictions are of the same shape
    prediction_arrays = []
    for pred_file in predictions:
        pred_image = sitk.ReadImage(pred_file)
        pred_array = sitk.GetArrayFromImage(pred_image)
        prediction_arrays.append(pred_array)

    # Stack the prediction arrays
    prediction_stack = np.stack(prediction_arrays, axis=0)

    # Perform ensembling using majority voting
    ensemble_pred_array = (np.sum(prediction_stack, axis=0) >= 3).astype(np.uint8)

    # Convert the ensembled array back to SimpleITK image
    ensemble_pred_image = sitk.GetImageFromArray(ensemble_pred_array)

    # Use one of the prediction images as a reference for spacing, origin, direction
    reference_image = sitk.ReadImage(predictions[0])
    ensemble_pred_image.CopyInformation(reference_image)

    # Write the ensembled prediction to output_file in .mha format
    sitk.WriteImage(ensemble_pred_image, output_file, useCompression=True)

    print(f"Ensembled prediction saved to {output_file}")


if __name__ == "__main__":
    # Automatically run the main function when the script starts
    run()
