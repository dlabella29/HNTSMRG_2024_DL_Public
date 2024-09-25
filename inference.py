import os
import sys
import logging
from typing import Optional, Sequence, Union
import SimpleITK as sitk
import json
import yaml
import numpy as np
from scipy import ndimage
import torch

# Insert the current script's directory to the system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Determine the device to use for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import necessary components from the segmenter module
if __package__ in (None, ""):
    from segmenter import Segmenter, dist_launched, run_segmenter
else:
    from .segmenter import Segmenter, dist_launched, run_segmenter

# Define base paths
BASE_PATH = '/'

INPUT_PATH = os.path.join(BASE_PATH, 'input', 'images', 'pre-rt-t2w-head-neck')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
HN_PRE_PATH = os.path.join(BASE_PATH, 'HN_pre_3.14')
TMP_PATH = os.path.join(BASE_PATH, 'tmp')
PREPROCESS_PATH = os.path.join(TMP_PATH, 'output_preprocessed')

# Create necessary directories
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(PREPROCESS_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'images', 'mri-head-neck-segmentation'), exist_ok=True)

# Define the JSON output file path
output_json_file = os.path.join(TMP_PATH, 'HN_preRT_data.json')

# Locate the input .mha file
def get_input_mha(input_dir: str) -> str:
    for file in os.listdir(input_dir):
        if file.endswith('.mha'):
            return os.path.join(input_dir, file)
    raise FileNotFoundError("No .mha file found in the input directory.")

input_file_mha = get_input_mha(INPUT_PATH)
print(f"Input .mha file: {input_file_mha}")

# Define the final output .mha file path
final_output_file = os.path.join(OUTPUT_PATH, 'images', 'mri-head-neck-segmentation', 'output.mha')
print(f"Final output file will be saved to: {final_output_file}")

# Create the JSON structure for inference
json_data = {
    "testing": [
        {
            "image": [
                input_file_mha
            ]
        }
    ]
}

# Write the JSON data to the file
with open(output_json_file, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
print(f"JSON data file created at {output_json_file}")

# Define the configuration files for each segresnet model
NUM_MODELS = 6
CONFIG_FILES = [
    os.path.join(HN_PRE_PATH, f"segresnet_{i}", "configs", "hyper_parameters.yaml") for i in range(NUM_MODELS)
]

# Define the weights for each model
MODEL_WEIGHTS = {
    0: 0.20,
    1: 0.1375,
    2: 0.175,
    3: 0.175,
    4: 0.175,
    5: 0.1375
}

# Verify all configuration files exist
for idx, cfg in enumerate(CONFIG_FILES):
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"Configuration file not found for model segresnet_{idx}: {cfg}")

class InferenceManager:
    def __init__(self, config_files: list, json_input: str, tmp_path: str, model_weights: dict):
        """
        Initializes the InferenceManager with configuration files, input JSON, temporary path, and model weights.
        """
        self.config_files = config_files
        self.json_input = json_input
        self.tmp_path = tmp_path
        self.model_weights = model_weights
        self.predictions = []
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        """
        Sets up the logger for the InferenceManager.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger("InferenceManager")

    def run_inference(self):
        """
        Runs inference for each model, converts prediction files, and stores the prediction arrays.
        """
        for idx, config_file in enumerate(self.config_files):
            self.logger.info(f"Running inference for model segresnet_{idx} with config: {config_file}")

            # Define the output directory for this model's predictions
            prediction_output_dir = os.path.join(HN_PRE_PATH, f"segresnet_{idx}", "prediction_testing")
            os.makedirs(prediction_output_dir, exist_ok=True)

            # Override parameters for inference
            override = {
                "infer#enabled": True,
                "infer#data_list_file_path": self.json_input,
                "infer#output_dir": prediction_output_dir,
            }

            # Run the segmenter
            run(config_file=config_file, **override)

            # Locate the predicted .nii.gz file
            pred_file = self.get_prediction_file(prediction_output_dir, idx)
            print(f"Prediction file for model segresnet_{idx}: {pred_file}")

            # Convert .nii.gz to .mha and save to tmp/prediction_*
            mha_output_path = os.path.join(self.tmp_path, f"prediction_{idx}.mha")
            self.convert_nii_to_mha(pred_file, mha_output_path)
            print(f"Converted prediction saved to: {mha_output_path}")

            # Load the prediction array
            pred_image = sitk.ReadImage(mha_output_path)
            pred_array = sitk.GetArrayFromImage(pred_image)
            self.predictions.append(pred_array)

    def get_prediction_file(self, pred_dir: str, model_idx: int) -> str:
        """
        Retrieves the prediction file (.nii.gz) from the specified directory.
        """
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
        if len(pred_files) != 1:
            raise ValueError(f"Expected one prediction file in {pred_dir}, found {len(pred_files)} for model segresnet_{model_idx}")
        return os.path.join(pred_dir, pred_files[0])

    @staticmethod
    def convert_nii_to_mha(nii_path: str, mha_path: str):
        """
        Converts a .nii.gz file to a .mha file.
        """
        image = sitk.ReadImage(nii_path)
        sitk.WriteImage(image, mha_path)

    def ensemble_predictions_weighted_majority_vectorized(self):
        """
        Performs weighted majority voting to ensemble predictions and saves the ensembled prediction as .mha.
        """
        self.logger.info("Ensembling predictions from all models using weighted majority voting (vectorized)")
        preds_array = np.stack(self.predictions, axis=0)  # Shape: (NUM_MODELS, z, y, x)
        print(f"Shape of stacked predictions: {preds_array.shape}")

        # Define the weights array in the same order as predictions
        weights = np.array([self.model_weights[i] for i in range(NUM_MODELS)], dtype=np.float32)  # Shape: (NUM_MODELS,)

        # Get unique labels
        unique_labels = np.unique(preds_array)
        print(f"Unique labels found: {unique_labels}")

        # Initialize an array to store the sum of weights for each label
        vote_sums = np.zeros((*preds_array.shape[1:], len(unique_labels)), dtype=np.float32)  # Shape: (z, y, x, num_labels)

        # Accumulate weighted votes for each label
        for idx, label in enumerate(unique_labels):
            vote_sums[..., idx] = np.sum((preds_array == label) * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
            print(f"Computed weighted votes for label {label}")

        # Determine the label with the highest vote sum for each voxel
        ensemble_indices = np.argmax(vote_sums, axis=-1)
        ensemble_array = unique_labels[ensemble_indices].astype(np.uint8)

        print("Ensembled prediction array computed using weighted majority voting (vectorized)")

        # Save the ensembled prediction as .mha
        ensembled_image = sitk.GetImageFromArray(ensemble_array)
        # Copy the metadata from one of the prediction images
        reference_image = sitk.ReadImage(os.path.join(self.tmp_path, 'prediction_0.mha'))
        ensembled_image.CopyInformation(reference_image)
        ensembled_output_file = os.path.join(self.tmp_path, 'ensembled_prediction.mha')
        sitk.WriteImage(ensembled_image, ensembled_output_file)
        print(f"Ensembled prediction saved to: {ensembled_output_file}")

        return ensembled_output_file

    def post_process(self, ensembled_mha: str):
        """
        Performs post-processing on the ensembled prediction:
        1. Identifies and retains only the largest tumor lesion.
        2. Converts smaller tumor lesions to nodes.
        3. Removes tumors with volume <200 mm³.
        4. Removes nodes with volume <60 mm³.
        """
        self.logger.info("Starting post-processing of the ensembled prediction")

        # Read the ensembled image
        ensembled_image = sitk.ReadImage(ensembled_mha)
        ensembled_array = sitk.GetArrayFromImage(ensembled_image)

        # Get voxel spacing to compute volume
        spacing = ensembled_image.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # in mm³

        # Step 1: Identify discrete tumor lesions and retain only the largest
        binary_tumor = (ensembled_array == 1).astype(np.uint8)
        labeled_tumors, num_tumors = ndimage.label(binary_tumor)
        self.logger.info(f"Number of discrete tumor lesions detected: {num_tumors}")

        if num_tumors > 1:
            # Calculate sizes of each tumor lesion
            tumor_sizes = ndimage.sum(binary_tumor, labeled_tumors, range(1, num_tumors + 1))
            largest_tumor_idx = np.argmax(tumor_sizes) + 1  # +1 because labels start at 1

            # Create a mask for the largest tumor
            largest_tumor_mask = (labeled_tumors == largest_tumor_idx)

            # Retain the largest tumor as label 1
            ensembled_array = np.where(largest_tumor_mask, 1, ensembled_array)

            # Convert smaller tumors to nodes (label 2)
            smaller_tumor_mask = (labeled_tumors != largest_tumor_idx) & (labeled_tumors > 0)
            ensembled_array = np.where(smaller_tumor_mask, 2, ensembled_array)

            self.logger.info("Retained only the largest tumor lesion as label 1. Converted smaller lesions to label 2 (nodes).")
        elif num_tumors == 1:
            self.logger.info("Only one tumor lesion detected. No conversion needed.")
        else:
            self.logger.info("No tumor lesions detected.")

        # Step 2: Remove small tumors (label 1) and small nodes (label 2)
        try:
            # Remove small tumors (label 1) with volume <200 mm³
            ensembled_array = self.remove_small_components(
                ensembled_array,
                label_value=1,
                min_volume_mm3=175,
                voxel_volume=voxel_volume
            )
        except Exception as e:
            self.logger.error(f"Error removing small tumors (label 1): {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            # Depending on requirements, you can choose to continue or re-raise the exception
            # For now, we'll continue without modifying ensembled_array
            pass

        try:
            # Remove small nodes (label 2) with volume <60 mm³
            ensembled_array = self.remove_small_components(
                ensembled_array,
                label_value=2,
                min_volume_mm3=50,
                voxel_volume=voxel_volume
            )
        except Exception as e:
            self.logger.error(f"Error removing small nodes (label 2): {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            # Continue without modifying ensembled_array
            pass

        # Save the post-processed image
        post_processed_image = sitk.GetImageFromArray(ensembled_array)
        post_processed_image.CopyInformation(ensembled_image)
        sitk.WriteImage(post_processed_image, final_output_file)
        self.logger.info(f"Post-processed image saved to: {final_output_file}")

    @staticmethod
    def remove_small_components(label_array: np.ndarray, label_value: int, min_volume_mm3: float, voxel_volume: float) -> np.ndarray:
        """
        Removes connected components of a specified label that are smaller than the minimum volume.
        """
        binary_label = (label_array == label_value).astype(np.uint8)
        labeled, num_features = ndimage.label(binary_label)
        sizes = ndimage.sum(binary_label, labeled, range(1, num_features + 1))
        sizes_mm3 = sizes * voxel_volume

        # Identify components to remove
        remove_mask = sizes_mm3 < min_volume_mm3

        if not np.any(remove_mask):
            print(f"No components with label {label_value} are smaller than {min_volume_mm3} mm³. No removal needed.")
            return label_array

        try:
            remove_voxels = remove_mask[labeled - 1]  # labels start at 1
        except IndexError as ie:
            print(f"IndexError while removing small components: {ie}")
            # Return the original array if indexing fails
            return label_array

        # Remove small components
        binary_label[remove_voxels] = 0

        # Update the label array
        updated_label_array = np.where(binary_label, label_value, label_array)

        if label_value == 1:
            print(f"Removed tumors smaller than {min_volume_mm3} mm³")
        elif label_value == 2:
            print(f"Removed nodes smaller than {min_volume_mm3} mm³")

        return updated_label_array

def run(config_file: Optional[Union[str, Sequence[str]]], **override):
    """
    Validates configuration files and runs the segmenter.
    """
    if config_file:
        if isinstance(config_file, list):
            for cf in config_file:
                if not os.path.isfile(cf):
                    raise FileNotFoundError(f"Config file not found: {cf}")
                with open(cf, 'r') as f:
                    try:
                        yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ValueError(f"Invalid YAML format in {cf}: {e}")
        else:
            if not os.path.isfile(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with open(config_file, 'r') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML format in {config_file}: {e}")
    override["infer#enabled"] = True
    run_segmenter(config_file=config_file, **override)
    print(f"Segmenter run completed for config file(s): {config_file}")

def main():
    # Initialize the InferenceManager
    manager = InferenceManager(
        config_files=CONFIG_FILES,
        json_input=output_json_file,
        tmp_path=TMP_PATH,
        model_weights=MODEL_WEIGHTS
    )

    # Run inference for all models
    manager.run_inference()

    # Ensemble the predictions using weighted majority voting (vectorized method)
    ensembled_mha = manager.ensemble_predictions_weighted_majority_vectorized()

    # Perform post-processing and save the final output
    manager.post_process(ensembled_mha)

if __name__ == "__main__":
    main()

