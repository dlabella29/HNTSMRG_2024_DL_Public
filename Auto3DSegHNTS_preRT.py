from monai.apps.auto3dseg import AutoRunner
from datetime import datetime
import os
import yaml

from sympy import false


# use this if using modified files.
# make it resampled instead of number if using ground truth
# maskSize = "9"
# maskSize="resampled"

def main():

  for maskSize in [0]:
      runner = AutoRunner(
          input={
              "modality": "MRI",
              "dataroot": "/media/dlabella29/Extreme Pro/HNImageTrLabelTr/",
              "datalist": f"/media/dlabella29/Extreme Pro/HNImageTrLabelTr/HN_preRT_data.json",
              "sigmoid": False,
              "num_epochs": 900,
              "class_names": ["tumor","node"],
          },
          algos="segresnet",
          work_dir=f"./HN_9.10.24"
      )
      train_param = {
          "num_epochs": 900,
      }
      runner.set_training_params(train_param)
      print(f"Starting run for mask size: {maskSize}")
      # Get the current time
      print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
      runner.run()

if __name__ == '__main__':
  main()


