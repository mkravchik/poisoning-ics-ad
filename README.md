# poisoning-ics-ad
Official implementation of "Poisoning Attacks on Cyber Attack Detectors for Industrial Control Systems" by Moshe Kravchik, Battista Biggio, and Asaf Shabtai, SAC 2021.
# UNDER CONSTRUCTION
## Requirements
 * Python 3.6
 * tensorflow 1.15
 * Keras 2.2.4
 * munch 2.5.0
 * numpy 1.18.1
 * scikit-image==0.16.2
 * scikit-learn==0.22.1
 * scipy==1.4.1

For running the SWaT tests the code is expecting to have the dataset that can be requested at https://itrust.sutd.edu.sg/itrust-labs_datasets/.
The dataset train and test files should be subsampled at the 5 seconds rate and saved locally in the files named SWaT_Dataset_Normal_sub5.csv and SWaT_Dataset_Attack_sub5.csv, correspondingly.
  
## Usage
