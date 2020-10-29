# Gadnet
Gadnet is a 3D fully convolutional deep neural network designed to synthesize gadolinium enahnced T1 weighted brain MR images from pre-contrast images.

# Installation
The following command will clone a copy of Gadnet to your computer using git:
```bash
git clone https://github.com/ecalabr/gadnet.git
```

# Data directory tree setup
Gadnet expects your image data to be in Nifti format with a specific directory tree. The following example starts with any directory (referred to as data_dir).

```bash
data_dir/
```
This is an example of the base directory for all of the image data that you want to use. All subdirectories in this folder should contain individual patient image data.

```bash
data_dir/123456/
```
This is an example of an individual patient study directory. The directory name is typically a patient ID, but can be any folder name that does not contain the "_" character

```bash
data_dir/123456/123456_T1gad.nii.gz
```
This is an example of a single patient image. The image file name must start with the patient ID (i.e. the same as the patient directory name) followed by a "_" character. All Nifti files are expected to be g-zipped.

# Usage
## Train
The following command will train the network using the parameters specified in the param file (-p):
```bash
python train.py -p gadnet/gadnet_params.json
```
For help with parameter files, please refer to a separate markdown file in the "gadnet" subdirectory of this project.

Training outputs will be located in the model directory as specified in the param file.
 
## Predict
The following command will use the trained "full model" weights checkpoint (-c) to predict for a single patient (-s) with ID 123456:
```bash
python predict.py -p gadnet/gadnet_full_params.json -s data_dir/123456 -c full_model
```
To predict using the "reduced model", use the following command. Note that the specified paramter file (-p) must match the model checkpoint:
```bash
python predict.py -p gadnet/gadnet_reduced_params.json -s data_dir/123456 -c reduced_model
```
By default, the predicted output will be placed in the model directory in a subdirectory named "prediction"; however, the user can specify a different output directory using "-o":
```bash
python predict.py -p gadnet/gadnet_full_params.json -s data_dir/123456 -c full_model -o outputs/
```

## Evaluate
The following command will evaluate the trained network using the testing portion of the data as specified in the params file (-p) using the full model weights checkpoint (-c):
```bash
python evaluate.py -p gadnet/gadnet_full_params.json -c full_model
```
By default, the evaluation output will be placed in the model directory in a subdirectory named "evaluation"; however, the user can specify a different output directory using "-o":
```bash
python evaluate.py -p gadnet/gadnet_full_params.json -c full_model -o eval_outputs/
```
Evaluation metrics can be manually specified using "-t":
```bash
python evaluate.py -p gadnet/gadnet_full_params.json -c full_model -t smape ssim logac
```