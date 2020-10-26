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
The following command will train the network using the parameters specified in params.json:
```bash
python train.py -p params/params.json
```
For help with parameter files, please refer to a separate markdown file in the "params" directory of this project.

Training outputs will be located in the model directory as specified in the param file.
 
## Predict
The following command will use trained model weights to predict one image for patient ID 123456:
```bash
python predict.py -p params/params.json -d data_dir/123456
```
By default, the predicted output will be placed in the model directory in a subdirectory named "prediction"; however, the user can specify a different output directory using "-o":
```bash
python predict.py -p params/params.json -d data_dir/123456 -o outputs/
```

## Evaluate
The following command will evaluate the trained network using the testing portion of the data as specified in the params.json file:
```bash
python evaluate.py -p params/params.json
```
By default, the evaluation output will be placed in the model directory in a subdirectory named "evaluation"; however, the user can specify a different output directory using "-o":
```bash
python evaluate.py -p params/params.json -o eval_outputs/
```
Evaluation metrics can be manually specified using "-c":
```bash
python evaluate.py -p params/params.json -c smape ssim logac
```