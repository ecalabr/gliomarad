from input_fn_2 import *
import nibabel as nib
import os
import scipy.ndimage as ndi

# define constants
file_prefixes = ["T1_wm", "T2_wm", "FLAIR_wm", "DWI_wm", "ASL_wm"]
label_prefix = ["T1gad_wm"]
mask_prefix = ["tumor_seg"]
study_dir = '/Users/ecalabrese/Dropbox/12286876'
mydir = '/Users/ecalabrese/Desktop/test'
data_format = 'channels_last'
patch_size = 16
augment = True

# run patch loader
data, labels = patch_loader(study_dir, file_prefixes, label_prefix, mask_prefix, data_format, patch_size, augment)


affine = np.eye(4)

nib.save(nib.Nifti1Image(data, affine), os.path.join(mydir, 'data.nii.gz'))
nib.save(nib.Nifti1Image(labels, affine), os.path.join(mydir, 'labels.nii.gz'))
#nib.save(nib.Nifti1Image(mask, affine), os.path.join(mydir, 'mask.nii.gz'))