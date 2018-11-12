import nibabel as nib
import numpy as np
import os

file = '/media/ecalabr/data/gbm_no_qc/11772918/11772918_DTI_eddy.nii.gz'

outname = os.path.join(os.path.dirname(file), 'temp_dwi.nii.gz')

nii = nib.load(file)

data = nii.get_data()

output = np.squeeze(np.mean(data[1:,:,:,:], 3))

affine = nii.get_affine()

nii_out = nib.Nifti1Image(output, affine)

nib.save(nii_out, outname)