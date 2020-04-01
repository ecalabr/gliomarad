""" takes a multi-label mask and converts it to a single label binary mask """

import os
from glob import glob
import argparse
import nibabel as nib
import numpy as np

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/media/ecalabr/scratch/qc_complete',
                    help="Path to data directory")
parser.add_argument('--mask', default='tumor_seg',
                    help="Reference file prefix")
parser.add_argument('--output', default="tumor_mask",
                    help="Target file prefix to copy header info in to")
parser.add_argument('--val', default=1,
                    help="Threshold value for generating  if singular, else list of values to match with only commas between: '1,2,5'")

if __name__ == '__main__':

    # check input arguments
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "No data directory found at {}".format(args.data_dir)
    masks = glob(args.data_dir + '/*/*' + args.mask + '.nii.gz')
    masks.sort()
    assert masks, "No reference files found with prefix {}".format(args.mask)
    vals_list = [float(x) for x in args.val.split(',')]

    # make the list of output arguments
    outputs = []
    for mask in masks:
        out = mask.split(args.mask)[0] + args.output + mask.split(args.mask)[1]
        outputs.append(out)

    # for loop
    for mask, output in zip(masks, outputs):
        if not os.path.isfile(output):
            nii = nib.load(mask)
            if len(vals_list) > 1:
                newimg = np.isin(nii.get_data(), vals_list)
            else:
                newimg = nii.get_data()>vals_list[0]
            newnii = nib.Nifti1Image(newimg.astype(np.uint8), nii.affine)
            nib.save(newnii, output)
            print(output)
