""" takes a multi-label mask and converts it to a single label binary mask """

import os
from glob import glob
import argparse
import nibabel as nib
import numpy as np

########################## define functions ##########################
def binary_mask(mask, output, vals_list):

    # if output doesnt exist then create it
    if not os.path.isfile(output):
        nii = nib.load(mask)
        if len(vals_list) > 1:
            newimg = np.isin(nii.get_data(), vals_list)
        else:
            newimg = nii.get_data() >= vals_list[0]
        newnii = nib.Nifti1Image(newimg.astype(np.float), nii.affine) # float actuall results in smaller zipped files
        nib.save(newnii, output)

    return output

########################## executed  as script ##########################
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--mask', default='tumor_seg',
                        help="Reference file prefix")
    parser.add_argument('--output', default="tumor_mask",
                        help="Target file prefix to copy header info in to")
    parser.add_argument('--val', default=1,
                        help="Inclusive hreshold value for generating if singular, "
                             "else list of values to match with only commas between: '1,2,5'")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False)

    # check input arguments
    args = parser.parse_args()
    # handle .nii.gz arguments
    args.mask = args.mask.split('.nii.gz')[0] if args.mask.endswith('.nii.gz') else args.mask
    args.output = args.output.split('.nii.gz')[0] if args.output.endswith('.nii.gz') else args.output
    assert args.data_dir, "Must specify data directory using --data_dir"
    assert os.path.isdir(args.data_dir), "No data directory found at {}".format(args.data_dir)
    masks = sorted(glob(args.data_dir + '/*/*' + args.mask + '.nii.gz'))
    assert masks, "No reference files found with prefix {}".format(args.mask)
    if not isinstance(args.val, str):
        args.val = str(args.val)
    my_vals_list = [float(x) for x in args.val.split(',')]

    # handle list argument
    if args.list:
        for i, item in enumerate(masks, 0):
            print(str(i) + ': ' + item)
        exit()

    # handle start and stop arguments
    if args.end:
        masks = masks[int(args.start):int(args.end)+1]
    else:
        masks = masks[int(args.start):]

    # make the list of output arguments
    outputs = []
    for my_mask in masks:
        out = my_mask.split(args.mask)[0] + args.output + my_mask.split(args.mask)[1]
        outputs.append(out)

    # for loop
    for my_mask, my_output in zip(masks, outputs):
        binary_mask(my_mask, my_output, my_vals_list)
