""" makes a binary mask from any image using a specified threshold and an optional brain mask """

import nibabel as nib
import os
import argparse
import numpy as np
import scipy.ndimage
from glob import glob
from skimage.morphology import remove_small_objects

########################## define functions ##########################
def thresh_mask(image, mask, thr, output, less=False):

    # load data
    im_nii = nib.load(image)
    im_data = im_nii.get_data()
    mask_data = nib.load(mask).get_data()

    # erode mask
    #struct = scipy.ndimage.generate_binary_structure(3, 2)  # rank 3, connectivity 2
    #mask_data = scipy.ndimage.morphology.binary_erosion(mask_data, structure=struct)  # erosion

    # threshold image data within mask
    if less:
        thresh_data = np.where(mask_data > 0, np.logical_and(im_data < thr, im_data > 0.), False).astype(float)
    else:
        thresh_data = np.where(mask_data > 0, im_data > thr, False).astype(float)

    # binary ops
    struct = scipy.ndimage.generate_binary_structure(3, 2)  # rank 3, connectivity 2
    thresh_data = scipy.ndimage.morphology.binary_erosion(thresh_data, structure=struct)  # erosion
    labeled_array, numpatches = scipy.ndimage.label(thresh_data, struct)  # labeling
    thresh_data = remove_small_objects(labeled_array, min_size=200, connectivity=3)  # remove small objects
    thresh_data = scipy.ndimage.morphology.binary_dilation(thresh_data, structure=struct)  # dilation
    thresh_data = scipy.ndimage.morphology.binary_fill_holes(thresh_data)  # fill holes
    thresh_data = scipy.ndimage.morphology.binary_opening(thresh_data, structure=struct)  # final opening

    # make output nii
    nii_out = nib.Nifti1Image(thresh_data, im_nii.affine, im_nii.header)
    nib.save(nii_out, output)

########################## executed  as script ##########################
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--data', default=None,
                        help="Suffix string for data to be thresholded")
    parser.add_argument('--direc', default=None,
                        help="Optionally name a specific directory to process")
    parser.add_argument('--thresh', default=None,
                        help="Threshold for mask")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False,
                        help="List all directories and exit")
    parser.add_argument('--mask', default="combined_brain_mask",
                        help="Suffix of the mask to be edited")
    parser.add_argument('--outname', default="thresh_mask",
                        help="Suffix of the output threshold mask")
    parser.add_argument('--less', action="store_true", default=False,
                        help="Use this flag for a less than threshold (default greater than)")

    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    spec_direc = args.direc
    if spec_direc:
        assert os.path.isdir(spec_direc), "Specified directory does not exist at {}".format(spec_direc)
    else:
        assert data_dir, "Must specify data directory using param --data_dir"
        assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)

    start = args.start
    end = args.end

    # handle specific directory
    if spec_direc:
        my_direcs = [spec_direc]
    else:
        # list all subdirs with the processed data
        my_direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
        my_direcs = sorted(my_direcs, key=lambda x: int(os.path.basename(x)))

        # set start and stop for subset/specific diectories only using options below
        if end:
            my_direcs = my_direcs[int(start):int(end) + 1]
        else:
            my_direcs = my_direcs[int(start):]
    if isinstance(my_direcs, str):
        my_direcs = [my_direcs]

    # handle list flag
    if args.list:
        for i, item in enumerate(my_direcs, 0):
            print(str(i) + ': ' + item)
        exit()

    # handle data and mask arguments
    tmp_direcs = my_direcs
    images = []
    masks = []
    outputs = []
    for item in tmp_direcs:
        if glob(item + '/*' + args.mask + '.nii.gz') and glob(item + '/*' + args.data + '.nii.gz'):
            images.append(glob(item + '/*' + args.data + '.nii.gz')[0])
            masks.append(glob(item + '/*' + args.mask + '.nii.gz')[0])
            outputs.append(glob(item + '/*' + args.data + '.nii.gz')[0].rsplit('.nii.gz', 1)[0] + "_" + args.outname + '.nii.gz')
        else:
            print("Directory {} is missing some required files".format(item))

    # handle thresh argument
    try:
        thresh = float(args.thresh)
    except:
        raise ValueError("Threshold specified with --thresh cannot be cast to float")

    # do work
    for i, item in enumerate(images):
        print("Making threshold mask for directory {}".format(item))
        thresh_mask(item, masks[i], thresh, outputs[i], args.less)