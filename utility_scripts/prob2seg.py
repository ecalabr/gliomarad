""" converts one or more 4D probabilty image(s) into a binary mask using softmax argmax"""

import nibabel as nib
import os
import argparse
import numpy as np
import scipy.ndimage
from glob import glob

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', default=None,
                    help="Path to data image(s). Can be an image or a directory.")
parser.add_argument('--name', default='',
                    help="Name of data image(s). Any string within the data filename. " +
                         "Leave blank for all images, which will be averaged.")
parser.add_argument('--outname', default='binary_mask.nii.gz',
                    help="Name of output image")
parser.add_argument('--outpath', default=None,
                    help="Output path")
parser.add_argument('--clean', action="store_true", default=False,
                    help="Delete inputs after conversion")

def softmax(X, theta=1.0, axis=None):
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def get_largest_component(img):
    s = scipy.ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = scipy.ndimage.label(img,s) # labeling
    sizes = scipy.ndimage.sum(img,labeled_array,range(1,numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if len(sizes) == 1:
        out_img = img
    else:
        max_size = sizes_list[-1]
        max_label = np.where(sizes == max_size)[0] + 1
        component = labeled_array == max_label
        out_img = component
    return out_img

if __name__ == '__main__':

    # check input arguments
    args = parser.parse_args()
    data_in_path = args.data
    assert  data_in_path, "Must specify input data using --data"
    namestr = args.name
    outname = args.outname
    outpath = args.outpath
    clean = args.clean
    if not '.nii.gz' in outname:
        outname = outname.split('.')[0] + '.nii.gz'
    files = []
    data_root = []
    if os.path.isfile(data_in_path):
        files = [data_in_path]
        data_root = os.path.dirname(data_in_path)
    elif os.path.isdir(data_in_path):
        files = glob(data_in_path + '/*' + namestr + '*.nii*')
        data_root = data_in_path
    else:
        raise ValueError("No data found at {}".format(data_in_path))

    # handle outpath creation and make sure output file is not an input
    if outpath and os.path.isdir(outpath):
        data_root = outpath
    nii_out_path = os.path.join(data_root, outname)
    if nii_out_path in files:
        files.remove(nii_out_path)

    # announce
    if files:
        print("Found the following file" + (":" if len(files)==1 else "s, which will be averaged:"))
        for f in files:
            print(f)
    else:
        raise ValueError("No data found at {}".format(data_in_path))

    # load data
    data = []
    nii = []
    for f in files:
        nii = nib.load(f)
        data.append(nii.get_data())
    data = np.mean(data, axis=0)

    # softmax argmax
    out_data = np.argmax(softmax(data, axis=-1), axis=-1)

    # binary ops
    struct = scipy.ndimage.generate_binary_structure(3, 2)
    out_data = scipy.ndimage.morphology.binary_closing(out_data, structure=struct)
    out_data = get_largest_component(out_data)

    # make output nii
    nii_out = nib.Nifti1Image(out_data.astype(np.uint8), nii.affine, nii.header)
    nib.save(nii_out, nii_out_path)

    # if output is created, and cleanup is true, then clean
    if os.path.isfile(nii_out_path) and clean:
        for f in files:
            os.remove(f)
