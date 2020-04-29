""" combines a folder full of seperate DTI niis into a single 4D nii while averaging b0 volumes """

import os
import csv
import nibabel as nib
import numpy as np
from glob import glob
import argparse


# define functions
# define function to combine a split 55 direction DTI file if necessary
def combine_multi_dti(directory, b0first=False):
    # define file names
    dtis = sorted(glob(directory + '/*.nii.gz'))
    bvecs = sorted(glob(directory + '/*.bvec'))
    bvals = sorted(glob(directory + '/*.bval'))
    outfile = os.path.join(directory, 'DTI.nii.gz')
    vals_out = os.path.join(directory, 'DTI.bval')
    vecs_out = os.path.join(directory, 'DTI.bvec')
    if outfile in dtis:
        dtis.remove(outfile)
    if vecs_out in bvecs:
        bvecs.remove(vecs_out)
    if vals_out in bvals:
        bvals.remove(vals_out)
    # check if files exist
    bvals_present = True
    if not dtis:
        print("- no DTI files found in data directory: " + directory)
        return
    if not bvals or not bvecs:
        print("- bvecs and bvals do not exist in data directory: " + directory)
        bvals_present = False
        if b0first:
            print('- flag --first was passed, assuming first volume is B0 and rest are DWIs')
        else:
            print('- cannot continue, use --first flag to assume first volume is B0 and rest are DWIs')
            return
    # iterate through DTI list building b0 and dwi volumes
    b0 = []
    dwi = []
    bvals_out = []
    bvecs_out = []
    dwi_nii = []
    for x, i in enumerate(dtis, 0):
        dwi_nii = nib.load(dtis[x])
        dwi_d = dwi_nii.get_fdata()
        # read bvals if present
        if bvals_present:
            with open(bvals[x], 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                rows = [item for item in reader]
                # check for space delimited
                rows = [rows[0][0].split(' ')] if len(rows[0]) == 1 else rows
        # make boolean arrays for b0 and DWI volumes
        if b0first:
            b0_bool = [True] + [False] * (dwi_nii.shape[3] - 1)
            dwi_bool = [False] + [True] * (dwi_nii.shape[3] - 1)
        else:
            b0_bool = [int(item) == 0 for item in rows[0]]
            dwi_bool = [int(item) > 0 for item in rows[0]]
        # read bvecs if present
        if bvals_present:
            with open(bvecs[x], 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                vecs_rows = [item for item in reader]
                # check for space delimited
                vecs_rows = [vecs_rows[y][0].split(' ') for y in range(len(vecs_rows))] if len(
                    vecs_rows[0]) == 1 else vecs_rows
        # report
        print("- combining file " + dtis[x] + " with " + str(sum(b0_bool)) + " B0(s) and " + str(
            sum(dwi_bool)) + " DWI(s)")
        # if first element, create arrays
        if x == 0:
            # b0
            b0 = dwi_d[:, :, :, b0_bool]
            if len(b0.shape) < 4:
                b0 = np.reshape(b0, [b0.shape[0], b0.shape[1], b0.shape[2], -1])
            # dwi
            dwi = dwi_d[:, :, :, dwi_bool]
            if len(dwi.shape) < 4:
                dwi = np.reshape(dwi, [dwi.shape[0], dwi.shape[1], dwi.shape[2], -1])
            if bvals_present:
                # vals
                bvals_out = np.concatenate([[['0']], np.array(rows)[:, dwi_bool]], axis=1)
                # vecs
                bvecs_out = np.concatenate([[['0'], ['0'], ['0']], np.array(vecs_rows)[:, dwi_bool]], axis=1)
        # if not first element, concatenate
        else:
            # b0
            b0_cat = dwi_d[:, :, :, b0_bool]
            if len(b0_cat.shape) < 4:
                b0_cat = np.reshape(b0_cat, [b0_cat.shape[0], b0_cat.shape[1], b0_cat.shape[2], -1])
            b0 = np.concatenate([b0, b0_cat], axis=3)
            # dwi
            dwi_cat = dwi_d[:, :, :, dwi_bool]
            if len(dwi_cat.shape) < 4:
                dwi_cat = np.reshape(dwi_cat, [dwi_cat.shape[0], dwi_cat.shape[1], dwi_cat.shape[2], -1])
            dwi = np.concatenate([dwi, dwi_cat], axis=3)
            if bvals_present:
                # vals
                bvals_out = np.concatenate([bvals_out, np.array(rows)[:, dwi_bool]], axis=1)
                # vecs
                bvecs_out = np.concatenate([bvecs_out, np.array(vecs_rows)[:, dwi_bool]], axis=1)
    # after loop, average b0, combine with DWI, and write
    data_out = np.concatenate([np.expand_dims(np.mean(b0, axis=3), axis=-1), dwi], axis=3)
    dwi_nii = nib.Nifti1Image(data_out, dwi_nii.affine, dwi_nii.header)
    nib.save(dwi_nii, outfile)
    if bvals_present:
        # write bvals and vecs
        with open(vals_out, 'w+') as f:
            writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(bvals_out)
        with open(vecs_out, 'w+') as f:
            writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(bvecs_out)

    return [outfile, bvals_out, bvecs_out]


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--first', action="store_true", default=False,
                        help="Forces b0 as first volume in each file")

    # parse args
    args = parser.parse_args()

    # sanity check
    assert args.data_dir, "No data directory specified. Use --data_dir"

    # do work if not already done
    if not os.path.isdir(args.data_dir):
        print("- no data dir found at " + args.data_dir)
    else:
        combine_multi_dti(args.data_dir, args.first)
