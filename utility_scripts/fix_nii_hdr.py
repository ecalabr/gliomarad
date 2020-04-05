""" copies nii header information from one image to another """

import os
from glob import glob
import argparse
import subprocess

########################## define functions ##########################
def fix_nii_hdr(refs, targets):
    # initialize outputs
    cmds = []
    # for loop
    for ref, target in zip(refs, targets):
        cmd = 'CopyImageHeaderInformation ' + ref + ' ' + target + ' ' + target + ' 1 1 1'
        print(cmd)
        subprocess.call(cmd, shell=True)
        cmds.append(cmd)

    return cmds

########################## executed  as script ##########################
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--ref', default='T1gad_wmtb',
                        help="Reference file prefix")
    parser.add_argument('--target', default="tumor_seg",
                        help="Target file prefix to copy header info in to")

    # check input arguments
    args = parser.parse_args()
    assert args.data_dir, "No data directory specified. Use --data_dir"
    assert os.path.isdir(args.data_dir), "No data directory found at {}".format(args.data_dir)
    my_refs = glob(args.data_dir + '/*/*' + args.ref + '.nii.gz')
    my_refs.sort()
    assert my_refs, "No reference files found with prefix {}".format(args.ref)
    my_targets = glob(args.data_dir + '/*/*' + args.target + '.nii.gz')
    my_targets.sort()
    assert my_targets, "No target files found with prefix {}".format(args.targets)
    assert len(my_refs) == len(my_targets), "Reference/target number mismatch"

    # do work
    my_cmds = fix_nii_hdr(my_refs, my_targets)
