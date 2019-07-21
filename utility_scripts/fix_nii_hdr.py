import os
from glob import glob
import argparse
import subprocess

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/media/ecalabr/scratch/qc_complete',
                    help="Path to data directory")
parser.add_argument('--ref', default='T1gad_wmtb',
                    help="Reference file prefix")
parser.add_argument('--target', default="tumor_seg",
                    help="Target file prefix to copy header info in to")

if __name__ == '__main__':

    # check input arguments
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "No data directory found at {}".format(args.data_dir)
    refs = glob(args.data_dir + '/*/*' + args.ref + '.nii.gz')
    refs.sort()
    assert refs, "No reference files found with prefix {}".format(args.ref)
    targets = glob(args.data_dir + '/*/*' + args.target + '.nii.gz')
    targets.sort()
    assert targets, "No target files found with prefix {}".format(args.targets)
    assert len(refs) == len(targets), "Reference/target number mismatch"

    # for loop
    for ref, target in zip(refs, targets):
        cmd = 'CopyImageHeaderInformation ' + ref + ' ' + target + ' ' + target + ' 1 1 1'
        print(cmd)
        subprocess.call(cmd, shell=True)