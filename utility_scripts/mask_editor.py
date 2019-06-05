import os
from glob import glob
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="/media/ecalabr/scratch/work/download_22",
                    help="Path to data directory")
parser.add_argument('--skip', default=0,
                    help="Index of directories to start editing at")
parser.add_argument('--mask', default="combined_brain_mask.nii.gz",
                    help="Suffix of the mask to be edited")
parser.add_argument('--anat', default="T1gad_w.nii.gz",
                    help="Suffix of the anatomy image to use for editing")

if __name__ == '__main__':
    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    mask_suffix = args.mask
    anat_suffix = args.anat

    # list all subdirs with the processed data
    direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
    direcs = sorted(direcs, key=lambda x: int(os.path.basename(x)))

    # set skip for already edited masks here
    skip = args.skip if args.skip else 0
    n_total = len(direcs)
    direcs = direcs[skip:]
    if isinstance(direcs, str):
        direcs = [direcs]

    # run ITK-snap on each one
    for i, direc in enumerate(direcs, 1):
        t1gad = glob(direc + "/*" + anat_suffix)
        mask = glob(direc + "/*" + mask_suffix)
        if t1gad and mask:
            t1gad = t1gad[0]
            mask = mask[0]
            cmd = "itksnap -g " + t1gad + " -s " + mask
            os.system(cmd)
            print("Done with study " + os.path.basename(direc) + ": " + str(i+skip) + " of " + str(n_total))
