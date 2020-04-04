""" opens label files from a given data directory in ITK snap for manual correction """

import os
from glob import glob
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=None,
                    help="Path to data directory")
parser.add_argument('--start', default=0,
                    help="Index of directories to start processing at")
parser.add_argument('--end', default=None,
                    help="Index of directories to end processing at")
parser.add_argument('--list', action="store_true", default=False,
                    help="List all directories and exit")
parser.add_argument('--mask', default="combined_brain_mask.nii.gz",
                    help="Suffix of the mask to be edited")
parser.add_argument('--anat', default="T1gad_w.nii.gz",
                    help="Suffix of the anatomy image to use for editing")
parser.add_argument('--addl', default="FLAIR_w.nii.gz",
                    help="Suffix of additional anatomy image to use for editing")
parser.add_argument('--direc', default=None,
                    help="Optionally name a specific directory to edit")

if __name__ == '__main__':
    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    spec_direc = args.direc
    if spec_direc:
        assert os.path.isdir(spec_direc), "Specified directory does not exist at {}".format(spec_direc)
    else:
        assert data_dir, "Must specify data directory using param --data_dir"
        assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    mask_suffix = args.mask
    anat_suffix = args.anat
    addl_suffix = args.addl

    start = args.start
    end = args.end

    # handle specific directory
    if spec_direc:
        direcs = [spec_direc]
    else:
        # list all subdirs with the processed data
        direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
        direcs = sorted(direcs, key=lambda x: int(os.path.basename(x)))

        # set start and stop for subset/specific diectories only using options below
        if end:
            direcs = direcs[int(start):int(end) + 1]
        else:
            direcs = direcs[int(start):]

    # get number of direcs and announce
    n_total = len(direcs)
    print("Performing mask editing for a total of " + str(n_total) + " directories")
    if isinstance(direcs, str):
        direcs = [direcs]

    # handle list flag
    if args.list:
        for i, item in enumerate(direcs, 0):
            print(str(i) + ': ' + item)
        exit()

    # run ITK-snap on each one
    for i, direc in enumerate(direcs, 1):
        t1gad = glob(direc + "/*" + anat_suffix)
        mask = glob(direc + "/*" + mask_suffix)
        if t1gad and mask:
            t1gad = t1gad[0]
            mask = mask[0]
            cmd = "itksnap --geometry 1920x1080 -g " + t1gad + " -s " + mask
            if addl_suffix:
                addl = glob(direc + "/*" + addl_suffix)[0]
                cmd = cmd + " -o " + addl
            #print(cmd)
            os.system(cmd)
            print("Done with study " + os.path.basename(direc) + ": " + str(i) + " of " + str(n_total))
