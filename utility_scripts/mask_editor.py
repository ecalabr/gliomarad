""" opens label files from a given data directory in ITK snap for manual correction """

import os
from glob import glob
import argparse

########################## define functions ##########################
def seg_edit(direcs, anat_suffix, mask_suffix, addl_suffix=None, addl_suffix2=None):

    # handle suffixes without extension
    anat_suffix = anat_suffix + '.nii.gz' if not anat_suffix.endswith('.nii.gz') else anat_suffix
    mask_suffix = mask_suffix + '.nii.gz' if not mask_suffix.endswith('.nii.gz') else mask_suffix
    addl_suffix = addl_suffix + '.nii.gz' if not addl_suffix.endswith('.nii.gz') else addl_suffix
    addl_suffix2 = addl_suffix2 + '.nii.gz' if not addl_suffix2.endswith('.nii.gz') else addl_suffix2

    # define outputs
    cmds = []

    # get number of direcs and announce
    n_total = len(direcs)
    print("Performing mask editing for a total of " + str(n_total) + " directories")

    # run ITK-snap on each one
    for ind, direc in enumerate(direcs, 1):
        anatomy = glob(direc + "/*" + anat_suffix)
        mask = glob(direc + "/*" + mask_suffix)
        if anatomy and mask and all([os.path.isfile(f) for f in [anatomy[0], mask[0]]]):
            anatomy = anatomy[0]
            mask = mask[0]
            cmd = "itksnap --geometry 1920x1080 -g " + anatomy + " -s " + mask
            addl = None
            if addl_suffix:
                addl = glob(direc + "/*" + addl_suffix)
                if not addl:
                    print("No image found with suffix {}".format(addl_suffix))
                else:
                    cmd = cmd + " -o " + addl[0]
            if addl_suffix2:
                addl2 = glob(direc + "/*" + addl_suffix2)
                if not addl2:
                    print("No image found with suffix {}".format(addl_suffix2))
                else:
                    if not addl:
                        cmd = cmd + " -o"
                    cmd = cmd + " " + addl2[0]
            #print(cmd)
            os.system(cmd)
            print("Done with study " + os.path.basename(direc) + ": " + str(ind) + " of " + str(n_total))
            cmds.append(cmd)
        else:
            print("Skipping study " + os.path.basename(direc) + ", which is missing data.")

    return cmds


########################## executed  as script ##########################
if __name__ == '__main__':

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
    parser.add_argument('--addl2', default=None,
                        help="Suffix of second additional anatomy image to use for editing")
    parser.add_argument('--direc', default=None,
                        help="Optionally name a specific directory to edit")

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

    # do work
    commands = seg_edit(my_direcs, args.anat, args.mask, addl_suffix=args.addl, addl_suffix2=args.addl2)
