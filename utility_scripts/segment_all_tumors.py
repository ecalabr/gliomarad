""" generates automated tumor segmentations for a list of directories """

import external_software.brats17_master.test_ecalabr2 as test_ecalabr2
from glob import glob
import os
import argparse

########################## executed  as script ##########################
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--direc', default=None,
                        help="Specific directory to segment tumor from")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False,
                        help="List the directories to be processed in order")
    parser.add_argument('--bias', action="store_true", default=False,
                        help="Use bias corrected images for segmentation")

    # parse args
    args = parser.parse_args()
    start = int(args.start)
    end = args.end

    # get input directory arguments and check them
    spec_dir = args.direc
    data_dir = args.data_dir
    if spec_dir:
        spec_path = os.path.join(data_dir, spec_dir)
        assert (os.path.isdir(spec_path)), "Specific directory not found at {}".format(spec_path)
        dir_list = [spec_dir + '/']  # required trailing slash for directories?
    else:
        assert data_dir, "Must specify data directory using --data_dir"
        assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
        dir_list = sorted(glob(data_dir + "/*/"))

        # handle list argument
        if args.list:
            for i, item in enumerate(dir_list, 0):
                print(str(i) + ': ' + item)
            exit()

        # handle start and end arguments
        if end:
            dir_list = dir_list[int(start):int(end)+1]
        else:
            dir_list = dir_list[int(start):]
        if isinstance(dir_list, str):
            dir_list = [dir_list]

    # run seg
    test_ecalabr2.test(dir_list, args.bias) # currently using non-bias corrected images, change in test_ecalabr2.py
