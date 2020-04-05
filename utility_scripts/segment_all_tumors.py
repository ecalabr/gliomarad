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
    parser.add_argument('--skip', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--spec_dir', default=None,
                        help="Specific directory to segment tumor from")

    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    assert data_dir, "Must specify data directory using --data_dir"
    assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    start = int(args.skip)
    spec_dir = args.spec_dir
    if spec_dir:
        spec_path = os.path.join(data_dir, spec_dir)
        assert(os.path.isdir(spec_path)), "Specific directory not found at {}".format(spec_path)

    # define dir_list
    dir_list = glob(data_dir + "/*/")
    dir_list = dir_list[start:]
    if isinstance(dir_list, str):
        dir_list = [dir_list]

    # handle specific directory argument
    if spec_dir:
        dir_list = [spec_dir + '/']  # required trailing slash for directories?

    # run seg
    test_ecalabr2.test(dir_list)
