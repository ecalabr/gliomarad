""" Uses a regex keyword to search all dicom directories in a parent directory and identify series descriptions matching
that keyword. Then provides commands to delete those folders. Meant for deleting tractography series to free up space.
Use keyword "(?i)(track|tract|left|right|arcuate|optic|ifof|slf|radiation|motor|roi)" to do so. """

import argparse
import os
from glob import glob
import pydicom as dicom
import re

########################## define functions ##########################
def id_dcmdir_keyword(dicom_dirs, keyword):
    # initialize outputs
    del_dir_list = []
    del_desc_list = []
    # loop through directories
    for dcm_dir in dicom_dirs:
        # make sure these are directories
        if os.path.isdir(dcm_dir):
            # make sure there are dicoms present
            dcms = glob(dcm_dir + '/*.dcm')
            if dcms:
                # make sure dicom file actually exists
                if os.path.isfile(dcms[0]):
                    # read the dicom header and get the series description
                    hdr = dicom.read_file(dcms[0])
                    # check for series description
                    if hasattr(hdr, 'SeriesDescription') and hdr.SeriesDescription:
                        desc = str(hdr.SeriesDescription)
                        # check for reformat tag
                        #if hasattr(hdr, 'ImageType') and 'REFORMATTED' in hdr.ImageType:
                        #    desc = desc + ' reformat'
                        # check if series description has keyword(s)
                        if re.search(keyword, desc): #and re.search('reformat', desc):
                            del_dir_list.append(dcm_dir)
                            del_desc_list.append(desc)
    # return the outputs
    print("SERIES DESCRIPTION \t\t\t DIRECTORY")
    for ind in range(len(del_dir_list)):
        print(del_desc_list[ind] + '\t\t\t' + del_dir_list[ind])
    print('\n\n')
    # print remove strings
    print("REMOVE COMMANDS")
    for ind in range(len(del_dir_list)):
        print('rm -r ' + del_dir_list[ind])
    return del_dir_list, del_desc_list

########################## executed  as script ##########################
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--key', default=None,
                        help="Regex keyword used to search series description")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False)

    # check input directory argument
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Data directory does not exist at: {}".format(args.data_dir)

    # get all subdirectories that could contain dicom directories
    data_dirs_glob = glob(args.data_dir + '/*')
    data_dirs = []
    for item in data_dirs_glob:
        if os.path.isdir(item):
            data_dirs.append(item)

    # handle list flag
    if args.list:
        for i, item in enumerate(data_dirs, 0):
            print(str(i) + ': ' + item)  # removes dicom dir and only gives parent dir
        exit()

    # handle key argument
    if not args.key:
        assert ValueError("A regex keyword must be specified using the --key argument")

    # handle start and end arguments
    if args.end:
        data_dirs = data_dirs[int(args.start):int(args.end) + 1]
    else:
        data_dirs = data_dirs[int(args.start):]
    if not isinstance(data_dirs, list):
        data_dirs = [data_dirs]

    # get the actual dicom directories
    dcm_dirs = []
    for d in data_dirs:
        dicom_direc_candidates = glob(d + '/*/*')
        if dicom_direc_candidates:
            for item in dicom_direc_candidates:
                if os.path.isdir(item):
                    dcm_dirs.append(item)

    # do work
    del_dir, del_desc = id_dcmdir_keyword(dcm_dirs, args.key)
