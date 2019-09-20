import time
from gbm_prep_func import *
import argparse

# wrapper function for reading a complete dicom directory with/without registration to one of the images
def read_dicom_dir(dcm_dir, rep=False):
    """
    This function takes a directory containing UCSF air formatted dicom folders
    and converts all relevant files to nifti format. It also processes DTI, makes brain masks, registers the different
    series to the same space, and finally creates a 4D nifti to review all of the data.
    It will also perform 3 compartment tumor segmentation, though this is a work in progress at the moment
    :param dcm_dir: the full path to the dicom containing folder as a string
    :param rep: boolean, repeat work or not
    :return: returns the path to a metadata file dict (as *.npy file) that contains lots of relevant info
    Currently this does not force any specific image orientation. To change this, edit the filter_series function
    """

    # Pipeline step by step
    make_log(os.path.dirname(dcm_dir), rep)
    #dcm_dir = unzip_file(dcm_zip)
    dicoms1, hdrs1, series1, dirs1 = get_series(dcm_dir)
    series_dict = make_serdict(reg_atlas, dcm_dir, param_file)
    series_dict = filter_series(dicoms1, hdrs1, series1, dirs1, series_dict)
    series_dict = dcm_list_2_niis(series_dict, dcm_dir, rep)
    series_dict = reg_series(series_dict)
    series_dict = bias_correct(series_dict)
    #series_dict = print_series_dict(series_dict)

    return series_dict

#######################
#######################
#######################
#######################
#######################

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dcm_dir', default="/media/ecalabr/data2/breast_work/",
                    help="Path to dicom data directory")
parser.add_argument('--support_dir', default="/media/ecalabr/data1/mammo_data/support_files",
                    help="Path to support file directory containing bvecs and atlas data")
parser.add_argument('--start', default=0,
                    help="Index of directories to start processing at")
parser.add_argument('--end', default=None,
                    help="Index of directories to end processing at")
parser.add_argument('--redo', default=False,
                    help="Repeat work boolean")

if __name__ == '__main__':

    # get arguments and check them
    args = parser.parse_args()
    dcm_data_dir = args.dcm_dir
    assert os.path.isdir(dcm_data_dir), "Data directory not found at {}".format(dcm_data_dir)
    support_dir = args.support_dir
    assert os.path.isdir(support_dir), "Support directory not found at {}".format(support_dir)
    start = args.start
    end = args.end
    redo = args.redo

    # check that all required files are in support directory
    param_file = os.path.join(support_dir, "param_files/mammo.json")
    reg_atlas = os.path.join(support_dir, "atlases/breast_test_atlas.nii.gz")
    for file_path in [param_file, reg_atlas]:
        assert os.path.isfile(file_path), "Required support file not found at {}".format(file_path)

    # get a list of zip files from a dicom zip folder
    dcms = [item for item in glob(dcm_data_dir + "/*/*") if os.path.isdir(item)]
    dcms = sorted(dcms, key=lambda x: int(os.path.basename(os.path.dirname(x))))  # sorts on accession no

    # iterate through all dicom folders or just a subset/specific diectories only using options below
    if end:
        dcms = dcms[int(start):int(end)]
    else:
        dcms = dcms[int(start):]

    if not isinstance(dcms, list):
        dcms = [dcms]
    for i, dcm in enumerate(dcms, 1):
        start_t = time.time()
        serdict = read_dicom_dir(dcm, rep=redo)
        elapsed_t = time.time() - start_t
        print("\nCOMPLETED # "+str(i)+" of "+str(len(dcms))+" in "+str(round(elapsed_t/60, 2))+" minute(s)\n")