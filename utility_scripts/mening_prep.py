import time
from utility_scripts.gbm_prep_func import *
import argparse


# define functions
# wrapper function for reading a complete dicom directory with/without registration to one of the images
def dcm_dir_proc(dcm_dir, param_file, reg_atlas, rep=False):
    """
    This function takes a directory containing UCSF air formatted dicom folders
    and converts all relevant files to nifti format. It also processes DTI, makes brain masks, registers the different
    series to the same space, and finally creates a 4D nifti to review all of the data.
    It will also optionally perform 3 compartment tumor segmentation, though this is a work in progress at the moment
    :param dcm_dir: the full path to the dicom containing folder as a string
    :param param_file: the full path to parameter json file
    :param reg_atlas: the full path to a registration atlas
    :param rep: boolean, repeat work or not
    :return: returns the path to a metadata file dict (as *.npy file) that contains lots of relevant info
    """

    # Pipeline step by step
    make_log(os.path.dirname(dcm_dir), rep)
    dicoms1, hdrs1, series1, dirs1 = get_series(dcm_dir)
    series_dict = make_serdict(reg_atlas, dcm_dir, param_file)
    series_dict = filter_series(dicoms1, hdrs1, series1, dirs1, series_dict)
    series_dict = dcm_list_2_niis(series_dict, dcm_dir, rep)
    # series_dict = dti_proc(series_dict, dti_index, dti_acqp, dti_bvec, dti_bval)
    series_dict = reg_series(series_dict)
    series_dict = brain_mask(series_dict)
    series_dict = bias_correct(series_dict)
    # series_dict = norm_niis(series_dict)
    # series_dict = make_nii4d(series_dict)
    # series_dict = tumor_seg(series_dict) # skipping for now... currently doing segmentation as batch at end
    series_dict = print_series_dict(series_dict)

    return series_dict


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to dicom data directory")
    parser.add_argument('--support_dir', default="support_files",
                        help="Name of support file directory relative to script path containing bvecs and atlas data")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False)
    parser.add_argument('--direc', default=None,
                        help="Optionally name a specific directory to edit")

    # get arguments and check them
    args = parser.parse_args()
    dcm_data_dir = args.data_dir
    spec_direc = args.direc
    if spec_direc:
        assert os.path.isdir(spec_direc), "Specified directory does not exist at {}".format(spec_direc)
    else:
        assert dcm_data_dir, "Must specify data directory using param --data_dir"
        assert os.path.isdir(dcm_data_dir), "Data directory not found at {}".format(dcm_data_dir)
    support_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.support_dir)
    assert os.path.isdir(support_dir), "Support directory not found at {}".format(support_dir)
    start = args.start
    end = args.end

    # check that all required files are in support directory
    my_param_file = os.path.join(support_dir, "param_files/mening.json")
    my_reg_atlas = os.path.join(support_dir, "atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz")
    # my_dti_index = os.path.join(support_dir, "DTI_files/GE_hardi_55_index.txt")
    # my_dti_acqp = os.path.join(support_dir, "DTI_files/GE_hardi_55_acqp.txt")
    # my_dti_bvec = os.path.join(support_dir, "DTI_files/GE_hardi_55.bvec")
    # my_dti_bval = os.path.join(support_dir, "DTI_files/GE_hardi_55.bval")
    for file_path in [my_param_file, my_reg_atlas]:
        assert os.path.isfile(file_path), "Required support file not found at {}".format(file_path)

    # handle specific directory
    if spec_direc:
        print("Using specific directory mode")
        dcms = [item for item in glob(spec_direc + "/*") if os.path.isdir(item)]
    else:
        print("Using parent data directory mode")
        # get a list of dicom data directories
        dcms = [item for item in glob(dcm_data_dir + "/*/*") if os.path.isdir(item)]
        # sorts on accession no
        dcms = sorted(dcms, key=lambda x: int(os.path.basename(os.path.dirname(x))))

    # handle list flag
    if args.list:
        for i, item in enumerate(dcms, 0):
            print(str(i) + ': ' + item.rsplit('/', 1)[0])  # removes dicom dir and only gives parent dir
        exit()

    # iterate through all dicom folders or just a subset/specific diectories only using options below
    if end:
        dcms = dcms[int(start):int(end) + 1]
    else:
        dcms = dcms[int(start):]
    if not isinstance(dcms, list):
        dcms = [dcms]

    # do work
    for i, dcm in enumerate(dcms, 1):
        start_t = time.time()
        serdict = dcm_dir_proc(dcm, my_param_file, my_reg_atlas)
        elapsed_t = time.time() - start_t
        print("\nCOMPLETED # " + str(i) + " of " + str(len(dcms)) + " in " + str(
            round(elapsed_t / 60, 2)) + " minute(s)\n")
