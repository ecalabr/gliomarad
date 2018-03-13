import sys
from gbm_prep_func import *

# wrapper function for reading a complete dicom directory with/without registration to one of the images
def read_dicom_dir(dcm_zip, rep=False):
    """
    This function takes a DICOM directory for a single patient study, and converts all appropriate series to NIFTI.
    It will also register any/all datasets to a given target.
    :param dcm_zip: the full path to the dicom zipfile as a string
    :param rep: boolean, repeat work or not
    :return: returns the path to a metadata file dict that contains lots of relevant info
    Note this forces axial plane image orientation. To change this, edit the filter_series function
    """

    # Pipeline step by step
    make_log(os.path.splitext(dcm_zip)[0])
    dcm_dir = unzip_file(dcm_zip)
    dicoms1, hdrs1, series1, dirs1 = get_series(dcm_dir)
    series_dict = make_serdict(reg_atlas, dcm_dir)
    series_dict = filter_series(dicoms1, hdrs1, series1, dirs1, series_dict)
    series_dict = dcm_list_2_niis(series_dict, dcm_dir, rep)
    series_dict = split_dwi(series_dict)
    series_dict = split_asl(series_dict)
    series_dict = dti_proc(series_dict, dti_index, dti_acqp, dti_bvec, dti_bval)
    series_dict = reg_series(series_dict)
    series_dict = split_dwi(series_dict)
    series_dict = brain_mask(series_dict)
    series_dict = norm_niis(series_dict)
    series_dict = make_nii4d(series_dict)
    series_dict = tumor_seg(series_dict)
    np.save(os.path.join(os.path.dirname(dcm_dir), series_dict["info"]["id"] + "_metadata.npy"), series_dict)

    return series_dict

#######################
#######################
#######################
#######################
#######################
# outside function code

# Define global variables and required files and check that they exist
reg_atlas = "/Users/edc15/Desktop/strokecode/atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii"
dti_index = "/Users/edc15/Desktop/gbm/gbm_data/DTI_files/GE_hardi_55_index.txt"
dti_acqp = "/Users/edc15/Desktop/gbm/gbm_data/DTI_files/GE_hardi_55_acqp.txt"
dti_bvec = "/Users/edc15/Desktop/gbm/gbm_data/DTI_files/GE_hardi_55.bvec"
dti_bval = "/Users/edc15/Desktop/gbm/gbm_data/DTI_files/GE_hardi_55.bval"
for file_path in [reg_atlas, dti_index, dti_acqp, dti_bvec, dti_bval]:
    if not os.path.isfile(file_path):
        sys.exit("Could not find required file: " + file_path)

# Define dicom zip directory and get a list of zip files from a dicom zip folder
dcm_zip_dir = "/Users/edc15/Desktop/gbm/gbm_data/"
zip_dcm = glob(dcm_zip_dir + "*.zip")

# iterate through all dicom directories
for dcmz in [zip_dcm[0]]:# #[zip_dcm[0]]:#
    serdict = read_dicom_dir(dcmz)