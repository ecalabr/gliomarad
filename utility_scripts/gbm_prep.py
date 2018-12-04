import sys
import time
from gbm_prep_func import *

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
    make_log(os.path.dirname(dcm_dir))
    #dcm_dir = unzip_file(dcm_zip)
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
    series_dict = bias_correct(series_dict, repeat=True)
    series_dict = norm_niis(series_dict, repeat=True)
    #series_dict = make_nii4d(series_dict)
    #series_dict = tumor_seg(series_dict) # skipping for now... currently doing segmentation as batch at end
    series_dict = print_series_dict(series_dict)

    return series_dict

#######################
#######################
#######################
#######################
#######################
# outside function code


# Define global variables and required files and check that they exist
reg_atlas = "/media/ecalabr/data/support_files/atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
dti_index = "/media/ecalabr/data/support_files/DTI_files/GE_hardi_55_index.txt"
dti_acqp = "/media/ecalabr/data/support_files/DTI_files/GE_hardi_55_acqp.txt"
dti_bvec = "/media/ecalabr/data/support_files/DTI_files/GE_hardi_55.bvec"
dti_bval = "/media/ecalabr/data/support_files/DTI_files/GE_hardi_55.bval"
for file_path in [reg_atlas, dti_index, dti_acqp, dti_bvec, dti_bval]:
    if not os.path.isfile(file_path):
        sys.exit("Could not find required file: " + file_path)

# Define dicom directory and get a list of zip files from a dicom zip folder
dcm_data_dir = "/media/ecalabr/data/qc_complete/"
dcms = [item for item in glob(dcm_data_dir + "/*/*") if os.path.isdir(item)]
dcms = sorted(dcms, key=lambda x: int(os.path.basename(os.path.dirname(x))))  # sorts on accession no

# iterate through all dicom folders or just a subset/specific diectories only using options below
#dcms = dcms[:61] #[46:]  #0:36 done
dcms = ["/media/ecalabr/data/qc_complete/10672000/1.2.124.113532.80.22017.45499.20151103.80830.293969749"]

if not isinstance(dcms, list):
    dcms = [dcms]
for i, dcm in enumerate(dcms, 1):
    start_t = time.time()
    serdict = read_dicom_dir(dcm)
    elapsed_t = time.time() - start_t
    print("\nCOMPLETED # "+str(i)+" of "+str(len(dcms))+" in "+str(round(elapsed_t/60, 2))+" minute(s)\n")