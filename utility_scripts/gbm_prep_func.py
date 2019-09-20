import os
import pydicom as dicom
import nibabel as nib
import numpy as np
from glob import glob
import subprocess
import logging
import yaml
import multiprocessing
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.fsl import BET
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl import Eddy
from nipype.interfaces.fsl import DTIFit
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.fsl.utils import CopyGeom
from nipype.interfaces.fsl import Merge
from nipype.interfaces.ants import N4BiasFieldCorrection
import external_software.brats17_master.test_ecalabr as test_ecalabr
import json

### Set up logging
# takes work dir
# returns logger
def make_log(work_dir, repeat=False):
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    # make log file, append to existing
    idno = os.path.basename(work_dir)
    log_file = os.path.join(work_dir, idno + "_log.txt")
    if repeat:
        open(log_file, 'w').close()
    else:
        open(log_file, 'a').close()
    # make logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG) # should this be DEBUG?
    # set all existing handlers to null to prevent duplication
    logger.handlers = []
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterch = logging.Formatter('%(message)s')
    formatterfh = logging.Formatter("[%(asctime)s]  [%(levelname)s]:     %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatterch)
    fh.setFormatter(formatterfh)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("####################### STARTING NEW LOG #######################")

# first check for series_dict to find the appropriate image series from the dicom folder
# matching strings format is [[strs to match AND], OR [strs to match AND]
# for NOT strings that are in all caps, the algorithm will ensure the series does not start with that string
def make_serdict(reg_atlas, dcm_dir, params):

    # load json file
    assert os.path.isfile(params), "Param file does not exist at " + params
    with open(params, 'r') as f:
        json_str = f.read()
    sdict = json.loads(json_str)

    # handle atlas option for registration target
    for key in sdict:
        if 'reg_target' in sdict[key].keys():
            if sdict[key]['reg_target'] == 'atlas':
                sdict[key].update({'reg_target': reg_atlas})

    # add info section to series dict
    sdict.update({"info":{
        "filename": "None",
        "dcmdir": dcm_dir,
        "id": os.path.basename(os.path.dirname(dcm_dir)),
    }})

    return sdict

# unzip dicom directory
def unzip_file(dicom_zip):
    # logging
    logger = logging.getLogger("my_logger")
    # unzip and get out_dir, tmp_dir, dicom_dir
    acc_no = dicom_zip.rsplit("/", 1)[1].split(".", 1)[0]
    tmp_dir = os.path.join(os.path.dirname(dicom_zip), acc_no)
    logger.info("UNZIPPING:")
    logger.info("- " + dicom_zip + ". If files are already unzipped, work will not be repeated.")
    unz_cmd = "unzip -n -qq -d " + tmp_dir + " " + dicom_zip
    _ = subprocess.call(unz_cmd, shell=True)
    dicomdir = glob(tmp_dir + "/*/")
    dicomdir = dicomdir[0].rsplit("/", 1)[0]  # must remove trailing slash so that os.path.dirname returns one dir up
    # print some stating info
    logger.info("- working directory = " + os.path.dirname(dicomdir))
    return dicomdir

# function to get complete series list from dicom directory
def get_series(dicom_dir):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("GETTING SERIES LIST:")
    logger.info("- DICOM directory = " + dicom_dir)
    # indo prep
    idno = os.path.basename(os.path.dirname(dicom_dir))
    # define variables
    dicoms = []
    hdrs = []
    series = []

    # list directories
    dirs = [os.path.join(dicom_dir, o) for o in os.listdir(dicom_dir)
            if os.path.isdir(os.path.join(dicom_dir, o))]

    # get lists of all dicom series
    for ind, direc in enumerate(dirs):
        dicoms.append(glob(direc + "/*.dcm"))
        hdrs.append(dicom.read_file(dicoms[ind][0]))
        try:
            series.append(hdrs[ind].SeriesDescription.decode('utf8') + " [dir=" + direc[-5:] + "]")
        except Exception:
            logger.info("- Skipping series " + str(ind + 1) + " without series description")
            series.append("none"  + " [dir=" + direc[-5:] + "]")

    # save series list
    series_file = os.path.join(os.path.dirname(dicom_dir), idno + "_series_list.txt")
    if not os.path.isfile(series_file):
        fileout = open(series_file, 'w')
        for ind, item in enumerate(series):
            fileout.write("%s" % item)
            nspaces = 75 - len(item) # assume no series description longer than 75
            fileout.write("%s" % " " * nspaces)
            fileout.write("%s" % "\tslthick=")
            try:
                fileout.write("%s" % str(hdrs[ind].SliceThickness).rstrip("0").rstrip("."))
            except Exception:
                pass
            fileout.write("%s" % "\tacqmtx=")
            try:
                fileout.write("%s" % str(hdrs[ind].AcquisitionMatrix))
            except Exception:
                pass
            fileout.write("%s" % "\trowcol=")
            try:
                fileout.write("%s" % str(hdrs[ind].Rows) + "x" + str(hdrs[ind].Columns))
            except Exception:
                pass
            fileout.write("%s" % "\tslices=")
            try:
                fileout.write("%s" % str(hdrs[ind].ImagesInAcquisition))
            except Exception:
                fileout.write("%s" % "None" )
            fileout.write("%s" % "\tacqtime=")
            try:
                fileout.write("%s" % str(hdrs[ind].AcquisitionTime))
            except Exception:
                pass
            fileout.write("%s" % "\n")

    return dicoms, hdrs, series, dirs

# first define function to get filter substring matches by criteria
def substr_list(strings, substrs, substrnot):
    # logging
    logger = logging.getLogger("my_logger")
    # define variables
    inds = []
    # determine if any not strings are all caps, if  so treat them as "does not start with this string"
    substrnotstart = [x.lower() for x in substrnot if x.isupper()]
    # remove all caps (not starts with) substrs from the substrnot list
    substrnot = [x for x in substrnot if not x.isupper()]
    # for each input series description
    number = 1
    for ind, string in enumerate(strings, 0):
        match = False
        # for each substr list in substrs
        for substrlist in substrs:
            # if all substrs match and no NOT substrs match then append to matches
            # also, if there are any all caps NOT substrs, make sure the string does not start with that substr
            if all(ss.lower() in string.lower() for ss in substrlist) \
                    and not any(ssn.lower() in string.lower() for ssn in substrnot) \
                    and not any(string.lower().startswith(ssns) for ssns in substrnotstart):
                inds.append(ind)
                match = True
        if match:
            logger.info("- matched series: " + string + " (" + str(number) + ")")
            number = number + 1 # step the number of matching series
    # return only unique inds if inds is not empty
    if inds:
        inds = list(set(inds)) # returns only unique values for inds
    return inds

# get filtered series list
def filter_series(dicoms, hdrs, series, dirs, srs_dict):
    # logging
    logger = logging.getLogger("my_logger")
    # define variables
    new_dicoms = []
    new_hdrs = []
    new_series = []
    new_dirs = []
    # for each output, find match and append to new list for conversion
    keeper = []
    number = 1 # series number chosen to report in the logger info
    for srs in srs_dict:
        # only search if terms are provided
        if "or" in srs_dict[srs].keys() and "not" in srs_dict[srs].keys():
            logger.info("FINDING SERIES: " + srs)
            inds = substr_list(series, srs_dict[srs]["or"], srs_dict[srs]["not"]) # calls above function
            # if there are inds of matches, pick the first match by default and check for more slices or repeat series
            if inds:  # if only 1 ind, then use it
                if len(inds) == 1:
                    inds = inds[0]
                else:  # if more than 1 inds, try to find the one with the most slices, otherwise just pick first one
                    for i in inds:
                        if inds.index(i) == 0: # pick first ind by default (this method works bc inds is unique set)
                            keeper = i
                        if hasattr(hdrs[i], "ImagesInAcquisition") and hasattr(hdrs[keeper], "ImagesInAcquisition"):
                            # if another series has more images, pick it instead
                            if int(hdrs[i].ImagesInAcquisition) > int(hdrs[keeper].ImagesInAcquisition):
                                keeper = i
                            # if another series has the same # images:
                            elif int(hdrs[i].ImagesInAcquisition) == int(hdrs[keeper].ImagesInAcquisition):
                                # and was acquired later, keep it instead
                                if hasattr(hdrs[i], "AcquisitionTime") and hasattr(hdrs[keeper], "AcquisitionTime"):
                                    if int(hdrs[i].AcquisitionTime) > int(hdrs[keeper].AcquisitionTime):
                                        keeper = i
                                # and description contains "repeat" or "redo", keep it instead
                                if any(example in series[i] for example in ["repeat", "redo"]):
                                    keeper = i
                    number = inds.index(keeper) + 1 # number of index chosen (starting at 1 for index 0)
                    inds = keeper # replace inds with just the keeper index
            if inds or inds == 0:  # this handles 0 index matches
                logger.info("- keeping series: " + series[inds] + " (" + str(number) + ")")
                new_dicoms.append(dicoms[inds])
                srs_dict[srs].update({"dicoms": dicoms[inds]})
                new_hdrs.append(hdrs[inds])
                srs_dict[srs].update({"hdrs": hdrs[inds]})
                new_series.append(series[inds])
                srs_dict[srs].update({"series": series[inds]})
                new_dirs.append(dirs[inds])
                srs_dict[srs].update({"dirs": dirs[inds]})
            else:
                logger.info("- no matching series found!")
                new_dicoms.append([])
                srs_dict[srs].update({"dicoms": []})
                new_hdrs.append([])
                srs_dict[srs].update({"hdrs": []})
                new_series.append("None")
                srs_dict[srs].update({"series": "None"})
                new_dirs.append("None")
                srs_dict[srs].update({"dirs": []})
    return srs_dict

# define function to convert selected dicoms
def dcm_list_2_niis(strs_dict, dicom_dir, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # id prep
    idno = strs_dict["info"]["id"]
    # allocate vars
    output_ser = []
    logger.info("CONVERTING FILES:")
    # basic converter initiation
    converter = Dcm2niix()
    converter.inputs.bids_format = False
    converter.inputs.single_file = True
    converter.inputs.args = '-w 1'
    converter.inputs.compress = "y"
    converter.inputs.output_dir = os.path.dirname(dicom_dir)
    converter.terminal_output = "allatonce"
    converter.anonymize = True

    # convert all subdirectories from dicom to nii
    for series in strs_dict:
        if "dicoms" in strs_dict[series].keys():
            converter.inputs.source_names = strs_dict[series]["dicoms"]
            # series_string = strs_dict[series]["series"]
            # outfilename = series_string.replace(" ", "_").lower().replace("(", "").replace(")", "").replace("/", "").replace(":", "").replace("\"","")
            outfilename = idno + "_" + series
            converter.inputs.out_filename = outfilename
            outfilepath = os.path.join(os.path.dirname(dicom_dir), outfilename + ".nii.gz")
            # don't repeat conversion if already done
            if os.path.isfile(outfilepath) and repeat:
                logger.info("- " + outfilename + " already exists, but repeat is True, so it will be overwritten")
                logger.info("- Converting " + outfilename)
                logger.debug(converter.cmdline)
                result = converter.run()
                # make sure that file wasnt named something else during conversion, if so, rename to expected name
                if isinstance(result.outputs.converted_files, list):
                    converted = result.outputs.converted_files[0]
                else:
                    converted = result.outputs.converted_files
                if not converted == outfilepath:
                    logger.info("- converted file is named " + converted + ", renaming " + outfilepath)
                    os.rename(converted, outfilepath)
            if not os.path.isfile(outfilepath) and not strs_dict[series]["series"] == "None":
                logger.info("- Converting " + outfilename)
                logger.debug(converter.cmdline)
                result = converter.run()
                # make sure that file wasnt named something else during conversion
                if isinstance(result.outputs.converted_files, list):
                    converted = result.outputs.converted_files[0]
                else:
                    converted = result.outputs.converted_files
                if not converted == outfilepath:
                    logger.info("- " + series + " converted file is named " + converted + ", renaming " + outfilepath)
                    os.rename(converted, outfilepath)
            if os.path.isfile(outfilepath) and not repeat:
                logger.info("- " + outfilename + " already exists and will not be overwritten")
            if not os.path.isfile(outfilepath) and strs_dict[series]["series"] == "None":
                logger.info("- No existing file and no matching series found: " + series)
            # after running through conversion options, check if file actually exists
            if os.path.isfile(outfilepath):
                # if output file exists, regardless of whether created or not append name to outfile list
                output_ser.append(outfilepath)
                strs_dict[series].update({"filename": outfilepath})
                # handle options for post-coversion nifti processing here (only if output file exists)
                if "split" in strs_dict[series].keys():
                    logger.info("- Splitting " + outfilename + " per specified parameters")
                    outnames = split_multiphase(outfilepath, strs_dict[series]['split'], repeat=False)
                    if outnames:
                        for k in outnames.keys():
                            # if series does not already exist in series list then it will not be updated
                            if k in strs_dict.keys():
                                strs_dict[k].update({"filename": outnames[k]})
    # print outputs of file conversion
    logger.info("CONVERTED FILES LIST:")
    for ser in strs_dict:
        if "dicoms" in strs_dict[ser] and "filename" in strs_dict[ser] and os.path.isfile(strs_dict[ser]["filename"]):
            logger.info("- " + ser + " = " + strs_dict[ser]["filename"])
    return strs_dict


# Split multiphase
# takes a multiphase (or other 4D) nifti and splits into one or more additional series based on options
def split_multiphase(nii_in, options, repeat=False):
    # setup return variable
    outnames = {}
    # logging
    logger = logging.getLogger("my_logger")
    # path prep
    basepath = nii_in.rsplit('_', 1)[1]
    # first check if all desired outputs already exist, if so, don't do any work
    if not repeat and all([os.path.isfile(os.path.join(basepath, ser + '.nii.gz')) for ser in options.keys()]):
        logger.info("- All specified split outputs already exist and will not be regenerated")
        for k in options.keys():
            outnames.update({k: os.path.join(basepath, k + '.nii.gz')})
        return outnames
    else:
        # data loading
        nii = nib.load(nii_in)
        data = nii.get_data()
        # if data is not 4D, then return
        if len(data.shape) < 4 or data.shape[3] < 2:
            logger.info("- Split option was specified, but data is not 4D")
            return outnames
        # loop through splitting options - THIS WILL OVERWRITE OTHER SERIES
        for k in options.keys():
            dirname = os.path.dirname(nii_in)
            basename = os.path.basename(nii_in)
            orig_ser = basename.split('_')[1].split('.')[0]
            outname = os.path.join(dirname, basename.split('_')[0] + '_' + k + '.nii.gz')
            logger.info("- Splitting " + orig_ser + " phase " + str(options[k]) + " as " + k)
            new_data = data[:, :, :, options[k]]
            new_nii = nib.Nifti1Image(new_data, nii.affine)
            nib.save(new_nii, outname)
            outnames.update({k: outname})
        return outnames

# Fast ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def affine_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args='--float'
    antsreg.inputs.fixed_image=template_nii
    antsreg.inputs.moving_image=moving_nii
    antsreg.inputs.output_transform_prefix=outprefix
    antsreg.inputs.num_threads=multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas=[[6, 4, 1, 0]] * 2
    antsreg.inputs.sigma_units=['vox'] * 2
    antsreg.inputs.transforms=['Rigid', 'Affine']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.metric=['Mattes', 'Mattes']  # ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight=[1.0] * 2
    antsreg.inputs.number_of_iterations=[[1000, 1000, 1000, 1000]] * 2  # [100, 70, 50, 20]
    antsreg.inputs.convergence_threshold=[1e-07, 1e-07]
    antsreg.inputs.convergence_window_size=[10, 10]
    antsreg.inputs.radius_or_number_of_bins=[32, 32]  # 4
    antsreg.inputs.sampling_strategy=['Regular', 'Regular']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25, 0.25]  # 1
    antsreg.inputs.shrink_factors=[[4, 3, 2, 1]] * 2  # *3
    antsreg.inputs.transform_parameters=[(0.1,), (0.1,)]  # (0.1, 3.0, 0.0) # affine gradient step

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Faster ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def fast_affine_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args='--float'
    antsreg.inputs.fixed_image=template_nii
    antsreg.inputs.moving_image=moving_nii
    antsreg.inputs.output_transform_prefix=outprefix
    antsreg.inputs.num_threads=multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas=[[6, 4, 1]] * 2
    antsreg.inputs.sigma_units=['mm'] * 2
    antsreg.inputs.transforms=['Rigid', 'Affine']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.metric=['Mattes', 'Mattes']  # ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight=[1.0] * 2
    antsreg.inputs.number_of_iterations=[[1000, 1000, 1000]] * 2  # [100, 70, 50, 20]
    antsreg.inputs.convergence_threshold=[1e-04, 1e-04]
    antsreg.inputs.convergence_window_size=[5, 5]
    antsreg.inputs.radius_or_number_of_bins=[32, 32]  # 4
    antsreg.inputs.sampling_strategy=['Regular', 'Regular']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25, 0.25]  # 1
    antsreg.inputs.shrink_factors=[[6, 4, 2]] * 2  # *3
    antsreg.inputs.transform_parameters=[(0.1,), (0.1,)]  # (0.1, 3.0, 0.0) # affine gradient step

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Fast ants diffeomorphic registration
# takes moving and template niis and a work dir
# performs fast diffeomorphic registration and returns a list of transforms
def diffeo_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args='--float'
    antsreg.inputs.fixed_image=template_nii
    antsreg.inputs.moving_image=moving_nii
    antsreg.inputs.output_transform_prefix=outprefix
    antsreg.inputs.num_threads=multiprocessing.cpu_count()
    antsreg.terminal_output='none'
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.shrink_factors=[[8, 4, 2, 1], [4, 2, 1]]
    antsreg.inputs.smoothing_sigmas=[[4, 2, 1, 0], [2, 1, 0]]
    antsreg.inputs.sigma_units=['vox', 'mm']
    antsreg.inputs.transforms=['Affine', 'SyN']
    antsreg.inputs.use_histogram_matching=[True, True]
    antsreg.inputs.write_composite_transform=True
    antsreg.inputs.metric=['MI', 'Mattes']
    antsreg.inputs.metric_weight=[1.0, 1.0]
    antsreg.inputs.number_of_iterations=[[1000, 500, 250, 50], [50, 50, 25]]
    antsreg.inputs.convergence_threshold=[1e-07, 1e-07]
    antsreg.inputs.convergence_window_size=[5, 5]
    antsreg.inputs.radius_or_number_of_bins=[32, 32]
    antsreg.inputs.sampling_strategy=['Regular', 'None']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25, 1]
    antsreg.inputs.transform_parameters=[(0.1,), (0.1, 3.0, 0.0)]

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Faster ants diffeomorphic registration
# takes moving and template niis and a work dir
# performs fast diffeomorphic registration and returns a list of transforms
def fast_diffeo_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args='--float'
    antsreg.inputs.fixed_image=template_nii
    antsreg.inputs.moving_image=moving_nii
    antsreg.inputs.output_transform_prefix=outprefix
    antsreg.inputs.num_threads=multiprocessing.cpu_count()
    antsreg.terminal_output='none'
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.shrink_factors=[[6, 4, 2], [4, 2]]
    antsreg.inputs.smoothing_sigmas=[[4, 2, 1], [2, 1]]
    antsreg.inputs.sigma_units=['mm', 'mm']
    antsreg.inputs.transforms=['Affine', 'SyN']
    antsreg.inputs.use_histogram_matching=[True, True]
    antsreg.inputs.write_composite_transform=True
    antsreg.inputs.metric=['Mattes', 'Mattes']
    antsreg.inputs.metric_weight=[1.0, 1.0]
    antsreg.inputs.number_of_iterations=[[1000, 500, 250], [50, 50]]
    antsreg.inputs.convergence_threshold=[1e-05, 1e-05]
    antsreg.inputs.convergence_window_size=[5, 5]
    antsreg.inputs.radius_or_number_of_bins=[32, 32]
    antsreg.inputs.sampling_strategy=['Regular', 'None']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25, 1]
    antsreg.inputs.transform_parameters=[(0.1,), (0.1, 3.0, 0.0)]

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# ANTS translation
# takes moving and template niis and a work dir
# performs fast translation only registration and returns a list of transforms
def trans_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args='--float'
    antsreg.inputs.fixed_image=template_nii
    antsreg.inputs.moving_image=moving_nii
    antsreg.inputs.output_transform_prefix=outprefix
    antsreg.inputs.num_threads=multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas=[[6, 4, 1, 0]]
    antsreg.inputs.sigma_units=['vox']
    antsreg.inputs.transforms=['Translation']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.metric=['Mattes']  # ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight=[1.0]
    antsreg.inputs.number_of_iterations=[[1000, 500, 250, 50]]  # [100, 70, 50, 20]
    antsreg.inputs.convergence_threshold=[1e-07]
    antsreg.inputs.convergence_window_size=[10]
    antsreg.inputs.radius_or_number_of_bins=[32]  # 4
    antsreg.inputs.sampling_strategy=['Regular']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25]  # 1
    antsreg.inputs.shrink_factors=[[4, 3, 2, 1]]  # *3
    antsreg.inputs.transform_parameters=[(0.1,)]  # (0.1, 3.0, 0.0) # affine gradient step

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm

# Ants apply transforms to list
# takes moving and reference niis, an output filename, plus a transform list
# applys transform and saves output as output_nii
def ants_apply(moving_nii, reference_nii, interp, transform_list, work_dir, invert_bool=False, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # enforce list
    if not isinstance(moving_nii, list):
        moving_nii = [moving_nii]
    if not isinstance(transform_list, list):
        transform_list = [transform_list]
    # create output list of same shape
    output_nii = moving_nii
    # define extension
    ext = ".nii"
    # for loop for applying reg
    for ind, mvng in enumerate(moving_nii, 0):
        # define output path
        output_nii[ind] = os.path.join(work_dir, os.path.basename(mvng).split(ext)[0] + '_w.nii.gz')
        # do registration if not already done
        antsapply = ApplyTransforms()
        antsapply.inputs.dimension=3
        antsapply.terminal_output='none'  # suppress terminal output
        antsapply.inputs.input_image=mvng
        antsapply.inputs.reference_image=reference_nii
        antsapply.inputs.output_image=output_nii[ind]
        antsapply.inputs.interpolation=interp
        antsapply.inputs.default_value=0
        antsapply.inputs.transforms=transform_list
        antsapply.inputs.invert_transform_flags=[invert_bool] * len(transform_list)
        if not os.path.isfile(output_nii[ind]) or repeat:
            logger.info("- Creating warped image " + output_nii[ind])
            logger.debug(antsapply.cmdline)
            antsapply.run()
        else:
            logger.info("- Transformed image already exists at " + output_nii[ind])
            logger.debug(antsapply.cmdline)
    # if only 1 label, don't return array
    if len(output_nii) == 1:
        output_nii = output_nii[0]
    return output_nii

# register data together using reg_target as target, if its a file, use it
# if not assume its a dict key for an already registered file
# there are multiple loops here because dicts dont preserve order, and we need it for some registration steps
def reg_series(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("REGISTERING IMAGES:")
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    # if reg is false, or if there is no input file found, then just make the reg filename same as unreg filename
    for ser in ser_dict:
        # first, if there is no filename, set to None
        if not "filename" in ser_dict[ser].keys():
            ser_dict[ser].update({"filename": "None"})
        if ser_dict[ser]["filename"] == "None" or not "reg" in ser_dict[ser].keys() or not ser_dict[ser]["reg"]:
            ser_dict[ser].update({"filename_reg": ser_dict[ser]["filename"]})
            ser_dict[ser].update({"transform": "None"})
            ser_dict[ser].update({"reg": False})
        # if reg True, then do the registration using translation, affine, nonlin, or just applying existing transform
    # handle translation registration
    for ser in ser_dict:
        if ser_dict[ser]["reg"] == "trans":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            # handle registration options here
            if "reg_option" in ser_dict[ser].keys():
                option = ser_dict[ser]["reg_option"]
            else:
                option = None
            transforms = trans_reg(movingr, template, os.path.dirname(dcm_dir), option, repeat)
            niiout = ants_apply(movinga, template, 'Linear', transforms, os.path.dirname(dcm_dir), repeat)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms})
    # handle affine registration
    for ser in ser_dict:
        if ser_dict[ser]["reg"] == "affine":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template): # make sure template and moving files exist
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = affine_reg(movingr, template, os.path.dirname(dcm_dir), option, repeat)
                niiout = ants_apply(movinga, template, 'Linear', transforms, os.path.dirname(dcm_dir), repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle faster affine registration
    for ser in ser_dict:
        if ser_dict[ser]["reg"] == "fast_affine":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template):  # make sure template and moving files exist
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = fast_affine_reg(movingr, template, os.path.dirname(dcm_dir), option, repeat)
                niiout = ants_apply(movinga, template, 'Linear', transforms, os.path.dirname(dcm_dir), repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle diffeo registration
    for ser in ser_dict:
        if ser_dict[ser]["reg"] == "diffeo":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template): # check that all files exist prior to reg
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = diffeo_reg(movingr, template, os.path.dirname(dcm_dir), option, repeat)
                niiout = ants_apply(movinga, template, 'Linear', transforms, os.path.dirname(dcm_dir), repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle faster diffeo registration
    for ser in ser_dict:
        if ser_dict[ser]["reg"] == "fast_diffeo":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template):  # check that all files exist prior to reg
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = fast_diffeo_reg(movingr, template, os.path.dirname(dcm_dir), option, repeat)
                niiout = ants_apply(movinga, template, 'Linear', transforms, os.path.dirname(dcm_dir), repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle applying an existing transform (assumes reg entry is the key for another series' transform)
    for ser in ser_dict:
        try:
            if os.path.isfile(ser_dict[ser_dict[ser]["reg"]]["transform"]):
                transforms = ser_dict[ser_dict[ser]["reg"]]["transform"]
                template = ser_dict[ser_dict[ser]["reg"]]["filename_reg"]
                moving = ser_dict[ser]["filename"]
                niiout = ants_apply(moving, template, 'Linear', transforms, os.path.dirname(dcm_dir), repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
        except Exception:
            pass
    return ser_dict

# split b0 and dwi and discard b0 (if necessary)
def split_dwi(ser_dict):
    dwib0 = ser_dict["DWI"]["filename"]
    if os.path.isfile(dwib0):
        dwinii = nib.load(dwib0)
        if len(dwinii.get_shape()) > 3:
            dwib0img = dwinii.get_data()
            dwi = np.squeeze(dwib0img[:, :, :, 0])
            aff = dwinii.get_affine()
            dwinii = nib.Nifti1Image(dwi, aff)
            nib.save(dwinii, dwib0)
    return ser_dict

# split asl and anatomic image (if necessary)
def split_asl(ser_dict):
    aslperf = ser_dict["ASL"]["filename"]
    aslperfa = aslperf.rsplit(".nii", 1)[0] + "a.nii.gz"
    anat_outname = aslperf.rsplit(".nii", 1)[0] + "_anat.nii.gz"
    anat_json = aslperf.rsplit(".nii", 1)[0] + "_anat.json"
    perf_json = aslperf.rsplit(".nii", 1)[0] + ".json"
    a_json = aslperf.rsplit(".nii", 1)[0] + "a.json"
    # handle when dcm2niix converts to two different files
    if os.path.isfile(aslperfa):
        perfnii = nib.load(aslperf)
        perfanii = nib.load(aslperfa)
        if np.mean(perfnii.get_data()) > np.mean(perfanii.get_data()):
            os.rename(aslperf, anat_outname)
            os.rename(aslperfa, aslperf)
            os.rename(perf_json, anat_json)
            os.rename(a_json, perf_json)
        else:
            os.rename(aslperfa, anat_outname)
            os.rename(a_json, anat_json)
    # handle original case where asl perfusion is a 4D image with anat and perfusion combined
    if os.path.isfile(aslperf):
        if not os.path.isfile(anat_outname):
            nii = nib.load(aslperf)
            if len(nii.get_shape()) > 3:
                img = nii.get_data()
                aslimg = np.squeeze(img[:, :, :, 0])
                anatimg = np.squeeze(img[:, :, :, 1])
                aff = nii.get_affine()
                aslniiout = nib.Nifti1Image(aslimg, aff)
                asl_outname = aslperf
                anatniiout = nib.Nifti1Image(anatimg, aff)
                nib.save(aslniiout, asl_outname)
                nib.save(anatniiout, anat_outname)
        # add new dict entry to use anatomy image for registration
        ser_dict["ASL"].update({"reg_moving": anat_outname})
    else:
        ser_dict.update({"ASL": {"filename": "None", "reg": False}})
    return ser_dict

# DTI processing if present
def dti_proc(ser_dict, dti_index, dti_acqp, dti_bvec, dti_bval, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # id setup
    idno = ser_dict["info"]["id"]
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    if os.path.isfile(ser_dict["DTI"]["filename"]):
        # define DTI input
        dti_in = ser_dict["DTI"]["filename"]
        logger.info("PROCESSING DTI")
        logger.info("- DTI file found at " + dti_in)
        # separate b0 image if not already done
        b0 = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_b0.nii.gz")
        fslroi = ExtractROI(in_file=dti_in, roi_file=b0, t_min=0, t_size=1)
        fslroi.terminal_output = "none"
        if not os.path.isfile(b0):
            logger.info("- separating b0 image from DTI")
            logger.debug(fslroi.cmdline)
            fslroi.run()
        else:
            logger.info("- b0 image already exists at " + b0)
            logger.debug(fslroi.cmdline)
        # add b0 to list for affine registration
        ser_dict.update({"DTI_b0": {"filename": b0, "reg": "diffeo", "reg_target": "FLAIR", "no_norm": True}})
        # make BET mask
        dti_mask = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_mask.nii.gz")
        btr = BET()
        btr.inputs.in_file = b0  # base mask on b0 image
        btr.inputs.out_file = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI")  # output prefix _mask is autoadded
        btr.inputs.mask = True
        btr.inputs.no_output = True
        btr.inputs.frac = 0.2  # lower threshold for more inclusive mask
        if not os.path.isfile(dti_mask) or repeat:
            logger.info("- Making BET brain mask for DTI")
            logger.debug(btr.cmdline)
            _ = btr.run()
        else:
            logger.info("- DTI BET masking already exists at " + dti_mask)
            logger.debug(btr.cmdline)
        # do eddy correction with outlier replacement
        dti_out = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_eddy")
        dti_outfile = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_eddy.nii.gz")
        eddy = Eddy()
        eddy.inputs.in_file = dti_in
        eddy.inputs.out_base = dti_out
        eddy.inputs.in_mask = dti_mask
        eddy.inputs.in_index = dti_index
        eddy.inputs.in_acqp = dti_acqp
        eddy.inputs.in_bvec = dti_bvec
        eddy.inputs.in_bval = dti_bval
        eddy.inputs.use_cuda = True
        eddy.inputs.repol = True
        eddy.terminal_output = "none"
        if not os.path.isfile(dti_outfile) or repeat:
            logger.info("- Eddy correcting DWIs")
            try:
                logger.debug(eddy.cmdline)
                _ = eddy.run()
            except Exception:
                logger.info("- Could not eddy correct DTI")
        else:
            logger.info("- Eddy corrected DWIs already exist at " + dti_outfile)
            logger.debug(eddy.cmdline)
        # do DTI processing
        fa_out = dti_out + "_FA.nii.gz"
        dti = DTIFit()
        dti.inputs.dwi = dti_outfile
        dti.inputs.bvecs = dti_out + ".eddy_rotated_bvecs"
        dti.inputs.bvals = dti_bval
        dti.inputs.base_name = dti_out
        dti.inputs.mask = dti_mask
        # dti.inputs.args = "-w"  # libc++abi.dylib: terminating with uncaught exception of type NEWMAT::SingularException
        dti.terminal_output = "none"
        dti.inputs.save_tensor = True
        # dti.ignore_exception = True  # for some reason running though nipype causes error at end
        if os.path.isfile(dti_outfile) and not os.path.isfile(fa_out) or os.path.isfile(dti_outfile) and repeat:
            logger.info("- Fitting DTI")
            try:
                logger.debug(dti.cmdline)
                _ = dti.run()
                # if DTI processing fails to create FA, it may be due to least squares option, so remove and run again
                if not os.path.isfile(fa_out):
                    dti.inputs.args = ""
                    logger.debug(dti.cmdline)
                    _ = dti.run()
            except Exception:
                if not os.path.isfile(fa_out):
                    logger.info("- could not process DTI")
                else:
                    logger.info("- DTI processing completed")
        else:
            if os.path.isfile(fa_out):
                logger.info("- DTI outputs already exist with prefix " + dti_out)
                logger.debug(dti.cmdline)
        # if DTI processing completed, add DTI to series_dict for registration (note, DTI is masked at this point)
        if os.path.isfile(fa_out):
            ser_dict.update({"DTI_FA": {"filename": dti_out + "_FA.nii.gz", "reg": "DTI_b0"}})
            ser_dict.update({"DTI_MD": {"filename": dti_out + "_MD.nii.gz", "reg": "DTI_b0", "no_norm": True}})
            ser_dict.update({"DTI_L1": {"filename": dti_out + "_L1.nii.gz", "reg": "DTI_b0", "no_norm": True}})
            ser_dict.update({"DTI_L2": {"filename": dti_out + "_L2.nii.gz", "reg": "DTI_b0", "no_norm": True}})
            ser_dict.update({"DTI_L3": {"filename": dti_out + "_L3.nii.gz", "reg": "DTI_b0", "no_norm": True}})
    return ser_dict

# make mask based on t1gad and flair and apply to all other images
def brain_mask(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # id setup
    idno = ser_dict["info"]["id"]
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    # BET brain masking based on combination of t2 and t1
    logger.info("BRAIN MASKING:")

    # for loop for masking different contrasts
    to_mask = ["FLAIR", "T1gad", "DWI"]  # list of contrasts to mask
    masks = []  # list of completed masks
    for contrast in to_mask:
        # if original and registered files exist for this contrast
        if os.path.isfile(ser_dict[contrast]["filename"]) and os.path.isfile(ser_dict[contrast]["filename_reg"]):
            maskfile = os.path.join(os.path.dirname(dcm_dir), idno + "_" + contrast + "_w_mask.nii.gz")
            bet = BET()
            bet.inputs.in_file = ser_dict[contrast]["filename_reg"]
            bet.inputs.out_file = ser_dict[contrast]["filename_reg"]  # no output prevents overwrite
            bet.inputs.mask = True
            bet.inputs.no_output = True
            bet.inputs.frac = 0.5
            bet.terminal_output = "none"
            if not os.path.isfile(maskfile):
                logger.info("- making brain mask based on " + contrast)
                logger.debug(bet.cmdline)
                _ = bet.run()
                if os.path.isfile(maskfile):
                    masks.append(maskfile)
            else:
                logger.info("- " + contrast + " brain mask already exists at " + maskfile)
                logger.debug(bet.cmdline)
                masks.append(maskfile)
        else:
            logger.info("- could not find registered " + contrast + ", skipping brain masking for this contrast")

    # combine brain masks using majority voting
    combined_mask = os.path.join(os.path.dirname(dcm_dir), idno + "_combined_brain_mask.nii.gz")
    majority_cmd = "ImageMath 3 " + combined_mask + " MajorityVoting " + " ".join(masks)
    if not os.path.isfile(combined_mask) and masks:  # if combined mask file does not exist, and indivudual masks exist
        logger.info("- making combined brain mask at " + combined_mask)
        logger.debug(majority_cmd)
        _ = subprocess.call(majority_cmd, shell=True)
    else:
        logger.info("- combined brain mask already exists at " + combined_mask)
        logger.debug(majority_cmd)

    # now apply to all other images if mask exists
    if os.path.isfile(combined_mask):
        for sers in ser_dict:
            if "filename_reg" in ser_dict[sers]:  # check that filename_reg entry exists
                os.path.join(os.path.dirname(dcm_dir), idno + "_" + sers + "_wm.nii.gz")
                ser_masked = ser_dict[sers]["filename_reg"].rsplit(".nii", 1)[0] + "m.nii.gz"

                if repeat or (not os.path.isfile(ser_masked) and os.path.isfile(ser_dict[sers]["filename_reg"])):
                    # apply mask using fsl maths
                    if not ser_dict[sers]["filename_reg"] == ser_dict[sers]["filename"]:
                        logger.info("- masking " + ser_dict[sers]["filename_reg"])
                        # prep command line regardless of whether or not work will be done
                        mask_cmd = ApplyMask()
                        mask_cmd.inputs.in_file = ser_dict[sers]["filename_reg"]
                        mask_cmd.inputs.mask_file = combined_mask
                        mask_cmd.inputs.out_file = ser_masked
                        mask_cmd.terminal_output = "none"
                        logger.debug(mask_cmd.cmdline)
                        _ = mask_cmd.run()
                        ser_dict[sers].update({"filename_masked": ser_masked})
                elif os.path.isfile(ser_masked):
                    logger.info("- masked file already exists for " + sers + " at " + ser_masked)
                    # prep command line regardless of whether or not work will be done
                    mask_cmd = ApplyMask()
                    mask_cmd.inputs.in_file = ser_dict[sers]["filename_reg"]
                    mask_cmd.inputs.mask_file = combined_mask
                    mask_cmd.inputs.out_file = ser_masked
                    mask_cmd.terminal_output = "none"
                    logger.debug(mask_cmd.cmdline)
                    ser_dict[sers].update({"filename_masked": ser_masked})
                elif sers == "info":
                    pass
                else:
                    logger.info("- skipping masking for " + sers + " as file does not exist")
            else:
                logger.info("- no filename_reg entry exists for series " + sers)
    else:
        logger.info("- combined mask file not found, expected location is: " + combined_mask)
    return ser_dict

# bias correction
# takes series dict, returns series dict, performs intensity windsorization and n4 bias correction on select volumes
def bias_correct(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("BIAS CORRECTING IMAGES:")
    # id setup
    idno = ser_dict["info"]["id"]
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    # identify files to bias correct
    for srs in ser_dict:
        if "bias" in ser_dict[srs] and ser_dict[srs]["bias"]: # only do bias if specified in series dict
            # get name of masked file and the mask itself
            f = os.path.join(os.path.dirname(dcm_dir), idno + "_" + srs + "_wm.nii.gz")
            mask = os.path.join(os.path.dirname(dcm_dir), idno + "_combined_brain_mask.nii.gz")
            if not os.path.isfile(f):
                logger.info("-skipping bias correction for " + srs + " as masked file does not exist.")
            elif not os.path.isfile(mask):
                logger.info("-skipping bias correction for " + srs + " as mask image does not exist.")
            else:
                # generate output filenames for corrected image and bias field
                biasfile = f.rsplit(".nii", 1)[0] + "_biasmap.nii.gz"
                truncated_img = f.rsplit(".nii", 1)[0] + "t.nii.gz"
                biasimg = truncated_img.rsplit(".nii", 1)[0] + "b.nii.gz"
                # first truncate image intensities, this also removes any negative values
                if not os.path.isfile(truncated_img) or repeat:
                    logger.info("- truncating image intensities for " + f)
                    thresh = [0.001, 0.999]  # define thresholds
                    mask_img = nib.load(mask)  # load data
                    mask_img = mask_img.get_data()
                    nii = nib.load(f)
                    img = nii.get_data()
                    affine = nii.get_affine()
                    vals = np.sort(img[mask_img>0.], None)  # sort data in order
                    thresh_lo = vals[int(np.round(thresh[0] * len(vals)))]  # define hi and low thresholds
                    thresh_hi = vals[int(np.round(thresh[1] * len(vals)))]
                    img_trunc = np.where(img<thresh_lo, thresh_lo, img)  # truncate low
                    img_trunc = np.where(img_trunc>thresh_hi, thresh_hi, img_trunc)  # truncate hi
                    img_trunc = np.where(mask_img>0., img_trunc, 0.)  # remask data
                    nii = nib.Nifti1Image(img_trunc, affine)  # make nii and save
                    nib.save(nii, str(truncated_img))
                else:
                    logger.info("- truncated image already exists at " + truncated_img)
                # run bias correction on truncated image
                # apply N4 bias correction
                n4_cmd = N4BiasFieldCorrection()
                n4_cmd.inputs.copy_header=True
                n4_cmd.inputs.input_image=truncated_img
                n4_cmd.inputs.save_bias=True
                n4_cmd.inputs.bias_image=biasfile
                n4_cmd.inputs.bspline_fitting_distance=300
                n4_cmd.inputs.bspline_order=3
                n4_cmd.inputs.convergence_threshold=1e-6
                n4_cmd.inputs.dimension=3
                # n4_cmd.inputs.environ=
                n4_cmd.inputs.mask_image=mask
                n4_cmd.inputs.n_iterations=[50,50,50,50]
                n4_cmd.inputs.num_threads=multiprocessing.cpu_count()
                n4_cmd.inputs.output_image=biasimg
                n4_cmd.inputs.shrink_factor=3
                #n4_cmd.inputs.weight_image=
                if not os.path.isfile(biasimg) or repeat:
                    logger.info("- bias correcting " + truncated_img)
                    logger.debug(n4_cmd.cmdline)
                    _ = n4_cmd.run()
                    ser_dict[srs].update({"filename_bias": biasimg})
                else:
                    logger.info("- bias corrected image already exists at " + biasimg)
                    logger.debug(n4_cmd.cmdline)
                    ser_dict[srs].update({"filename_bias": biasimg})
    return ser_dict

# normalize nifits
# takes a series dict and returns a series dict, normalizes all masked niis
# currently uses zero mean and unit variance
def norm_niis(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("NORMALIZING NIIs:")
    # get list of filenames and normalize them
    for srs in ser_dict:
        if "bias" in ser_dict[srs] and ser_dict[srs]["bias"]:  # do normalization if bias was done
            fn = ser_dict[srs]["filename_bias"]
            normname = os.path.join(os.path.dirname(fn), os.path.basename(fn).split(".")[0] + "n.nii.gz")
            # if normalized file doesn't exist, make it
            if not os.path.isfile(normname) or repeat:
                logger.info("- normalizing " + srs + " at " + fn)
                # load image into memory
                nii = nib.load(fn)
                affine = nii.get_affine()
                img = nii.get_data()
                nzi = np.nonzero(img)
                mean = np.mean(img[nzi])
                std = np.std(img[nzi])
                img = np.where(img != 0., (img - mean) / std, 0.)  # zero mean unit variance for nonzero indices
                nii = nib.Nifti1Image(img, affine)
                nib.save(nii, normname)
                ser_dict[srs].update({"filename_norm": normname})
            else:
                logger.info("- normalized " + srs + " already exists at " + normname)
                ser_dict[srs].update({"filename_norm": normname})
    return ser_dict

# make 4d nii
# takes series dict, returns series dict, makes a 4d nii using all normed series
def make_nii4d(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("MAKING 4D NII:")
    # id setup
    idno = ser_dict["info"]["id"]
    # define vars
    files = []
    # get all normalized images and collect them in a list
    for srs in ser_dict:
        if "filename_norm" in ser_dict[srs]: # normalized files only, others ignored
            files.append(ser_dict[srs]["filename_norm"])
            logger.info("- adding " + srs + " to nii4D list at " + ser_dict[srs]["filename_norm"])
    if files and len(files)>1: # only attempt work if normalized files exist and there is more than 1
        # get dirname from first normalized image, make nii4d name from this
        bdir = os.path.dirname(files[0])
        nii4d = os.path.join(bdir, idno + "_nii4d.nii.gz")
        # if nii4d doesn't exist, make it
        #nii4dcmd = "ImageMath 4 " + nii4d + " TimeSeriesAssemble 1 1 " + " ".join(files)
        #os.system(nii4dcmd)
        merger = Merge()
        merger.inputs.in_files = files
        merger.inputs.dimension = "t"
        merger.inputs.merged_file = nii4d
        merger.terminal_output = "none"
        if not os.path.isfile(nii4d) or repeat:
            logger.info("- creating 4D nii at " + nii4d)
            logger.debug(merger.cmdline)
            merger.run()
        else:
            logger.info("- 4D Nii already exists at " + nii4d)
            logger.debug(merger.cmdline)
    else:
        logger.info("- not enough files to make 4D Nii")
    return ser_dict

# create tumor segmentation
def tumor_seg(ser_dict):
    # logging
    logger = logging.getLogger("my_logger")
    # id setup
    idno = ser_dict["info"]["id"]
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    logger.info("SEGMENTING TUMOR VOLUMES:")
    seg_file = os.path.join(os.path.dirname(dcm_dir), idno + "_tumor_seg.nii.gz")
    files = ["FLAIR", "T1", "T1gad", "T2"]
    for item in files:
        if not os.path.isfile(ser_dict[item]["filename"]) or not os.path.isfile(ser_dict[item]["filename_masked"]):
            print("- missing file for segmentation")
            return ser_dict
    if not os.path.isfile(seg_file):
        logger.info("- segmenting tumor")
        test_ecalabr.test(os.path.dirname(dcm_dir))
        # copy header information from warped masked flair to tumor seg
        logger.info("- correcting nii header for segmentation file")
        hdrfix = CopyGeom()
        hdrfix.inputs.dest_file = seg_file
        hdrfix.inputs.in_file = ser_dict["FLAIR"]["filename_masked"]
        hdrfix.terminal_output = "none"
        logger.debug(hdrfix.cmdline)
        _ = hdrfix.run()
    else:
        logger.info("- tumor segmentation file aready exists at " + seg_file)
    return ser_dict

# print and save series dict
def print_series_dict(series_dict, repeat=False):
    dcm_dir = series_dict["info"]["dcmdir"]
    # first save as a numpy file
    serdict_outfile = os.path.join(os.path.dirname(dcm_dir), series_dict["info"]["id"] + "_metadata.npy")
    if not os.path.isfile(serdict_outfile) or repeat:
        np.save(serdict_outfile, series_dict)
    # save human readable serdict file with binary data removed
    hr_serdict_outfile = os.path.join(os.path.dirname(dcm_dir), series_dict["info"]["id"] + "_metadata_HR.txt")
    if not os.path.isfile(hr_serdict_outfile) or repeat:
        # remove binary entries from dict, also will only print the first dicom path from the list
        def remove_nonstr_from_dict(a_dict):
            new_dict = {}
            for k, v in a_dict.items():
                if isinstance(v, dict):
                    v = remove_nonstr_from_dict(v)
                if isinstance(v, (int, long, float, complex, str, list, dict)):
                    if k == "dicoms" and isinstance(v, list) and v:  # ensure dicoms is a list and is not empty
                        new_dict[k] = v[0]
                    else:
                        new_dict[k] = v
            return new_dict
        hr_serdict = remove_nonstr_from_dict(series_dict)
        with open(hr_serdict_outfile, 'w') as f:
            f.write("%s" % yaml.dump(hr_serdict))
    return series_dict
