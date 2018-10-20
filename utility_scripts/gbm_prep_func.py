import os
import pydicom as dicom
import nibabel as nib
import numpy as np
from glob import glob
import subprocess
import logging
import yaml
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.fsl import BET
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl import Eddy
from nipype.interfaces.fsl import DTIFit
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.fsl.utils import ImageStats
from nipype.interfaces.fsl.utils import ImageMaths
from nipype.interfaces.fsl.utils import CopyGeom
from nipype.interfaces.fsl import Merge
import external_software.brats17_master.test_ecalabr as test_ecalabr


### Set up logging
# takes work dir
# returns logger
def make_log(work_dir):
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    # make log file, append to existing
    idno = os.path.basename(work_dir)
    log_file = os.path.join(work_dir, idno + "_log.txt")
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
def make_serdict(reg_atlas, dcm_dir):
    t1_str = [["ax t1"], ["ax", "3d", "bravo", "brainnav"], ["fspgr", "pre", "t1"]]
    t1_not = ["post", "flair", "+", " pg ", "c-sp", "t-sp", "l-sp"]
    t2_str = [["t2"]]
    t2_not = ["flair", "optic", "motor", "track", "tract", "radiation", "reform"]
    flair_str = [["flair"]]
    flair_not = ["t1", "reform", "rfmt"]
    dwi_str = [["ax", "dwi"], ["trace"]]
    dwi_not = []
    adc_str = [["adc"], ["apparent", "diffusion"], ["avdc"]]
    adc_not = ["exp", "cor", "sag", "eadc"]
    t1gad_str = [["spgr", "gad"], ["bravo", "gad"], ["+c", "t1"], ["fspgr", "bravo"], ["t1", " pg "], ["t1", "gad"], ["t1", "post"]]
    t1gad_not = ["pre", "without", "w/o", "reform", "c-sp", "t-sp", "l-sp", "track", "motor", "left", "right"]
    swi_str = [["isi"], ["swan"]]
    swi_not = ["ref", "filt", "pha", "rf", "mip", "min"]
    asl_str = [["asl"]]
    asl_not = ["cerebral"]
    dti_str = [["dti"], ["hardi"]]
    dti_not = ["roi", "topup"]
    # note for "reg" key option can be False (no reg), "trans" or "affine" for thos reg types, and "SERIES" to apply transform from another previous series
    sdict = {"FLAIR": {"or": flair_str, "not": flair_not, "reg": "trans", "reg_target": reg_atlas},
             "T1": {"or": t1_str, "not": t1_not, "reg": "affine", "reg_target": "FLAIR"},
             "T1gad": {"or": t1gad_str, "not": t1gad_not, "reg": "affine", "reg_target": "FLAIR"},
             "T2": {"or": t2_str, "not": t2_not, "reg": "affine", "reg_target": "FLAIR"},
             "DWI": {"or": dwi_str, "not": dwi_not, "reg": "diffeo", "reg_target": "FLAIR"},
             "ADC": {"or": adc_str, "not": adc_not, "reg": "DWI"},
             "SWI": {"or": swi_str, "not": swi_not, "reg": "affine", "reg_target": "FLAIR"},
             "ASL": {"or": asl_str, "not": asl_not, "reg": "diffeo", "reg_target": "FLAIR"},
             "DTI": {"or": dti_str, "not": dti_not, "reg": False},
             "info":{"filename": "None", "dcmdir": dcm_dir, "id": os.path.basename(os.path.dirname(dcm_dir))}
             }
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
            series.append(hdrs[ind].SeriesDescription)
        except Exception:
            logger.info("- Skipping series " + str(ind + 1) + " without series description")
            series.append("none")

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
                pass
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
    # for each input series description
    number = 1
    for ind, string in enumerate(strings, 0):
        match = False
        # for each substr list in substrs
        for substrlist in substrs:
            # if all substrs match and no NOT substrs match then append to matches
            if all(ss.lower() in string.lower() for ss in substrlist) \
                    and not any(ssn.lower() in string.lower() for ssn in substrnot):
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
            if strs_dict[series]["series"] == "None":
                logger.info("- no matching series found: " + series)
                outfilepath = "None"
            elif not os.path.isfile(outfilepath) or repeat:
                logger.info("- Converting " + outfilename)
                logger.debug(converter.cmdline)
                converter.run()
            else:
                logger.info("- " + outfilename + " already exists")
            # append to outfile list
            output_ser.append(outfilepath)
            strs_dict[series].update({"filename": outfilepath})
    # print outputs of file conversion
    logger.info("CONVERTED FILES LIST:")
    for ser in strs_dict:
        if "dicoms" in strs_dict[ser] and "filename" in strs_dict[ser] and os.path.isfile(strs_dict[ser]["filename"]):
            logger.info("- " + ser + " = " + strs_dict[ser]["filename"])
    return strs_dict

# Fast ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def affine_reg(moving_nii, template_nii, work_dir, repeat=False):
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
    antsreg.inputs.num_threads=8
    antsreg.inputs.smoothing_sigmas=[[6, 4, 1, 0]] * 2
    antsreg.inputs.sigma_units=['vox'] * 2
    antsreg.inputs.transforms=['Rigid', 'Affine']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    antsreg.inputs.initial_moving_transform_com=1  # use center of mass for initial transform
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
    return trnsfm

# Fast ants diffeomorphic registration
# takes moving and template niis and a work dir
# performs fast diffeomorphic registration and returns a list of transforms
def diffeo_reg(moving_nii, template_nii, work_dir, repeat=False):
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
    antsreg.inputs.num_threads=8
    antsreg.terminal_output='none'
    antsreg.inputs.initial_moving_transform_com=1
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
    return trnsfm

# ANTS translation
# takes moving and template niis and a work dir
# performs fast translation only registration and returns a list of transforms
def trans_reg(moving_nii, template_nii, work_dir, repeat=False):
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
    antsreg.inputs.num_threads=8
    antsreg.inputs.smoothing_sigmas=[[6, 4, 1, 0]]
    antsreg.inputs.sigma_units=['vox']
    antsreg.inputs.transforms=['Translation']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    antsreg.inputs.initial_moving_transform_com=1  # use center of mass for initial transform
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
        if not os.path.isfile(output_nii[ind]) or repeat:
            logger.info("- Creating warped image " + output_nii[ind])
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
            logger.debug(antsapply.cmdline)
            antsapply.run()
        else:
            logger.info("- Transformed image already exists at " + output_nii[ind])
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
        if ser_dict[ser]["filename"] == "None" or not ser_dict[ser]["reg"]:
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
            transforms = trans_reg(movingr, template, os.path.dirname(dcm_dir), repeat)
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
                transforms = affine_reg(movingr, template, os.path.dirname(dcm_dir), repeat)
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
                transforms = diffeo_reg(movingr, template, os.path.dirname(dcm_dir), repeat)
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

# normalize nifits
# takes a series dict and returns a series dict, normalizes all masked niis
def norm_niis(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("NORMALIZING NIIs:")
    # get list of filenames and normalize them
    for srs in ser_dict:
        if "filename_masked" in ser_dict[srs] and os.path.isfile(ser_dict[srs]["filename_masked"]):
            if "no_norm" in ser_dict[srs] and ser_dict[srs]["no_norm"] == True:
                logger.info("- skipping normalization for " + srs)
            else:
                fn = ser_dict[srs]["filename_masked"]
                normname = os.path.join(os.path.dirname(fn), os.path.basename(fn).split(".")[0] + "n.nii.gz")
                ser_dict[srs].update({"filename_norm": normname})
                # if normalized file doesn't exist, make it
                if not os.path.isfile(normname) or repeat:
                    logger.info("- normalizing " + srs + " at " + fn)
                    #norm_cmd = "ImageMath 3 " + ser_dict[srs]["filename_norm"] + " Normalize " + fn
                    #_ = os.system(norm_cmd)
                    # first get max and min
                    im_stats = ImageStats()
                    im_stats.inputs.in_file=fn
                    im_stats.inputs.op_string="-R"
                    im_stats.terminal_output="allatonce"
                    logger.debug(im_stats.cmdline)
                    result = im_stats.run()
                    minv, maxv = result.outputs.out_stat
                    # then normalize by subtracting min and divding by difference between max and min
                    norm = ImageMaths()
                    norm.inputs.in_file=fn
                    norm.inputs.out_file=normname
                    norm.inputs.out_data_type="float"
                    norm.terminal_output="none"
                    norm.inputs.op_string="-sub %s -div %s" % (minv, (maxv-minv))
                    logger.debug(norm.cmdline)
                    norm.run()
                else:
                    logger.info("- normalized " + srs + " already exists at " + normname)
    # remove stat result json file if exists, check home and current dir
    if os.path.isfile("~/stat_result.json"):
        os.remove("~/stat_result.json")
    if os.path.isfile("stat_result.json"):
        os.remove("stat_result.json")
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
        if not os.path.isfile(nii4d) or repeat:
            logger.info("- creating 4D nii at " + nii4d)
            #nii4dcmd = "ImageMath 4 " + nii4d + " TimeSeriesAssemble 1 1 " + " ".join(files)
            #os.system(nii4dcmd)
            merger = Merge()
            merger.inputs.in_files = files
            merger.inputs.dimension = "t"
            merger.inputs.merged_file = nii4d
            merger.terminal_output = "none"
            logger.debug(merger.cmdline)
            merger.run()
        else:
            logger.info("- 4D Nii already exists at " + nii4d)
    else:
        logger.info("- not enough files to make 4D Nii")
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
    if os.path.isfile(aslperf):
        anat_outname = aslperf.rsplit(".nii", 1)[0] + "_anat.nii.gz"
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
        # separate b0 image
        b0 = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_b0.nii.gz")
        if not os.path.isfile(b0):
            logger.info("- separating b0 image from DTI")
            fslroi = ExtractROI(in_file=dti_in, roi_file=b0, t_min=0, t_size=1)
            fslroi.terminal_output = "none"
            logger.debug(fslroi.cmdline)
            fslroi.run()
        else:
            logger.info("- b0 image already exists at " + b0)
        # add b0 to list for affine registration
        ser_dict.update({"DTI_b0": {"filename": b0, "reg": "diffeo", "reg_target": "FLAIR", "no_norm": True}})
        # make BET mask
        dti_mask = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_mask.nii.gz")
        if not os.path.isfile(dti_mask) or repeat:
            logger.info("- Making BET brain mask for DTI")
            btr = BET()
            btr.inputs.in_file = b0  # base mask on b0 image
            btr.inputs.out_file = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI")  # output prefix _mask is autoadded
            btr.inputs.mask = True
            btr.inputs.no_output = True
            btr.inputs.frac = 0.2  # lower threshold for more inclusive mask
            logger.debug(btr.cmdline)
            _ = btr.run()
        else:
            logger.info("- DTI BET masking already exists at " + dti_mask)
        # do eddy correction with outlier replacement
        dti_out = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_eddy")
        dti_outfile = os.path.join(os.path.dirname(dcm_dir), idno + "_DTI_eddy.nii.gz")
        if not os.path.isfile(dti_outfile) or repeat:
            logger.info("- Eddy correcting DWIs")
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
            try:
                logger.debug(eddy.cmdline)
                _ = eddy.run()
            except Exception:
                logger.info("- Could not eddy correct DTI")
        else:
            logger.info("- Eddy corrected DWIs already exist at " + dti_outfile)
        # do DTI processing
        fa_out = dti_out + "_FA.nii.gz"
        if os.path.isfile(dti_outfile) and not os.path.isfile(fa_out) or os.path.isfile(dti_outfile) and repeat:
            logger.info("- Fitting DTI")
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
        # if DTI processing completed, add DTI to series_dict for registration (note, DTI is masked at this point)
        if os.path.isfile(fa_out):
            ser_dict.update({"DTI_FA": {"filename": dti_out + "_FA.nii.gz", "reg": "DTI_b0"}})
            ser_dict.update({"DTI_MD": {"filename": dti_out + "_MD.nii.gz", "reg": "DTI_b0", "no_norm": True}})
            ser_dict.update({"DTI_L1": {"filename": dti_out + "_L1.nii.gz", "reg": "DTI_b0", "no_norm": True}})
            ser_dict.update({"DTI_L2": {"filename": dti_out + "_L2.nii.gz", "reg": "DTI_b0", "no_norm": True}})
            ser_dict.update({"DTI_L3": {"filename": dti_out + "_L3.nii.gz", "reg": "DTI_b0", "no_norm": True}})
    return ser_dict

# make mask based on t1gad and flair and apply to all other images
def brain_mask(ser_dict):
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
            if not os.path.isfile(maskfile):
                logger.info("- making brain mask based on " + contrast)
                bet = BET()
                bet.inputs.in_file = ser_dict[contrast]["filename_reg"]
                bet.inputs.out_file = ser_dict[contrast]["filename_reg"]  # no output prevents overwrite
                bet.inputs.mask = True
                bet.inputs.no_output = True
                bet.inputs.frac = 0.5
                bet.terminal_output = "none"
                logger.debug(bet.cmdline)
                _ = bet.run()
                if os.path.isfile(maskfile):
                    masks.append(maskfile)
            else:
                logger.info("- " + contrast + " brain mask already exists at " + maskfile)
                masks.append(maskfile)
        else:
            logger.info("- could not find registered " + contrast + ", skipping brain masking for this contrast")

    # combine brain masks using majority voting
    combined_mask = os.path.join(os.path.dirname(dcm_dir), idno + "_combined_brain_mask.nii.gz")
    if not os.path.isfile(combined_mask) and masks:  # if combined mask file does not exist, and indivudual masks exist
        logger.info("- making combined brain mask at " + combined_mask)
        majority_cmd = "ImageMath 3 " + combined_mask + " MajorityVoting " + " ".join(masks)
        logger.debug(majority_cmd)
        _ = subprocess.call(majority_cmd, shell=True)

    # now apply to all other images if mask exists
    if os.path.isfile(combined_mask):
        for sers in ser_dict:
            if "filename_reg" in ser_dict[sers]:  # check that filename_reg entry exists
                os.path.join(os.path.dirname(dcm_dir), idno + "_" + sers + "_wm.nii.gz")
                ser_masked = ser_dict[sers]["filename_reg"].rsplit(".nii", 1)[0] + "m.nii.gz"
                if not os.path.isfile(ser_masked) and os.path.isfile(ser_dict[sers]["filename_reg"]):
                    # check that warping was actually done or not
                    if not ser_dict[sers]["filename_reg"] == ser_dict[sers]["filename"]:
                        logger.info("- masking " + ser_dict[sers]["filename_reg"])
                        # apply mask using fsl maths
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