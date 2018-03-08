import os
import dicom
import nibabel as nib
import numpy as np
from glob import glob
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
import nibabel as nib


def dicom2nii(study):
    # get base directory and make output directory
    output_dir = os.path.join(study.rsplit("/", 2)[0], "converted")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # make temp dir
    tmp_dir = os.path.join(study.rsplit("/", 2)[0], "temp")
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        # basic converter initiation
        converter = Dcm2niix()
        converter.inputs.source_dir = study
        converter.inputs.compress = "y"
        converter.inputs.output_dir = tmp_dir
        converter.terminal_output = "none"
        converter.anonymize = True
        converter.inputs.out_filename = "%i-%s-%q-"  # define output filenames
        print(converter.cmdline)
        converter.run()
    return output_dir, tmp_dir


def find_dwi(out_dir, nii_dir):
    niis = glob(nii_dir + "/*.nii.gz")
    dwi = ""
    nifti = ""
    idno = ""
    ser = ""
    # work through each nifti in the output folder until one matches criteria
    for nii in niis:
        filename = os.path.basename(nii)
        idno = filename.split("-")[0]
        ser = filename.split("-")[1]
        seq = filename.split("-")[2]
        nifti = nib.load(nii)
        # matching criteria are series < 100, sequence is EP_SE and is 4D (i.e. composed with b0 img)
        if int(ser) < 100 and seq == "EP_SE" and nifti.header["dim"][4] == 2:
            dwi = nii
            break
    # if a dwi was found, then split b0 from it and save as dwi
    dwi_out = os.path.join(out_dir, idno + "_dwi.nii.gz")
    if dwi:
        if not os.path.isfile(dwi_out):
            img = nifti.get_data()
            img = np.squeeze(img[:, :, :, 0])
            affine = nifti.get_affine()
            niiout = nib.Nifti1Image(img, affine)
            nib.save(niiout, dwi_out)
            print("Saved dwi for ID " + idno + " at " + dwi_out)
        else:
            print("DWI already exists at " + dwi_out)
    else:
        print("Could not find dwi for ID " + idno)
    # return some important info, series number and nifti dims
    return int(ser), nifti.header["dim"], dwi_out


def find_adc(out_dir, nii_dir, dwi_series, dwi_dimensions):
    niis = glob(nii_dir + "/*.nii.gz")
    adc = ""
    nifti = ""
    idno = ""
    # work through each nifti in the output folder until one matches criteria
    for nii in niis:
        filename = os.path.basename(nii)
        idno = filename.split("-")[0]
        ser = filename.split("-")[1]
        seq = filename.split("-")[2]
        nifti = nib.load(nii)
        # matching criteria are: series 300 if dwi is 3, sequence is EP_SE and is same dims as dwi
        if int(ser) == (dwi_series*100) and seq == "EP_SE" and all(nifti.header["dim"][1:3] == dwi_dimensions[1:3]):
            adc = nii
            break
    # if a dwi was found, then split b0 from it and save as dwi
    adc_out = os.path.join(out_dir, idno + "_adc.nii.gz")
    if adc:
        if not os.path.isfile(adc_out):
            nib.save(nifti, adc_out)
            print("Saved adc for ID " + idno + " at " + adc_out)
        else:
            print("ADC already exists at " + adc_out)
    else:
        print("Could not find adc for ID " + idno)
    return adc_out


def find_flair(out_dir, nii_dir):
    niis = glob(nii_dir + "/*.nii.gz")
    flair = ""
    nifti = ""
    idno = ""
    # work through each nifti in the output folder until one matches criteria
    for nii in niis:
        filename = os.path.basename(nii)
        idno = filename.split("-")[0]
        ser = filename.split("-")[1]
        seq = filename.split("-")[2]
        nifti = nib.load(nii)
        # matching criteria are: series single digit, sequence is SE_IR
        if int(ser) < 100 and seq == "SE_IR":
            flair = nii
            break
    # if a dwi was found, then split b0 from it and save as dwi
    flair_out = os.path.join(out_dir, idno + "_flair.nii.gz")
    if flair:
        if not os.path.isfile(flair_out):
            nib.save(nifti, flair_out)
            print("Saved flair for ID " + idno + " at " + flair_out)
        else:
            print("Flair already exists at " + flair_out)
    else:
        print("Could not find flair for ID " + idno)
    return flair_out


def resample_flair(flair):
    output = os.path.join(os.path.dirname(flair), os.path.basename(flair).rsplit(".", 2)[0] + "_resampled.nii.gz")
    if not os.path.isfile(output):
        cmd = "ResampleImageBySpacing 3 " + flair + " " + output + " 1 1 1 0 0 0"
        print("Resampling flair")
        os.system(cmd)
    else:
        print("Resampled flair already exists at " + output)
    return output


def reg_2_flair(flair, dwi, adc):
    return flair, dwi, adc


def convert_dicoms(parent_dir):
    # get a list of all patient directories
    patients = glob(parent_dir + "/*/")
    # for each patient
    for n, patient in enumerate(patients, 1):
        # limit number of entries processed for testing purposes
        if n > 200:
            return
        # for each series within each patient, remove prior converted or temp dirs
        studies = glob(patient + "/*/")
        try:
            studies.remove(os.path.join(patient, "converted/"))
            studies.remove(os.path.join(patient, "temp/"))
        except:
            pass
        for study in studies:
            print("\nWorking on study " + study)
            #dicoms = glob(study + "/*")
            output_dir, temp_dir = dicom2nii(study)
            dwi_ser, dwi_dims, dwi = find_dwi(output_dir, temp_dir)
            adc = find_adc(output_dir, temp_dir, dwi_ser, dwi_dims)
            flair = find_flair(output_dir, temp_dir)
            resample_flair(flair)


kpdir = "/Users/edc15/Desktop/kp_test/"
convert_dicoms(kpdir)


"""
# wrapper function for reading a complete dicom directory with/without registration to one of the images
def read_dicom_dir(dcm_dir, series_dict=(), reg_target="DWI", rep=False):
    """"""
    This function takes a DICOM directory for a single patient study, and converts all appropriate series to NIFTI.
    It will also register any/all datasets to a given target.
    :param dcm_dir: the full path to the dicom directory as a string
    :param series_dict: a dict containing the name of each series, the substrings used to search for it, and reg bool
    :param reg_target: a string specifying the series to use as template for registration
    :return: returns the path to a metadata file dict that contains lots of relevant info
    Note this forces axial plane image orientation. To change this, edit the filter_series function
    """"""
    # first check for series_dict
    if not series_dict:
        series_dict = {"T1": {"or": [["ax", "t1"]], "not": ["post", "cor", "gad"], "reg": True},
                       "T2": {"or": [["ax", "t2"]], "not": ["cor"], "reg": True},
                       "FLAIR": {"or": [["ax", "flair"]], "not": ["cor"], "reg": True},
                       "DWI": {"or": [["ax", "dwi"]], "not": ["cor"], "reg": False},
                       "ADC": {"or": [["ax", "adc"], ["apparent", "diffusion"]], "not": ["exp", "cor"], "reg": False},
                       "T1gad": {"or": [["ax", "t1", "gad"]], "not": ["pre"], "reg": True}
                       }

    # function to get complete series list from dicom directory (KP style)
    def get_series(dicom_dir):
        # define variables
        dicoms = []
        hdrs = []
        series = []
        dirs =[]

        # get unique prefixes
        prfxs = glob(dicom_dir + "/*")
        prfxs = [p.rsplit('.', 1)[0] for p in prfxs]
        prfxs = set(prfxs)
        prfxs = list(prfxs)

        # get lists of all dicom series
        for ind, prfx in enumerate(prfxs):
            # get list of dicoms for each prefix
            dicoms.append(glob(os.path.join(dicom_dir, prfx) + "*"))
            hdrs.append(dicom.dcmread(dicoms[ind][0]))
            try:
                series.append(hdrs[ind].SeriesDescription)
            except:
                print("Skipping series " + str(ind + 1) + " without series description")

        # save series list
        series_file = os.path.join(dicom_dir, "series_list.txt")
        if not os.path.isfile(series_file):
            fileout = open(series_file, 'w')
            for item in series:
                fileout.write("%s\n" % item)

        return dicoms, hdrs, series, dirs

    # first define function to get filter substring matches by criteria
    def substr_list(strings, substrs, substrnot):
        # define variables
        inds = []

        # for each input filename
        for ind, string in enumerate(strings, 0):
            # for each substr list in substrs
            for substrlist in substrs:
                # if all substrs match and no NOT substrs match then append to matches
                if all(ss.lower() in string.lower() for ss in substrlist) \
                        and not any(ssn.lower() in string.lower() for ssn in substrnot):
                    inds.append(ind)
                    print("matched series: " + string)

        return inds

    # get filtered series list
    def filter_series(dicoms, hdrs, series, dirs, srs_dict):
        # define variables
        new_dicoms = []
        new_hdrs = []
        new_series = []
        new_dirs = []

        # for each output, find match and append to new list for conversion
        keeper = []
        for srs in srs_dict:
            print("\nFINDING SERIES: " + srs)
            inds = substr_list(series, srs_dict[srs]["or"], srs_dict[srs]["not"])

            # if there are inds of matches, pick the first match in the axial orientation and continue
            # if no axial reformats, then pick another orientation
            if inds:  # if only 1 ind, then use it
                if len(inds) == 1:
                    inds = inds[0]
                else:  # if more than 1 inds, try to find axial
                    for i in inds:  # if there is an axial, keep it only if it has more slices than others
                        if all(np.round(hdrs[i].ImageOrientationPatient) == [1., 0., 0., 0., 1., 0.]):
                            # if there is already a keeper, only replace if there are more slices in new one
                            if keeper and int(hdrs[i].ImagesInAcquisition) > int(hdrs[keeper].ImagesInAcquisition):
                                keeper = i
                            else:
                                keeper = i
                        if not keeper:  # if no inds correspond to axials, then use the first img in other orientation
                            keeper = inds[0]
                            # flag orientation, possible that ants handles orientation already, and this  is not needed
                            srs_dict[srs]["reorient"] = True
                    inds = keeper

            if inds or inds == 0:  # this handles 0 index matches
                print("keeping series: " + series[inds])
                new_dicoms.append(dicoms[inds])
                srs_dict[srs]["dicoms"] = dicoms[inds]
                new_hdrs.append(hdrs[inds])
                srs_dict[srs]["hdrs"] = hdrs[inds]
                new_series.append(series[inds])
                srs_dict[srs]["series"] = series[inds]
                new_dirs.append(dirs[inds])
                srs_dict[srs]["dirs"] = dirs[inds]
            else:
                print("no matching series found! \n")
                new_dicoms.append([])
                srs_dict[srs]["dicoms"] = []
                new_hdrs.append([])
                srs_dict[srs]["hdrs"] = []
                new_series.append("None")
                srs_dict[srs]["series"] = "None"
                new_dirs.append("None")
                srs_dict[srs]["dirs"] = "None"

        return srs_dict

    # define function to convert selected dicoms
    def dcm_list_2_niis(strs_dict, dicom_dir, repeat=False):
        # define vars
        output_ser = []
        print("\nCONVERTING FILES:")

        # basic converter initiation
        converter = Dcm2niix()
        converter.inputs.compress = 'y'
        converter.inputs.output_dir = dicom_dir
        converter.terminal_output = 'none'
        converter.anonymize = True
        converter.inputs.out_filename = '%d%p%q%z%s'  # define output filenames





        # convert all subdirectories from dicom to nii
        for series in strs_dict:
            converter.inputs.source_names = strs_dict[series]["dicoms"]
            infile = strs_dict[series]["series"]
            outfilename = infile.replace(" ", "_").lower().replace("(", "").replace(")", "").replace("/", "").replace(":", "")
            converter.inputs.out_filename = outfilename
            outfilepath = os.path.join(dicom_dir, outfilename + ".nii.gz")
            # don't repeat conversion if already done
            if strs_dict[series]["series"] == "None":
                print("No matching series found: " + series)
                outfilepath = "None"
            elif not os.path.isfile(outfilepath) or repeat:
                print("\nAttempting to convert: " + outfilename + "\n")
                print(converter.cmdline)
                converter.run()
            else:
                print(outfilename + " already exists")

            # append to outfile list
            output_ser.append(outfilepath)
            strs_dict[series].update({"filename": outfilepath})

        return strs_dict

    # Fast ants affine
    # takes moving and template niis and a work dir
    # performs fast linear and/or non-linear registration and returns a list of transforms
    def affine_reg(moving_nii, template_nii, work_dir, repeat=False):
        # get basenames
        moving_name = os.path.basename(moving_nii).split(".")[0]
        template_name = os.path.basename(template_nii).split(".")[0]
        outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

        # registration setup
        antsreg = Registration(args='--float',
                               fixed_image=template_nii,
                               moving_image=moving_nii,
                               output_transform_prefix=outprefix,
                               num_threads=8,
                               # output_inverse_warped_image=True,
                               # output_warped_image=True,
                               smoothing_sigmas=[[6, 4, 1, 0]] * 2,
                               sigma_units=['vox'] * 2,
                               transforms=['Rigid', 'Affine'],  # ['Rigid', 'Affine', 'SyN']
                               terminal_output='none',
                               use_histogram_matching=True,
                               write_composite_transform=True,
                               initial_moving_transform_com=1,  # use center of mass for initial transform
                               winsorize_lower_quantile=0.005,
                               winsorize_upper_quantile=0.995,
                               convergence_threshold=[1e-07],
                               convergence_window_size=[10],
                               metric=['Mattes', 'Mattes'],  # ['MI', 'MI', 'CC']
                               metric_weight=[1.0] * 2,
                               number_of_iterations=[[1000, 1000, 1000, 1000]] * 2,  # [100, 70, 50, 20]
                               radius_or_number_of_bins=[32, 32],  # 4
                               sampling_percentage=[0.25, 0.25],  # 1
                               sampling_strategy=['Regular', 'Regular'],  # 'None'
                               shrink_factors=[[4, 3, 2, 1]] * 2,  # *3
                               transform_parameters=[(0.1,), (0.1,)])  # (0.1, 3.0, 0.0) # affine gradient step
        trnsfm = outprefix + "Composite.h5"
        if not os.path.isfile(trnsfm) or repeat:
            print("Registering image " + moving_nii + " to " + template_nii)
            antsreg.run()
        else:
            print("Warp file already exists at " + trnsfm)

        return trnsfm

    # Ants apply transforms to list
    # takes moving and reference niis, an output filename, plus a transform list
    # applys transform and saves output as output_nii
    def ants_apply(moving_nii, reference_nii, interp, transform_list, work_dir, invert_bool=False, repeat=False):
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
            output_nii[ind] = os.path.join(work_dir, os.path.basename(mvng).split(ext)[0] + '_warped.nii.gz')
            # do registration if not already done
            if not os.path.isfile(output_nii[ind]) or repeat:
                print("Creating warped image " + output_nii[ind])
                antsapply = ApplyTransforms(dimension=3,
                                            terminal_output='none',  # suppress terminal output
                                            input_image=mvng,
                                            reference_image=reference_nii,
                                            output_image=output_nii[ind],
                                            interpolation=interp,
                                            default_value=0,
                                            transforms=transform_list,
                                            invert_transform_flags=[invert_bool] * len(transform_list))
                print(antsapply.cmdline)
                antsapply.run()
            else:
                print("Transformed image already exists at " + output_nii[ind])
        # if only 1 label, don't return array
        if len(output_nii) == 1:
            output_nii = output_nii[0]
        return output_nii

    def norm_niis(ser_dict, repeat=False):
        # get list of filenames and normalize them
        for srs in ser_dict:
            fn = ser_dict[srs]["filename_reg"]
            if not fn == "None":
                normname = os.path.join(os.path.dirname(fn), os.path.basename(fn).split(".")[0] + "_norm.nii.gz")
                ser_dict[srs].update({"filename_norm": normname})
                # if normalized file doesn't exist, make it
                if not os.path.isfile(normname) or repeat:
                    norm_cmd = "ImageMath 3 "+ser_dict[srs]["filename_norm"]+" Normalize "+ser_dict[srs]["filename_reg"]
                    os.system(norm_cmd)
            else:
                ser_dict[srs].update({"filename_norm": "None"})
        return ser_dict

    def make_nii4d(ser_dict, repeat=False):
        # define vars
        files = []
        # get all normalized images and collect them in a list
        for srs in ser_dict:
            if not ser_dict[srs]["filename_norm"] == "None":
                files.append(ser_dict[srs]["filename_norm"])
        # get dirname from first normalized image, make nii4d name from this
        bdir = os.path.dirname(files[0])
        nii4d = os.path.join(bdir, "nii4d.nii.gz")
        # if nii4d doesn't exist, make it
        if not os.path.isfile(nii4d) or repeat:
            nii4dcmd = "ImageMath 4 " + nii4d + " TimeSeriesAssemble 1 1 " + " ".join(files)
            print(nii4dcmd)
            os.system(nii4dcmd)
        else:
            print("4D Nii already exists at " + nii4d)
        return ser_dict

    # get the series list, the filtered series list, and convert the appropriate dicoms to niis
    dicoms1, hdrs1, series1, dirs1 = get_series(dcm_dir)
    series_dict = filter_series(dicoms1, hdrs1, series1, dirs1, series_dict)
    series_dict = dcm_list_2_niis(series_dict, dcm_dir, rep)

    # print outputs of file conversion
    print("\nCONVERTED FILES LIST:")
    for ser in series_dict:
        print(ser + " = " + series_dict[ser]["filename"])

    # split b0 and dwi and discard b0 (if necessary)
    dwib0 = series_dict["DWI"]["filename"]
    nii = nib.load(dwib0)
    if len(nii.get_shape()) > 3:
        img = nii.get_data()
        img = np.squeeze(img[:, :, :, 0])
        affine = nii.get_affine()
        niiout = nib.Nifti1Image(img, affine)
        nib.save(niiout, dwib0)

    # register data together using reg_target as target, if its a file, use it, if not assume its a dict key
    print("\nREGISTERING IMAGES:")
    if os.path.isfile(reg_target):
        template = reg_target
    else:
        template = series_dict[reg_target]["filename"]
    for sers in series_dict:
        # if reg is false, or if there is no input file found, then just make the reg filename same as unreg filename
        if series_dict[sers]["filename"] == "None" or not series_dict[sers]["reg"]:
            series_dict[sers].update({"filename_reg": series_dict[sers]["filename"]})
            series_dict[sers].update({"transform": "None"})
        else:  # if reg is True, then do the registration
            moving = series_dict[sers]["filename"]
            transforms = affine_reg(moving, template, dcm_dir, rep)
            niiout = ants_apply(moving, template, 'Linear', transforms, dcm_dir, rep)
            series_dict[sers].update({"filename_reg": niiout})
            series_dict[sers].update({"transform": transforms})

    # normalize and nii4d registered data
    print("\nMAKING 4D NII:")
    series_dict = norm_niis(series_dict, rep)
    series_dict = make_nii4d(series_dict, rep)

    # now save metadata file, with np?
    dict_outfile = os.path.join(dcm_dir, "metadata.npy")
    np.save(dict_outfile, series_dict)

    return series_dict


# outside function code
dcmdir = ["/Users/edc15/Desktop/kp_test_data/0B33F047/2.16.840.1.114151.254008125529382180860998799084080385760171030"
          ]

# matching strings format is [[strs to match AND], OR [strs to match AND]
t1_str = [["t1"]]
t1_not = ["post", "gad", "flair"]

t2_str = [["t2"]]
t2_not = ["flair"]

flair_str = [["ax", "flair"]]
flair_not = []

dwi_str = [["ax", "dwi"]]
dwi_not = []

adc_str = [["ax", "adc"], ["apparent", "diffusion"]]
adc_not = ["exp"]

t1gad_str = [["spgr", "gad"], ["bravo", "gad"], ["+c", "t1"]]
t1gad_not = ["pre"]

swi_str = [["isi"], ["swan"]]
swi_not = ["ref",  "filt", "pha", "rf"]

mydict = {"T1": {"or": t1_str, "not": t1_not, "reg": False},
          "T2": {"or": t2_str, "not": t2_not, "reg": False},
          "FLAIR": {"or": flair_str, "not": flair_not, "reg": False},
          "DWI": {"or": dwi_str, "not": dwi_not, "reg": False},
          "ADC": {"or": adc_str, "not": adc_not, "reg": False},
          "T1gad": {"or": t1gad_str, "not": t1gad_not, "reg": False},
          "SWI": {"or": swi_str, "not": swi_not, "reg": False}
          }


# run read_dicom_dir in a loop with some basic stats to see what files we extracted
count = 0
stats = {}
for dcmd in dcmdir:
    # print some stating info
    print("\n\nWORKING DIRECTORY: " + dcmd)

    # reg_target can be a dict key or a file
    serdict = read_dicom_dir(dcmd, mydict,
                             reg_target="/Users/edc15/Desktop/strokecode/atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii",
                             rep=False)
    for s in serdict:
        if count == 0:
            stats.update({s: 0})
        if not serdict[s]["filename"] == "None":
            stats[s] = stats[s] + 1
    count = count + 1
print("\nSTATS:")
print(stats)
print("Total count: " + str(count))
"""