import os
import dicom
import nibabel as nib
import numpy as np
from glob import glob
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
import nibabel as nib


# wrapper function for reading a complete dicom directory with/without registration to one of the images
def read_dicom_dir(dcm_dir, series_dict=(), reg_target="DWI", rep=False):
    """
    This function takes a DICOM directory for a single patient study, and converts all appropriate series to NIFTI.
    It will also register any/all datasets to a given target.
    :param dcm_dir: the full path to the dicom directory as a string
    :param series_dict: a dict containing the name of each series, the substrings used to search for it, and reg bool
    :param reg_target: a string specifying the series to use as template for registration
    :return: returns the path to a metadata file dict that contains lots of relevant info
    Note this forces axial plane image orientation. To change this, edit the filter_series function
    """
    # first check for series_dict
    if not series_dict:
        series_dict = {"T1": {"or": [["ax", "t1"]], "not": ["post", "cor", "gad"], "reg": True},
                       "T2": {"or": [["ax", "t2"]], "not": ["cor"], "reg": True},
                       "FLAIR": {"or": [["ax", "flair"]], "not": ["cor"], "reg": True},
                       "DWI": {"or": [["ax", "dwi"]], "not": ["cor"], "reg": False},
                       "ADC": {"or": [["ax", "adc"], ["apparent", "diffusion"]], "not": ["exp", "cor"], "reg": False},
                       "T1gad": {"or": [["ax", "t1", "gad"]], "not": ["pre"], "reg": True}
                       }

    # function to get complete series list from dicom directory
    def get_series(dicom_dir):
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
dcmdir = ["/Users/edc15/Desktop/idh1_mutant_data/57c3d91de7a83a10ac8810e663dc5eda",
          "/Users/edc15/Desktop/idh1_mutant_data/73d72965b4bebf934f1344a88813b897",
          "/Users/edc15/Desktop/idh1_mutant_data/a2d3048dd62227a410df5f7f1fdefea9",
          "/Users/edc15/Desktop/idh1_mutant_data/c19ab1f810cc8261547cdc5c0239f7b0"
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

mydict = {"T1": {"or": t1_str, "not": t1_not, "reg": True},
          "T2": {"or": t2_str, "not": t2_not, "reg": True},
          "FLAIR": {"or": flair_str, "not": flair_not, "reg": True},
          "DWI": {"or": dwi_str, "not": dwi_not, "reg": True},
          "ADC": {"or": adc_str, "not": adc_not, "reg": True},
          "T1gad": {"or": t1gad_str, "not": t1gad_not, "reg": True},
          "SWI": {"or": swi_str, "not": swi_not, "reg": True}
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
