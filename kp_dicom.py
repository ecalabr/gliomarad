import os
import nibabel as nib
import numpy as np
from glob import glob
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.fsl import BET
from nipype.interfaces.fsl.maths import ApplyMask


# convert an entir folder of dicom files into individual nifits
def dicom2nii(study, nii_dir):
    # this file exists if conversion already done
    comp_file = os.path.join(nii_dir, "completed.txt")
    if not os.path.isfile(comp_file):
        print("- converting dicom directory " + study)
        # basic converter initiation and running
        converter = Dcm2niix()
        converter.inputs.source_dir = study
        converter.inputs.compress = "y"
        converter.inputs.output_dir = nii_dir
        converter.inputs.terminal_output = "allatonce"
        converter.anonymize = True
        converter.inputs.out_filename = "%i-%s-%q-"  # define output filenames
        converter.inputs.ignore_exception = True
        #print(converter.cmdline)
        converter.run()
        open(comp_file, 'a').close()
    else:
        print("- dicom directory already converted at " + nii_dir)


def find_dwi(out_dir, nii_dir):
    niis = glob(nii_dir + "/*.nii.gz")
    dwi = ""
    nifti = ""
    idno = ""
    ser = ""
    # work through each nifti in the output folder until one matches criteria
    for nii in niis:
        filename = os.path.basename(nii)
        if len(filename.split("-")) >= 3:
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
            print("- saved DWI for ID " + idno + " at " + dwi_out)
        else:
            print("- DWI already exists at " + dwi_out)
        dims = nifti.header["dim"]
    else:
        print("- could not find DWI for ID " + idno)
        ser = 0
        dims = 0
    # return some important info, series number and nifti dims
    return int(ser), dims, dwi_out


def find_adc(out_dir, nii_dir, dwi_series, dwi_dimensions):
    niis = glob(nii_dir + "/*.nii.gz")
    adc = ""
    nifti = ""
    idno = ""
    # work through each nifti in the output folder until one matches criteria
    for nii in niis:
        filename = os.path.basename(nii)
        if len(filename.split("-")) >= 3:
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
            print("- saved adc for ID " + idno + " at " + adc_out)
        else:
            print("- ADC already exists at " + adc_out)
    else:
        print("- could not find adc for ID " + idno)
    return adc_out


def find_flair(out_dir, nii_dir):
    niis = glob(nii_dir + "/*.nii.gz")
    flair = ""
    nifti = ""
    idno = ""
    # work through each nifti in the output folder until one matches criteria
    for nii in niis:
        filename = os.path.basename(nii)
        if len(filename.split("-")) >= 3:
            idno = filename.split("-")[0]
            ser = filename.split("-")[1]
            seq = filename.split("-")[2]
            nifti = nib.load(nii)
            # matching criteria are: series single digit, sequence is SE_IR, slices > 64
            if int(ser) < 100 and seq == "SE_IR" and nifti.header["dim"][3]>64:
                flair = nii
                break
    # if a dwi was found, then split b0 from it and save as dwi
    flair_out = os.path.join(out_dir, idno + "_flair.nii.gz")
    if flair:
        if not os.path.isfile(flair_out):
            nib.save(nifti, flair_out)
            print("- saved FLAIR for ID " + idno + " at " + flair_out)
        else:
            print("- FLAIR already exists at " + flair_out)
    else:
        print("- could not find FLAIR for ID " + idno)
    return flair_out


# ANTS translation
# takes moving and template niis and a work dir
# performs fast translation only registration and returns a list of transforms
def rigid_reg(moving_nii, template_nii, work_dir, repeat=False):
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
    antsreg.inputs.transforms=['Rigid']  # ['Rigid', 'Affine', 'SyN']
    antsreg.inputs.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    antsreg.inputs.initial_moving_transform_com=1  # use center of mass for initial transform
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.metric=['Mattes']  # ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight=[1.0]
    antsreg.inputs.number_of_iterations=[[1000, 500, 100, 50]]  # [100, 70, 50, 20]
    antsreg.inputs.convergence_threshold=[1e-07]
    antsreg.inputs.convergence_window_size=[10]
    antsreg.inputs.radius_or_number_of_bins=[32]  # 4
    antsreg.inputs.sampling_strategy=['Regular']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25]  # 1
    antsreg.inputs.shrink_factors=[[4, 3, 2, 1]]  # *3
    antsreg.inputs.transform_parameters=[(0.1,)]  # (0.1, 3.0, 0.0) # affine gradient step
    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        print("- Registering image " + moving_nii + " to " + template_nii)
        antsreg.run()
    else:
        print("- Warp file already exists at " + trnsfm)
    return trnsfm


# Fast ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def affine_reg(moving_nii, template_nii, work_dir, repeat=False):
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
    antsreg.inputs.terminal_output='none'
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
        print("- Registering image " + moving_nii + " to " + template_nii)
        antsreg.run()
    else:
        print("- Warp file already exists at " + trnsfm)
    return trnsfm


# Fast ants diffeomorphic registration
# takes moving and template niis and a work dir
# performs fast diffeomorphic registration and returns a list of transforms
def diffeo_reg(moving_nii, template_nii, work_dir, repeat=False):
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
    antsreg.inputs.terminal_output='none'
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
    antsreg.inputs.number_of_iterations=[[1000, 500, 250, 0], [50, 50, 0]]
    antsreg.inputs.convergence_threshold=[1e-07, 1e-07]
    antsreg.inputs.convergence_window_size=[5, 5]
    antsreg.inputs.radius_or_number_of_bins=[32, 32]
    antsreg.inputs.sampling_strategy=['Regular', 'None']  # 'None'
    antsreg.inputs.sampling_percentage=[0.25, 1]
    antsreg.inputs.transform_parameters=[(0.1,), (0.1, 3.0, 0.0)]
    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        print("- Registering image " + moving_nii + " to " + template_nii)
        antsreg.run()
    else:
        print("- Warp file already exists at " + trnsfm)
    return trnsfm


# Ants apply transforms to list
# takes moving and reference niis, an output filename, plus a transform list
# applys transform and saves output as output_nii
def ants_apply(moving_nii, reference_nii, transform_list, work_dir, interp="Linear", invert_bool=False, repeat=False):
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
            print("- Creating warped image " + output_nii[ind])
            antsapply = ApplyTransforms()
            antsapply.inputs.dimension=3
            antsapply.inputs.terminal_output='none'  # suppress terminal output
            antsapply.inputs.input_image=mvng
            antsapply.inputs.reference_image=reference_nii
            antsapply.inputs.output_image=output_nii[ind]
            antsapply.inputs.interpolation=interp
            antsapply.inputs.default_value=0
            antsapply.inputs.transforms=transform_list
            antsapply.inputs.invert_transform_flags=[invert_bool] * len(transform_list)
            antsapply.run()
        else:
            print("- Transformed image already exists at " + output_nii[ind])
    # if only 1 label, don't return array
    if len(output_nii) == 1:
        output_nii = output_nii[0]
    return output_nii


# make brain mask based on flair
def mask_flair(flair, work_dir):
    flairmask = os.path.join(work_dir, os.path.basename(work_dir).split(".")[0] + "_mask.nii.gz")
    if not os.path.isfile(flairmask):
        print("- making brain mask based on FLAIR")
        bet = BET()
        bet.inputs.in_file = flair
        bet.inputs.out_file = os.path.join(work_dir, os.path.basename(work_dir).split(".")[0])
        bet.inputs.mask = True
        bet.inputs.no_output = True
        bet.inputs.frac = 0.5
        bet.inputs.terminal_output = "none"
        _ = bet.run()
    return flairmask


# apply mask
def apply_mask(input_file, mask_file, work_dir):
    masked = os.path.join(work_dir, os.path.basename(input_file).split(".")[0] + "_masked.nii.gz")
    if not os.path.isfile(masked):
        print("- masking input file " + input_file)
        mask_cmd = ApplyMask()
        mask_cmd.inputs.in_file = input_file
        mask_cmd.inputs.mask_file = mask_file
        mask_cmd.inputs.out_file = masked
        mask_cmd.inputs.terminal_output = "none"
        mask_cmd.inputs.output_datatype = "input"
        _ = mask_cmd.run()
    return masked


def convert_dicoms(parent_dir):
    # get a list of all patient directories
    patients = glob(parent_dir + "/*/")
    # for each patient
    for n, patient in enumerate(patients, 1):
        # limit number of entries processed for testing purposes
        if n > N_LIMIT:
            return
        # for each series within each patient, remove prior converted or temp dirs
        studies = glob(patient + "/*/")
        try:
            studies.remove(os.path.join(patient, "converted/"))
            studies.remove(os.path.join(patient, "temp/"))
        except Exception:
            pass
        for study in studies:
            print("\nWorking on study " + study)

            # get base directory and make output directory
            output_dir = os.path.join(os.path.dirname(os.path.normpath(study)), "converted")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            # make temp dir
            temp_dir = os.path.join(os.path.dirname(os.path.normpath(study)), "temp")
            if not os.path.isdir(temp_dir):
                os.mkdir(temp_dir)

            # run processing pipeline
            dicom2nii(study, temp_dir)
            dwi_ser, dwi_dims, dwi = find_dwi(temp_dir, temp_dir)
            adc = find_adc(temp_dir, temp_dir, dwi_ser, dwi_dims)
            flair = find_flair(temp_dir, temp_dir)
            if os.path.isfile(dwi) and os.path.isfile(adc) and os.path.isfile(flair):
                flair_tnsfm = rigid_reg(flair, TEMPLATE, temp_dir)
                flair_warped = ants_apply(flair, TEMPLATE, flair_tnsfm, temp_dir)
                dwi_tnsfm = diffeo_reg(dwi, flair_warped, temp_dir)
                dwi_warped = ants_apply(dwi, flair_warped, dwi_tnsfm, temp_dir)
                adc_warped = ants_apply(adc, flair_warped, dwi_tnsfm, temp_dir)
                mask = mask_flair(flair_warped, temp_dir)
                flair_masked = apply_mask(flair_warped, mask, output_dir)
                dwi_masked = apply_mask(dwi_warped, mask, output_dir)
                adc_masked = apply_mask(adc_warped, mask, output_dir)
                print("- output files are:")
                print("FLAIR = " + flair_masked)
                print("DWI = " + dwi_masked)
                print("ADC = " + adc_masked)
            else:
                print("ERROR - COULD NOT FIND ALL FILES")



# GLOBAL VARS
TEMPLATE = "/Users/edc15/Desktop/strokecode/atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii"
KPDIR = "/Users/edc15/Desktop/kp_test_data/"
N_LIMIT = 200

convert_dicoms(KPDIR)

