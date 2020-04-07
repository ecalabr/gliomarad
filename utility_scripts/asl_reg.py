import os
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.fsl.maths import ApplyMask
import multiprocessing
import argparse
from glob import glob

########################## define functions ##########################
# Fast ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def affine_reg(moving_nii1, moving_nii2, template_nii1, template_nii2, work_dir, repeat=False):

    # get basenames
    moving_name = os.path.basename(moving_nii2).split(".")[0]
    template_name = os.path.basename(template_nii1).split(".")[0]
    idno = moving_name.rsplit('_')[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + idno + "_FLAIR_w_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args='--float'
    antsreg.inputs.fixed_image=[template_nii2, template_nii1]
    antsreg.inputs.moving_image=[moving_nii2, moving_nii1]
    antsreg.inputs.output_transform_prefix=outprefix
    antsreg.inputs.num_threads=multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas=[[6, 4, 1, 0], [2, 1, 0]]
    antsreg.inputs.sigma_units=['vox'] * 2
    antsreg.inputs.transforms=['Rigid', 'Affine']
    antsreg.terminal_output='none'
    antsreg.inputs.use_histogram_matching=True
    antsreg.inputs.write_composite_transform=True
    antsreg.inputs.initial_moving_transform_com = 0
    antsreg.inputs.winsorize_lower_quantile=0.005
    antsreg.inputs.winsorize_upper_quantile=0.995
    antsreg.inputs.metric=[['Mattes', 'Mattes'], ['Mattes', 'Mattes']]
    antsreg.inputs.metric_weight=[[0.5, 0.5], [0.5, 0.5]]
    antsreg.inputs.number_of_iterations=[[1000, 1000, 1000, 1000], [50, 50, 25]]
    antsreg.inputs.convergence_threshold=[1e-07, 1e-07]
    antsreg.inputs.convergence_window_size=[10, 10]
    antsreg.inputs.radius_or_number_of_bins=[[32, 32], [32, 32]]
    antsreg.inputs.sampling_strategy=[['Regular', 'Regular'], ['Regular', 'Regular']]
    antsreg.inputs.sampling_percentage=[[0.5, 0.5], [0.5, 0.5]]
    antsreg.inputs.shrink_factors=[[4, 3, 2, 1], [4, 2, 1]]
    antsreg.inputs.transform_parameters=[(0.2,), (0.1,)]

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        print("Registering image " + moving_nii1 + " to " + template_nii1)
        print(antsreg.cmdline)
        antsreg.run()
    else:
        print("Warp file already exists at " + trnsfm)
        print(antsreg.cmdline)
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
            print("Creating warped image " + output_nii[ind])
            print(antsapply.cmdline)
            antsapply.run()
        else:
            print("Transformed image already exists at " + output_nii[ind])
            print(antsapply.cmdline)
    # if only 1 label, don't return array
    if len(output_nii) == 1:
        output_nii = output_nii[0]
    return output_nii

def apply_mask(input_file, mask_file, repeat=False):

    # define output
    output_file = input_file.rsplit('.nii')[0] + 'm.nii.gz'

    # prep command line regardless of whether or not work will be done
    mask_cmd = ApplyMask()
    mask_cmd.inputs.in_file = input_file
    mask_cmd.inputs.mask_file = mask_file
    mask_cmd.inputs.out_file = output_file
    mask_cmd.terminal_output = "none"
    print(mask_cmd.cmdline)
    if not os.path.isfile(output_file) or repeat:
        _ = mask_cmd.run()

    return output_file

########################## executed  as script ##########################
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")

    # check inputs and get files
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Data directory does not exist at: {}".format(args.data_dir)
    fixed1 = glob(args.data_dir + '/*_DWI_wm.nii.gz')[0]
    assert os.path.isfile(fixed1), "File does not exist at: {}".format(fixed1)
    fixed2 = glob(args.data_dir + '/*_DWI_w.nii.gz')[0]
    assert os.path.isfile(fixed2), "File does not exist at: {}".format(fixed2)
    moving1 = glob(args.data_dir + '/*_ASL.nii.gz')[0]
    assert os.path.isfile(moving1), "File does not exist at: {}".format(moving1)
    moving2 = glob(args.data_dir + '/*_ASL_anat.nii.gz')[0]
    assert os.path.isfile(moving2), "File does not exist at: {}".format(moving2)
    mask = glob(args.data_dir + '/*_combined_brain_mask.nii.gz')[0]
    assert os.path.isfile(mask), "File does not exist at: {}".format(mask)

    # do work
    tnsfm = affine_reg(moving1, moving2, fixed1, fixed2, args.data_dir, repeat=True)
    print("Tansform created at: {}".format(tnsfm))
    warped = ants_apply(moving1, fixed1, 'Linear', tnsfm, args.data_dir, invert_bool=False, repeat=True)
    print("Warped image created at: {}".format(warped))
    masked = apply_mask(warped, mask, repeat=True)
    print("Masked image created at: {}".format(masked))
