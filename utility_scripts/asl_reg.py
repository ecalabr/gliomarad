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
def affine_reg(moving_nii1, moving_nii2, template_nii1, template_nii2, work_dir, diffeo=False, repeat=False):

    # get basenames
    moving_name = os.path.basename(moving_nii2).split(".")[0]
    template_name = os.path.basename(template_nii1).split(".")[0]
    idno = moving_name.rsplit('_')[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + idno + "_FLAIR_w_")

    # registration setup
    if diffeo:
        """--transform SyN[ 0.1, 3.0, 0.0 ]
        --metric Mattes[ fixed, moving, 1, 32, None, 1 ]
        --convergence [ 50x50x25, 1e-07, 5 ]
        --smoothing-sigmas 2.0x1.0x0.0mm
        --shrink-factors 4x2x1"""
        antsreg = Registration()
        antsreg.inputs.args = '--float'
        antsreg.inputs.fixed_image = [template_nii2, template_nii1]
        antsreg.inputs.moving_image = [moving_nii2, moving_nii1]
        antsreg.inputs.output_transform_prefix = outprefix
        antsreg.inputs.num_threads = multiprocessing.cpu_count()
        antsreg.inputs.smoothing_sigmas = [[6, 4, 1, 0], [2, 1, 0], [2, 1, 0]]
        antsreg.inputs.sigma_units = ['vox'] * 3
        antsreg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        antsreg.terminal_output = 'none'
        antsreg.inputs.use_histogram_matching = True
        antsreg.inputs.write_composite_transform = True
        antsreg.inputs.initial_moving_transform_com = 0
        antsreg.inputs.winsorize_lower_quantile = 0.005
        antsreg.inputs.winsorize_upper_quantile = 0.995
        antsreg.inputs.metric = [['Mattes', 'Mattes'], ['Mattes', 'Mattes'], ['Mattes', 'Mattes']]
        antsreg.inputs.metric_weight = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        antsreg.inputs.number_of_iterations = [[1000, 1000, 1000, 1000], [50, 50, 25], [50, 50, 25]]
        antsreg.inputs.convergence_threshold = [1e-07, 1e-07, 1e-07]
        antsreg.inputs.convergence_window_size = [10, 10, 5]
        antsreg.inputs.radius_or_number_of_bins = [[32, 32], [32, 32], [32, 32]]
        antsreg.inputs.sampling_strategy = [['Regular', 'Regular'], ['Regular', 'Regular'], ['None', 'None']]
        antsreg.inputs.sampling_percentage = [[0.5, 0.5], [0.5, 0.5], [1, 1]]
        antsreg.inputs.shrink_factors = [[4, 3, 2, 1], [4, 2, 1], [4, 2, 1]]
        antsreg.inputs.transform_parameters = [(0.2,), (0.1,), (0.1,)]
    else:
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
    parser.add_argument('--direc', default=None,
                        help="Path to data directory")
    parser.add_argument('--f1', default='DWI_wm',
                        help="Fixed image 1 file suffix")
    parser.add_argument('--f2', default='DWI_w',
                        help="Fixed image 2 file suffix")
    parser.add_argument('--m1', default='ASL',
                        help="Moving image 1 file suffix")
    parser.add_argument('--m2', default='ASL_anat',
                        help="Moving image 2 file suffix (used for transform name)")
    parser.add_argument('--mask', default='combined_brain_mask',
                        help="Mask file suffix")
    parser.add_argument('--diffeo', action="store_true", default=False,
                        help="Use this flag to add diffeomorphic reg step")
    parser.add_argument('--overwrite', action="store_false", default=True,
                        help="Use this flag to overwrite existing data")

    # check inputs and get files
    args = parser.parse_args()
    assert os.path.isdir(args.direc), "Data directory does not exist at: {}".format(args.direc)
    fixed1 = glob(args.direc + '/*' + args.f1 + '.nii.gz')[0]
    if not fixed1:
        raise ValueError("Fixed image 1 does not exist using suffix: {}".format(args.f1))
    fixed2 = glob(args.direc + '/*' + args.f2 + '.nii.gz')[0]
    if not fixed2:
        raise ValueError("Fixed image 2 does not exist using suffix: {}".format(args.f2))
    moving1 = glob(args.direc + '/*' + args.m1 + '.nii.gz')[0]
    if not moving1:
        raise ValueError("Moving image 1 does not exist using suffix: {}".format(args.m1))
    moving2 = glob(args.direc + '/*' + args.m2 + '.nii.gz')[0]
    if not moving2:
        raise ValueError("Moving image 2 does not exist using suffix: {}".format(args.m2))
    mask = glob(args.direc + '/*' + args.mask + '.nii.gz')[0]
    if not mask:
        raise ValueError("Mask image does not exist using suffix: {}".format(args.mask))

    # do work
    tnsfm = affine_reg(moving1, moving2, fixed1, fixed2, args.direc, diffeo=args.diffeo, repeat=args.overwrite)
    print("Tansform created at: {}".format(tnsfm))
    warped = ants_apply(moving1, fixed1, 'Linear', tnsfm, args.direc, invert_bool=False, repeat=args.overwrite)
    print("Warped image created at: {}".format(warped))
    masked = apply_mask(warped, mask, repeat=args.overwrite)
    print("Masked image created at: {}".format(masked))
