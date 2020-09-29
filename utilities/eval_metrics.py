from nipype.interfaces.ants import MeasureImageSimilarity
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as struct_sim


# get built in locals
start_globals = list(globals().keys())


# perform image comparrison using neighborhood CC
def cc(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
        sim = MeasureImageSimilarity()
        if not verbose:
            sim.terminal_output = 'allatonce'
        sim.inputs.dimension = 3
        sim.inputs.metric = 'CC'
        sim.inputs.fixed_image = true_nii
        sim.inputs.moving_image = pred_nii
        sim.inputs.metric_weight = 1.0
        sim.inputs.radius_or_number_of_bins = 4
        sim.inputs.sampling_strategy = 'None'  # None = dense sampling
        sim.inputs.sampling_percentage = 1.0
        if mask:
            sim.inputs.fixed_image_mask = mask_nii
            sim.inputs.moving_image_mask = mask_nii
        if verbose:
            print(sim.cmdline)

        return np.abs(sim.run().outputs.similarity)


# perform image comparrison using histogram MI
def mi(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
        sim = MeasureImageSimilarity()
        if not verbose:
            sim.terminal_output = 'allatonce'
        sim.inputs.dimension = 3
        sim.inputs.metric = 'MI'
        sim.inputs.fixed_image = true_nii
        sim.inputs.moving_image = pred_nii
        sim.inputs.metric_weight = 1.0
        sim.inputs.radius_or_number_of_bins = 32
        sim.inputs.sampling_strategy = 'None'  # None = dense sampling
        sim.inputs.sampling_percentage = 1.0
        if mask:
            sim.inputs.fixed_image_mask = mask_nii
            sim.inputs.moving_image_mask = mask_nii
        if verbose:
            print(sim.cmdline)

        return np.abs(sim.run().outputs.similarity)


# perform image comparrison using histogram MI
def mse(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    sim = MeasureImageSimilarity()
    if not verbose:
        sim.terminal_output = 'allatonce'
    sim.inputs.dimension = 3
    sim.inputs.metric = 'MeanSquares'
    sim.inputs.fixed_image = true_nii
    sim.inputs.moving_image = pred_nii
    sim.inputs.metric_weight = 1.0
    sim.inputs.radius_or_number_of_bins = 32
    sim.inputs.sampling_strategy = 'None'  # None = dense sampling
    sim.inputs.sampling_percentage = 1.0
    if mask:
        sim.inputs.fixed_image_mask = mask_nii
        sim.inputs.moving_image_mask = mask_nii
    if verbose:
        print(sim.cmdline)

    return np.abs(sim.run().outputs.similarity)


# mean absolute percentage error
def mape(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # divide by zero constant
    EPSILON = 1e-10
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        mask_img = np.ones_like(true_img, dtype=bool)
    # scale to 12 bit range
    true_img = true_img * (4095 / np.max(true_img))
    pred_img = pred_img * (4095 / np.max(pred_img))
    # verbosity
    if verbose:
        print("Calculating MAPE on 12 bit range scaled images")

    return np.mean((np.fabs((true_img - pred_img)) / (np.fabs(true_img) + EPSILON))[mask_img])


# structural similarity index
def ssim(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
        true_img = true_img * mask_img
        pred_img = pred_img * mask_img
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return struct_sim(true_img, pred_img, data_range=true_img.max() - true_img.min())


def metric_picker(metric, true_nii, pred_nii, mask_nii, mask=False, verbose=False):

    # sanity checks
    if not isinstance(metric, str):
        raise ValueError("Metric parameter must be a string")

    # check for specified loss method and error if not found
    if metric.lower() in globals():
        metric_val = globals()[metric.lower()](true_nii, pred_nii, mask_nii, mask, verbose)
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified metric: '{}' is not one of the available methods: {}".format(metric, methods))

    return metric_val
