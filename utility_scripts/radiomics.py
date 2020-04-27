import argparse
import os
from glob import glob
import csv
import subprocess
import multiprocessing


# define functions
def radiomics_extract(data_dirs, output_dir, norm_images, nonorm_images, mask, label_values):
    # define start and stop values for data to keep in the radiomics spreadsheet
    shape_s = 25
    shape_e = 39

    # get list of all images
    images = []
    if norm_images:
        images = images + norm_images
    if nonorm_images:
        images = images + nonorm_images

    # make the csvs for batch processing per directory for NON-normalized files
    if nonorm_images:
        for direc in data_dirs:
            output = [['Image', 'Mask', 'Label']]
            output_csv = os.path.join(output_dir, os.path.basename(direc) + '_radiomics_nonorm.csv')
            if not os.path.isfile(output_csv):
                for image in nonorm_images:
                    for lab in label_values:
                        line = [glob(direc + '/*' + image + '.nii.gz')[0], glob(direc + '/*' + mask + '.nii.gz')[0],
                                lab]
                        output.append(line)
                with open(output_csv, 'w+') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL, quotechar="\"")
                    wr.writerows(output)

    # make the csvs for batch processing per directory for NORMALIZED files
    if norm_images:
        for direc in data_dirs:
            output = [['Image', 'Mask', 'Label']]
            output_csv = os.path.join(output_dir, os.path.basename(direc) + '_radiomics_norm.csv')
            if not os.path.isfile(output_csv):
                for image in norm_images:
                    for lab in label_values:
                        line = [glob(direc + '/*' + image + '.nii.gz')[0], glob(direc + '/*' + mask + '.nii.gz')[0],
                                lab]
                        output.append(line)
                with open(output_csv, 'w+') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL, quotechar="\"")
                    wr.writerows(output)

    # run batch processing for NON-normalized files
    batch_csvs = glob(output_dir + '/*_radiomics_nonorm.csv')
    if batch_csvs:
        batch_csvs.sort()
        for batch in batch_csvs:
            output = batch.rsplit('_', 1)[0] + '_output_nonorm.csv'
            cmd = 'pyradiomics ' + batch + ' -o ' + output + ' -f csv --setting "normalize:False" -j ' + str(
                multiprocessing.cpu_count())
            if not os.path.isfile(output):
                print(cmd)
                subprocess.call(cmd, shell=True)

    # run batch processing for NORMALIZED files
    batch_csvs = glob(output_dir + '/*_radiomics_norm.csv')
    if batch_csvs:
        batch_csvs.sort()
        for batch in batch_csvs:
            output = batch.rsplit('_', 1)[0] + '_output_norm.csv'
            cmd = 'pyradiomics ' + batch + ' -o ' + output + ' -f csv --setting "normalize:True" -j ' + str(
                multiprocessing.cpu_count())
            if not os.path.isfile(output):
                print(cmd)
                subprocess.call(cmd, shell=True)

    # combined normalized and non-normalized outputs
    norm_outputs = glob(output_dir + '/*_output_norm.csv')
    nonorm_outputs = glob(output_dir + '/*_output_nonorm.csv')
    norm_outputs.sort()
    nonorm_outputs.sort()
    for ind, it in enumerate(norm_outputs):
        csvout_file = it.rsplit('_', 2)[0] + '_output.csv'
        if not os.path.isfile(csvout_file):
            csvout = []
            with open(it, 'r') as f1:
                norm_data = csv.reader(f1)
                for line in norm_data:
                    csvout.append(line)
            if nonorm_outputs:
                with open(nonorm_outputs[ind], 'r') as f2:
                    nonorm_data = csv.reader(f2)
                    for n, line in enumerate(nonorm_data, 1):
                        if n > 1:
                            csvout.append(line)
            with open(csvout_file, 'w+') as f3:
                wr = csv.writer(f3, quoting=csv.QUOTE_MINIMAL, quotechar="\"")
                wr.writerows(csvout)

    # combine output csvs into a single csv with one row per patient
    final_out = os.path.join(output_dir, 'combined_output.csv')
    if not os.path.isfile(final_out):
        output_csvs = glob(output_dir + '/*_output.csv')
        output_csvs.sort()
        combined_out = []
        for ind, output in enumerate(output_csvs):
            with open(output, 'r') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                data = list(reader)

            # make header line and add to final output
            if ind == 0:
                headerline = ['ID']
                # make shape header
                for lab in label_values:
                    headerline = headerline + [it + '_' + mask + '_' + str(lab) for it in data[0][shape_s:shape_e]]
                for image in images:
                    for lab in label_values:
                        headerline = headerline + [image + '_' + it + '_' + mask + '_' + str(lab) for it in
                                                   data[0][shape_e:]]
                combined_out.append(headerline)

            # make data line and add to final output
            # first add ID
            line = [output.rsplit('/', 1)[1].split('_')[0]]
            # get ROI shape data once and append to beginning of line
            for x, lab in enumerate(label_values, 1):
                line = line + [val for val in data[x][shape_s:shape_e]]
            # get image vale data for each image (skipping over shape data, since its the same for all images)
            for n, it in enumerate(data[1:]):
                line = line + it[shape_e:]
            combined_out.append(line)

        # write final combined output
        with open(final_out, 'w+') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            wr.writerows(combined_out)

    return final_out


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False,
                        help="List all directories and exit")
    parser.add_argument('--direc', default=None,
                        help="Optionally name a specific directory to edit")
    parser.add_argument('--output_dir', default=None,
                        help="Path to output directory")
    parser.add_argument('--mask', default="tumor_seg.nii.gz",
                        help="Suffix of the mask to extract radiomics data from")
    parser.add_argument('--maskval', default=1,
                        help="Mask label to extract radiomics data from")
    parser.add_argument('--norm', default="T1gad_w.nii.gz",
                        help="Suffix of the images to be normalized before radiomics feature extraction")
    parser.add_argument('--no_norm', default=None,
                        help="Suffix of the images that don't need normalization before radiomics feature extraction")

    # check arguments
    args = parser.parse_args()

    # handle data_dir, start, end, direc, and list arugments
    data_dir = args.data_dir
    spec_direc = args.direc
    if spec_direc:
        assert os.path.isdir(spec_direc), "Specified directory does not exist at {}".format(spec_direc)
    else:
        assert data_dir, "Must specify data directory using param --data_dir"
        assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)

    start = args.start
    end = args.end

    # handle specific directory
    if spec_direc:
        my_direcs = [spec_direc]
    else:
        # list all subdirs with the processed data
        my_direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
        my_direcs = sorted(my_direcs, key=lambda x: int(os.path.basename(x)))

        # set start and stop for subset/specific diectories only using options below
        if end:
            my_direcs = my_direcs[int(start):int(end) + 1]
        else:
            my_direcs = my_direcs[int(start):]
    if isinstance(my_direcs, str):
        my_direcs = [my_direcs]

    # handle list flag
    if args.list:
        for i, item in enumerate(my_direcs, 0):
            print(str(i) + ': ' + item)
        exit()

    # handle output_dir argument
    if args.output_dir:
        assert os.path.isdir(args.output_dir), "No output directory found at {}".format(args.output_dir)
    else:
        raise ValueError("Must specify output directory using --output_dir")

    # handle mask argument
    args.mask = args.mask.split('.nii.gz')[0] if args.mask.endswith('.nii.gz') else args.mask

    # handle maskval argument
    if isinstance(args.maskval, int):
        args.maskval = [args.maskval]
    if isinstance(args.mask, str):
        try:
            args.maskval = int(args.maskval)
        except:
            try:
                args.maskval = [int(i) for i in args.maskval.split(',')]
            except:
                raise ValueError("Maskval argument must be int or comma separated ints but is {}".format(args.maskval))

    # handle norm argument
    if args.norm:
        args.norm = args.norm.split(',')
        args.norm = [item.split('.nii.gz')[0] if item.endswith('.nii.gz') else item for item in args.norm]

    # handle no_norm argument
    if args.no_norm:
        args.no_norm = args.no_norm.split(',')
        args.no_norm = [item.split('.nii.gz')[0] if item.endswith('.nii.gz') else item for item in args.no_norm]

    output_data = radiomics_extract(my_direcs, args.output_dir, args.norm, args.no_norm, args.mask, args.maskval)
