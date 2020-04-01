""" checks if specified images are present in data directories, then displays them in imageJ and allows annotation """

import os
from glob import glob
import argparse
import time

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="/media/ecalabr/scratch/qc_complete",
                    help="Path to data directory")
parser.add_argument('--ij_dir', default="/home/ecalabr/ImageJ",
                    help="Path to imageJ")
parser.add_argument('--start', default=0,
                    help="Index of directories to start processing at")
parser.add_argument('--end', default=None,
                    help="Index of directories to end processing at")
parser.add_argument('--list', action="store_true", default=False,
                    help="List all directories and exit")
parser.add_argument('--m', action="store_true", default=False,
                    help="Include directories with missing data")
parser.add_argument('--direc', default=None,
                    help="Optionally name a specific directory to edit")

if __name__ == '__main__':
    # get arguments and check them
    args = parser.parse_args()
    spec_direc = args.direc
    data_dir = args.data_dir
    if spec_direc:
        assert os.path.isdir(spec_direc), "Specified directory does not exist at {}".format(spec_direc)
    else:
        assert data_dir, "Must specify data directory using param --data_dir"
        assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    ij_dir = args.ij_dir
    assert os.path.isdir(ij_dir), "ImageJ directory not found at {}".format(ij_dir)
    ij_java = os.path.join(ij_dir, "jre/bin/java")
    ij_jar = os.path.join(ij_dir, "ij.jar")
    start = args.start
    end = args.end

    # handle specific directory
    if spec_direc:
        direcs = [spec_direc]
    else:
        # list all subdirs with the processed data
        direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
        direcs = sorted(direcs, key=lambda x: int(os.path.basename(x)))

    # define ij macro out
    macro_file = os.path.join(data_dir, 'qc_macro.ijm')

    # handle list flag
    if args.list:
        for i, item in enumerate(direcs, 0):
            print(str(i) + ': ' + item)
        exit()

    # set start and stop for subset/specific diectories only using options below
    if end:
        direcs = direcs[int(start):int(end) + 1]
    else:
        direcs = direcs[int(start):]
    n_total = len(direcs)
    print("Performing image QC for a total of " + str(n_total) + " directories")
    if isinstance(direcs, str):
        direcs = [direcs]

    # make an output text file for QC results
    textout = os.path.join(data_dir, 'QC_log.txt')
    with open(textout, 'a+') as f:
        f.write("############ QC LOG " + time.strftime("%m-%d-%Y %H:%M") + " ############\n")

    # make sure all files exist, if they do then open in imagej
    for i, direc in enumerate(direcs, 1):
        adc = {'ADC': glob(direc + "/*ADC_wm.nii.gz")}
        asl = {'ASL': glob(direc + "/*ASL_wm.nii.gz")}
        fa = {'DTI': glob(direc + "/*DTI_eddy_FA_wm.nii.gz")}
        md = {'DTI': glob(direc + "/*DTI_eddy_MD_wm.nii.gz")}
        dwi = {'DWI': glob(direc + "/*DWI_wmtb.nii.gz")}
        flair = {'FLAIR': glob(direc + "/*FLAIR_wmtb.nii.gz")}
        swi = {'SWI': glob(direc + "/*SWI_wmtb.nii.gz")}
        t1 = {'T1': glob(direc + "/*T1_wmtb.nii.gz")}
        t1gad = {'T1gad': glob(direc + "/*T1gad_wmtb.nii.gz")}
        t2 = {'T2': glob(direc + "/*T2_wmtb.nii.gz")}
        seg = {'Seg': glob(direc + "/*tumor_seg.nii.gz")}
        #img_list = [adc, asl, fa, dwi, flair, swi, t1, t1gad, t2, seg]
        img_list = [t1, t1gad, t2, flair, dwi, asl, swi, fa, seg]
        # alternative for breast MRI project
        dwi = {'DWI': glob(direc + "/*DWI_w.nii.gz")}
        t1fs = {'T1FS': glob(direc + "/*T1FS.nii.gz")}
        T1gad = {'T1gad': glob(direc + "/*T1gad_w.nii.gz")}
        T2FS = {'T2FS': glob(direc + "/*T2FS_w.nii.gz")}
        T1 = {'T1': glob(direc + "/*T1_w.nii.gz")}
        #img_list = [dwi, t1fs, T1gad, T2FS, T1]

        # check if all data present
        compl = False
        missing_str = ''
        if all([item.values()[0] for item in img_list]):
            print("Directory " + direc + " is complete - (study " + str(i) + " of " + str(n_total) + ")")
            compl = True
        else:
            print("Directory " + direc + " is missing some sequences - (study " + str(i) + " of " + str(n_total) + ")")
            # write standard missing data line to log file if no QC performed, otherwise add to QC string in macro
            missing_str = "Missing data -"
            for item in img_list:
                if not item.values()[0]:
                    missing_str = missing_str + " " + item.keys()[0]
            missing_str = missing_str + ". "
            if not args.m:
                with open(textout, 'a+') as f:
                    f.write(os.path.basename(direc) + ": " + missing_str + "\n")

        # if all images are present or if include missing data flag was passed then run QC
        if compl or args.m:
            # build macro for imageJ image QC
            macro = []
            for item in [ser.values()[0] for ser in img_list if ser.values()[0] and os.path.isfile(ser.values()[0][0])]: # open images
                macro.append('open("' + item[0] + '");')
            macro.append('run("Tile");')
            # add for loop for setting slice and windowing
            macro.append("for (i=0;i<nImages;i++) {")
            macro.append("     selectImage(i+1);")
            macro.append("     setSlice(nSlices/2);")
            macro.append('     run("Enhance Contrast", "saturated=0.35");}')
            # run synchronize windows
            macro.append('run("Synchronize Windows");')
            # qc user input
            macro.append('qc_default = "Good";')
            macro.append('Dialog.createNonBlocking("QC");')
            macro.append('Dialog.addString("QC", qc_default);')
            macro.append('Dialog.show();')
            macro.append('qc_str = Dialog.getString();')
            macro.append('qc_str = "' + os.path.basename(direc) + ': " + "' + missing_str + '" + qc_str')
            macro.append('File.append(qc_str, "' + textout + '");')
            macro.append('run("Quit");')
            # write macro
            with open(macro_file, 'w') as f:
                for item in macro:
                    f.write("%s\n" % item)
            # Run macro
            cmd = ij_java+ ' -Xmx5000m -jar ' + ij_jar + ' -ijpath ' + ij_dir + ' -macro ' + macro_file
            os.system(cmd)
    # clean up
    if os.path.isfile(macro_file):
        os.remove(macro_file)