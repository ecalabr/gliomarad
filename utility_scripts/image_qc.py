import os
from glob import glob
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="/media/ecalabr/scratch/qc_complete",
                    help="Path to data directory")
parser.add_argument('--ij_dir', default="/home/ecalabr/ImageJ",
                    help="Path to imageJ")
parser.add_argument('--skip', default=0,
                    help="Index of directories to start processing at")

if __name__ == '__main__':
    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    ij_dir = args.ij_dir
    assert os.path.isdir(ij_dir), "ImageJ directory not found at {}".format(ij_dir)
    ij_java = os.path.join(ij_dir, "jre/bin/java")
    ij_jar = os.path.join(ij_dir, "ij.jar")
    start = int(args.skip)

    # list all subdirs with the processed data
    direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
    direcs = sorted(direcs, key=lambda x: int(os.path.basename(x)))

    # set skip for already QCed images here
    if start:
        skip = int(start)
    else:
        skip = 0
    n_total = len(direcs)
    print("Found a total of " + str(n_total) + " directories")
    direcs = direcs[skip:]
    if isinstance(direcs, str):
        direcs = [direcs]

    # make sure all files exist, if they do then open in imagej
    for i, direc in enumerate(direcs, 1):
        adc = glob(direc + "/*ADC_wm.nii.gz")
        asl = glob(direc + "/*ASL_wm.nii.gz")
        fa = glob(direc + "/*DTI_eddy_FA_wm.nii.gz")
        md = glob(direc + "/*DTI_eddy_MD_wm.nii.gz")
        dwi = glob(direc + "/*DWI_wmtb.nii.gz")
        flair = glob(direc + "/*FLAIR_wmtb.nii.gz")
        swi = glob(direc + "/*SWI_wmtb.nii.gz")
        t1 = glob(direc + "/*T1_wmtb.nii.gz")
        t1gad = glob(direc + "/*T1gad_wmtb.nii.gz")
        t2 = glob(direc + "/*T2_wmtb.nii.gz")
        seg = glob(direc + "/*tumor_seg.nii.gz")
        #img_list = [adc, asl, fa, dwi, flair, swi, t1, t1gad, t2, seg]
        img_list = [t1, t1gad, t2, flair, dwi, asl, swi, md, seg]
        # alternative for breast MRI project
        dwi = glob(direc + "/*DWI_w.nii.gz")
        t1fs = glob(direc + "/*T1FS.nii.gz")
        T1gad = glob(direc + "/*T1gad_w.nii.gz")
        T2FS = glob(direc + "/*T2FS_w.nii.gz")
        T1 = glob(direc + "/*T1_w.nii.gz")
        #img_list = [dwi, t1fs, T1gad, T2FS, T1]

        if all(img_list):
            print("Directory " + direc + " is complete!")
            # build macro for imageJ
            macro = []
            for item in img_list: # open images
                macro.append('open("' + item[0] + '");')
            macro.append('run("Tile");')
            # add for loop for setting slice and windowing
            macro.append("for (i=0;i<nImages;i++) {")
            macro.append("     selectImage(i+1);")
            macro.append("     setSlice(nSlices/2);")
            macro.append('     run("Enhance Contrast", "saturated=0.35");}')
            # run synchronize windows
            macro.append('run("Synchronize Windows");')
            # write macro
            macro_file = os.path.join(data_dir, 'qc_macro.ijm')
            with open(macro_file, 'w') as f:
                for item in macro:
                    f.write("%s\n" % item)

            # Run macro
            cmd = ij_java+ ' -Xmx5000m -jar ' + ij_jar + ' -ijpath ' + ij_dir + ' -macro ' + macro_file
            os.system(cmd)
        else:
            print("Directory " + direc + " is missing some sequences")
        print("Done with study " + os.path.basename(direc) + ": " + str(i+skip) + " of " + str(n_total))