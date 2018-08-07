import os
from glob import glob

# main directory containing subdirs with processed data
data_dir = "/media/ecalabr/data/gbm"

# list all subdirs with the processed data
direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
direcs = sorted(direcs, key=lambda x: int(os.path.basename(x)))

# set skip for already QCed images here
skip = 0
n_total = len(direcs)
direcs = direcs[skip:]
if isinstance(direcs, str):
    direcs = [direcs]

# manually set direcs
#direcs = ['/media/ecalabr/data/gbm/12184393']

# make sure all files exist, if they do then open in imagej
for i, direc in enumerate(direcs, 1):
    adc = glob(direc + "/*ADC_wm.nii.gz")
    asl = glob(direc + "/*ASL_wm.nii.gz")
    fa = glob(direc + "/*DTI_eddy_FA_wm.nii.gz")
    dwi = glob(direc + "/*DWI_wm.nii.gz")
    flair = glob(direc + "/*FLAIR_wm.nii.gz")
    swi = glob(direc + "/*SWI_wm.nii.gz")
    t1 = glob(direc + "/*T1_wm.nii.gz")
    t1gad = glob(direc + "/*T1gad_wm.nii.gz")
    t2 = glob(direc + "/*T2_wm.nii.gz")
    seg = glob(direc + "/*tumor_seg.nii.gz")
    img_list = [adc, asl, fa, dwi, flair, swi, t1, t1gad, t2, seg]
    if all(img_list):
        # build macro for imageJ
        macro = []
        for item in img_list: # open images
            macro.append('open("' + item[0] + '");')
        macro.append('run("Tile");')
        macro.append('run("Synchronize Windows");')

        # write macro
        macro_file = os.path.join(data_dir, 'qc_macro.ijm')
        with open(macro_file, 'w') as f:
            for item in macro:
                f.write("%s\n" % item)

        # Run macro
        cmd = '/home/ecalabr/ImageJ/jre/bin/java -Xmx5000m -jar /home/ecalabr/ImageJ/ij.jar -ijpath /home/ecalabr/ImageJ -macro ' + macro_file
        os.system(cmd)
    print("Done with study " + os.path.basename(direc) + ": " + str(i+skip) + " of " + str(n_total))