import os
from glob import glob

# main directory containing subdirs with processed data
data_dir = "/media/ecalabr/data1/new_gbm_download"

# list all subdirs with the processed data
direcs = [item for item in glob(data_dir + "/*") if os.path.isdir(item)]
direcs = sorted(direcs, key=lambda x: int(os.path.basename(x)))

# set skip for already edited masks here
skip = 0
n_total = len(direcs)
direcs = direcs[skip:]
if isinstance(direcs, str):
    direcs = [direcs]

# manual direcs here
#direcs = ['/media/ecalabr/data/gbm/11936344']

# run ITK-snap on each one
for i, direc in enumerate(direcs, 1):
    t1gad = glob(direc + "/*T1gad_w.nii.gz")
    mask = glob(direc + "/*combined_brain_mask.nii.gz")
    #mask = glob(direc + "/*tumor_seg.nii.gz")
    if t1gad and mask:
        t1gad = t1gad[0]
        mask = mask[0]
        cmd = "itksnap -g " + t1gad + " -s " + mask
        os.system(cmd)
        print("Done with study " + os.path.basename(direc) + ": " + str(i+skip) + " of " + str(n_total))
