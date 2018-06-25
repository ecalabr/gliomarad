import external_software.brats17_master.test_ecalabr2 as test_ecalabr2
from glob import glob

# define dir_list
base_dir = "/media/ecalabr/data/idh_wt_gbm/"
dir_list = glob(base_dir + "/*/")

test_ecalabr2.test(dir_list)