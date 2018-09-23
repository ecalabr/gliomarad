import external_software.brats17_master.test_ecalabr2 as test_ecalabr2
from glob import glob

# define dir_list
base_dir = "/media/ecalabr/data/gbm/"
dir_list = glob(base_dir + "/*/")
#dir_list = ["/media/ecalabr/data/gbm/11418822/", "/media/ecalabr/data/gbm/11446564/"]

test_ecalabr2.test(dir_list)