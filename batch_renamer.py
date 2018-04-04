import os

# main directory after unzipping
main_dir = '/Users/edc15/Desktop/idh_mutant_gbm/idh1_mutant_gbm'
# accessions list used for downloading
acces_list = '/Users/edc15/Desktop/idh_mutant_gbm/idh1_mutant_gbm.txt'

# get list of subdirs
subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

# get list of accessions
with open(acces_list, 'r') as f:
    acces = f.read().split('\r\n')

# loop for directories
if len(acces) != len(subdirs):
    exit('accessions and subdirs number not equal')
else:
    for i, direc in enumerate(subdirs, 0):
        zipname = os.path.join(main_dir, acces[i] + '.zip')
        cmd = 'cd ' + main_dir + '; zip -qr ' + zipname + ' ' + os.path.basename(direc)
        print(cmd)
        os.system(cmd)