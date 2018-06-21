import os

# main directory after unzipping
main_dir = '/media/ecalabr/data/idh_wt_gbm'
# accessions list used for downloading
acces_list = '/media/ecalabr/data/idh_wt_gbm/accessions.txt'

# get list of subdirs
subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
subdirs = sorted(subdirs, key=lambda x: (int(x.split('.')[7]), int(x.split('.')[8])))  # sorts on ucsf air date names
# for d in subdirs:
#    print(d)

# get list of accessions
with open(acces_list, 'r') as f:
    acces = f.read().split('\n')  # '\r\n' for mac
    acces = acces[:-1]  # seems not needed for mac
#    for a in acces:
#        print(acces)

# loop for directories
if len(acces) != len(subdirs):
    print('subdirs = ' + str(len(subdirs)))
    print('accessions = ' + str(len(acces)))
    exit('accessions and subdirs number not equal')
else:
    for i, direc in enumerate(subdirs, 0):
        zipname = os.path.join(main_dir, acces[i] + '.zip')
        cmd = 'cd ' + main_dir + '; zip -qr ' + zipname + ' ' + os.path.basename(direc)
        print(cmd)
        os.system(cmd)
