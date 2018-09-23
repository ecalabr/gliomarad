import os
from glob import glob

direc = "/Users/edc15/Desktop/kp_test_data/outfiles"
macro = "/Users/edc15/Desktop/kp_test_data/outfiles/macro.txt"

niis = glob(direc + "/*.nii.gz")

for ind, nii in enumerate(niis, 0):
    idname = os.path.basename(nii).split("_")[0]
    niis[ind] = idname

unqids = list(set(niis))

dwi_list = list()
for ids in unqids:
    flair = os.path.join(direc, ids + "_flair_warped_masked.nii.gz")
    dwi = os.path.join(direc, ids + "_dwi_warped_masked.nii.gz")
    adc = os.path.join(direc, ids + "_adc_warped_masked.nii.gz")
    if os.path.isfile(flair) and os.path.isfile(dwi) and os.path.isfile(adc):
        dwi_list.append(ids)

scpt = list()
scpt.append("basedir = \"/Users/edc15/Desktop/kp_test_data/outfiles/\";")
scpt.append("idlist = newArray" + str(tuple(dwi_list)) + ";")
scpt.append("for (i = 0; i < lengthOf(idlist); i++) {")
scpt.append("open(basedir+idlist[i]+\"_dwi_warped_masked.nii.gz\");")
scpt.append("run(\"In [+]\");")
scpt.append("run(\"In [+]\");")
scpt.append("run(\"In [+]\");")
scpt.append("doCommand(\"Start Animation [\\\\]\");")
scpt.append("Dialog.create(\"Stroke?\");")
scpt.append("Dialog.addCheckbox(\"Stroke?\", false);")
scpt.append("Dialog.show;")
scpt.append("bool = Dialog.getCheckbox();")
scpt.append("if (bool) {")
scpt.append("File.append(idlist[i], basedir+\"log.txt\");")
scpt.append("};")
scpt.append("run(\"Close All\");")
scpt.append("};")
with open(macro, "w") as macrofile:
    for line in scpt:
        macrofile.write("%s\n" % line)
