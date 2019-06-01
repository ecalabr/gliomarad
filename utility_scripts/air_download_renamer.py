"""Moves data downloaded from UCSF air into new unique directories named after the accession number"""

import pydicom as dicom
import os
import subprocess
import csv

data_dir = "/media/ecalabr/scratch/work/download_22"
patient_csv = "/home/ecalabr/Dropbox/idh1_gbm_project/download_result_spreadsheets/feb11_april25_2019_air_mpower_combined.csv"

# get patient list
mrn_list = []
access_list = []
with open(patient_csv, "rb") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    patient_list = list(reader)
for line in patient_list[1:]:
    mrn_list.append(int(line[0]))
    access_list.append(int(line[1]))

# walk through each directory and find the matching patient
folders = [direc for direc in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, direc))]

# function to find any dicom in a directory
def dicom_search(directory):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".dcm"):
                return os.path.join(root, filename)

# match patient function
def patient_match(mr_list, ac_list, medrecn, accession):
    def find(lst, itm):
        return [ind for ind, x in enumerate(lst) if itm == x]
    mr_ind = find(mr_list,medrecn)
    if len(mr_ind) != 1:
        print("Found " + str(len(mr_ind)) + " matching MRNs for " + str(medrecn) + " accession: " + str(accession))
        return False
    ac_ind = find(ac_list,accession)
    if len(ac_ind) != 1:
        print("Found " + str(len(ac_ind)) + " matching accessions for " + str(accession) + " MRN: " + str(medrecn))
        return False
    if not ac_ind[0] == mr_ind[0]:
        print(" Accession and MRN indices don't match!")
        return False
    return True

# match and move each folder into a new folder named by accession number
for i in folders:
    fullpath = os.path.join(data_dir, i)
    dcm = dicom_search(fullpath)
    hdr=dicom.read_file(dcm)
    access = int(hdr.AccessionNumber)
    mrn = int(hdr.PatientID)
    # if MRN/accession don't match add a _ to directory beginning
    if patient_match(mrn_list, access_list, mrn, access):
        outdir = os.path.join(data_dir, str(access).zfill(8))
    else:
        outdir = os.path.join(data_dir, "_" + str(access).zfill(8))
    print("mkdir " + outdir)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    cmd = "mv " + fullpath + " " + outdir + "/."
    print(cmd)
    subprocess.call(cmd, shell=True)