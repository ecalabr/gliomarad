"""Moves data downloaded from UCSF air into new unique directories named after the accession number"""

import pydicom as dicom
import os
import subprocess
import csv
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="/media/ecalabr/data1/lgg_data/access",
                    help="Path to data directory")
parser.add_argument('--pt_csv', default="/home/ecalabr/Dropbox/idh1_gbm_project/gbm_spreadsheets/master_preop_glioma_9-20-15--8-31-19.csv",
                    help="Path to csv with accession data")
parser.add_argument('--mrn_col', default=1,
                    help="Index of MRN column in patient CSV file")
parser.add_argument('--acc_col', default=2,
                    help="Index of accession number column in patient CSV file")

if __name__ == '__main__':
    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    patient_csv = args.pt_csv
    assert os.path.isfile(patient_csv), "Patient CSV not found at {}".format(patient_csv)

    # get patient list
    mrn_list = []
    access_list = []
    with open(patient_csv, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        patient_list = list(reader)
    for line in patient_list[1:]:
        mrn_list.append(int(line[args.mrn_col]))
        access_list.append(int(line[args.acc_col]))

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