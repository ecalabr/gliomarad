""" Moves data downloaded from UCSF air into new unique directories named after the accession number using csv """

import pydicom as dicom
import os
import subprocess
import csv
import argparse


# define functions
# function to find any dicom in a directory
def dicom_search(directory):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # if extension is dcm or there is no extension then test if it is a dicom
            if filename.endswith('.dcm') or len(filename.split('.')) == 1:
                try:
                    hdr = dicom.read_file(os.path.join(root, filename))
                    _ = hdr.AccessionNumber
                    _ = hdr.PatientID
                    return os.path.join(root, filename)
                except:
                    pass
    print("No dicom file found in directory {}".format(directory))
    return None


# match an accession number and MRN from lists
def patient_match(mr_list, ac_list, medrecn, accession):
    def find(lst, itm):
        return [ind for ind, x in enumerate(lst) if itm == x]
    mr_ind = find(mr_list, medrecn)
    if len(mr_ind) != 1:
        print("Found " + str(len(mr_ind)) + " matching MRNs for " + str(medrecn) + " accession: " + str(accession))
        return False
    ac_ind = find(ac_list, accession)
    if len(ac_ind) != 1:
        print("Found " + str(len(ac_ind)) + " matching accessions for " + str(accession) + " MRN: " + str(medrecn))
        return False
    if not ac_ind[0] == mr_ind[0]:
        print(" Accession and MRN indices don't match!")
        return True
    return True


# match and move each folder into a new folder named by accession number
def rename_dcm_dir(folders, acc_list, mr_list):
    outdirs = []
    for i in folders:
        fullpath = os.path.join(data_dir, i)
        dcm = dicom_search(fullpath)
        hdr = dicom.read_file(dcm)
        access = int(hdr.AccessionNumber)
        mrn = int(hdr.PatientID)
        print("Accession is {}".format(access))
        print("MRN is {}".format(mrn))
        # if MRN/accession don't match add a _ to directory beginning
        if patient_match(mr_list, acc_list, mrn, access):
            outdir = os.path.join(data_dir, str(access).zfill(8))
        else:
            outdir = os.path.join(data_dir, "_" + str(access).zfill(8))
        if not os.path.isdir(outdir):
            print("mkdir " + outdir)
            os.mkdir(outdir)
        if not outdir == fullpath:
            cmd = "mv " + fullpath + " " + outdir + "/."
            # print(cmd)
            subprocess.call(cmd, shell=True)
        outdirs.append(outdir)
    return outdirs


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--id_csv', default=None,
                        help="Path to csv with accession data")
    parser.add_argument('--mrn_col', default=1,
                        help="Index of MRN column in patient CSV file")
    parser.add_argument('--acc_col', default=2,
                        help="Index of accession number column in patient CSV file")

    # get arguments and check them
    args = parser.parse_args()
    data_dir = args.data_dir
    assert data_dir, "No data directory specified. Use --data_dir"
    assert os.path.isdir(data_dir), "Data directory not found at {}".format(data_dir)
    id_csv = args.id_csv
    assert id_csv, "No patient CSV specified. Use --id_csv"
    assert os.path.isfile(id_csv), "ID CSV not found at {}".format(id_csv)

    # get patient list
    mrn_list = []
    access_list = []
    with open(id_csv, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        patient_list = list(reader)
    for line in patient_list[1:]:
        try:
            mrn_list.append(int(line[int(args.mrn_col)]))
        except:
            pass
        try:
            access_list.append(int(line[int(args.acc_col)]))
        except:
            pass

    # walk through each directory and find the matching patient
    dirs = [direc for direc in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, direc))]

    # do work
    output_directories = rename_dcm_dir(dirs, access_list, mrn_list)
