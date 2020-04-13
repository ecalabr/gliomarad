"""Takes a mpower format csv and a air format csv and combines the relevant fields to make the same format used
in the master preop gbm spreadsheet"""

import csv
import os
import argparse

########################## define functions ##########################
def mpower_csv_convert(mpower_csv, air_csv):
    # define lists
    mrn = []
    acces = []
    first = []
    last = []
    sex = []
    dob = []
    scantime = []
    nimage = []

    # load mpower csv and get relevant items
    with open(mpower_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        patient_list = list(reader)
    for line in patient_list[1:]:
        mrn.append(int(line[13]))
        acces.append(int(line[3]))
        first.append(line[14])
        last.append(line[15])

    # load air csv and get relevant items for each accession
    with open(air_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        patient_list = list(reader)
        # if the first row is text header then just zero it out
        if isinstance(patient_list[0][0], str):
            patient_list[0] = [0] * len(patient_list[0])
    for item in acces:
        for line in patient_list:
            if item == int(line[4]):
                sex.append(line[2])
                dob.append(line[3])
                scantime.append(line[5])
                nimage.append(line[7])

    # make new csv
    lines = [['Patient MRN', 'Accession', 'First Name', 'Last Name', 'Sex', 'DOB', 'Scan Date/Time', 'Images', 'Path',
              'Path 2ndary', 'Ki-67/MIB-1', 'MGMT', 'MGMT index', '1p/19q', 'IDH', 'ATRX', 'TP53',  'PTEN', 'EGFR',
              '7/10 Aneuploidy', 'CDKN2', 'TERT', 'UCSF 500', 'ASL', 'QC issues', 'Status']]
    for i in range(len(mrn)):
        line = [mrn[i], acces[i], first[i], last[i], sex[i], dob[i], scantime[i], nimage[i]]
        lines.append(line)

    # save output
    outname = os.path.join(os.path.dirname(mpower_csv), 'air_mpower_combined.csv')
    with open(outname, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        writer.writerows(lines)

    return outname

########################## executed  as script ##########################
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpower_csv', help="Full path to mpower format csv")
    parser.add_argument('--air_csv', help="Full path to ucsf air format csv")

    # parse args
    args = parser.parse_args()
    assert args.mpower_csv, "Must specify mPower CSV using --mpower_csv"
    assert os.path.isfile(args.mpower_csv), "No mPower CSV found at {}".format(args.mpower_csv)
    assert args.air_csv, "Must specify AIR CSV using --mpower_csv"
    assert os.path.isfile(args.air_csv), "No AIR CSV found at {}".format(args.air_csv)

    # do work
    out_csv = mpower_csv_convert(args.mpower_csv, args.air_csv)

    # print
    print("Combined CSV created at : {}".format(out_csv))
