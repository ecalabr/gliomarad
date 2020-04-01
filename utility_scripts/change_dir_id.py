""" script for renaming data directories to a different alternate ID """

import os
import csv
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/media/ecalabr/scratch/qc_complete',
                    help="Path to data directory")
parser.add_argument('--id_csv', default='/home/ecalabr/Dropbox/idh1_gbm_project/gbm_spreadsheets/master_preop_gbm_9-20-15--8-31-19.csv',
                    help="Path to ID CSV")
parser.add_argument('--anonymize', default="True",
                    help="True if anonymizing, False if deanonymizing")

if __name__ == '__main__':

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "No data directory found at {}".format(args.data_dir)
    assert os.path.isfile(args.id_csv), "No CSV file found at {}".format(args.id_csv)
    assert args.anonymize in ['True', 'False'], "Anonymize option must be either 'True' or 'False'"

    # load csv and get accessions and IDs
    access_list = []
    id_list = []
    with open(args.id_csv, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        id_csv = list(reader)
    for line in id_csv[1:]:
        access_list.append(int(line[2]))
        id_list.append(int(line[0]))

    # handle anonymization direction
    if args.anonymize == 'True':
        current_id_list = access_list
        desired_id_list = id_list
    elif args.anonymize == 'False':
        desired_id_list = access_list
        current_id_list = id_list
    else:
        raise ValueError("Anonymize option must be either 'True' or 'False'")

    # get list of directories in data dir
    folders = [direc for direc in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, direc))]

    # walk through directories and change names
    for folder in folders:

        # match current ID to current id list
        current_id = folder
        id_ind = [ind for ind, x in enumerate(current_id_list) if int(current_id) == int(x)]
        if len(id_ind) > 1:
            raise ValueError("More than one match in ID CSV for current id " + current_id)
        else:
            try: # if no index is found this will error
                id_ind = id_ind[0]
            except:
                pass

        # only do something if ID index is found
        if id_ind:
            # move folder
            current_folder_path = os.path.join(args.data_dir, folder)
            new_folder_path = os.path.join(args.data_dir, str(desired_id_list[id_ind]).zfill(8))
            try:
                os.rename(current_folder_path, new_folder_path)
            except:
                pass

            # change names of all files in new folder
            for f in [os.path.join(new_folder_path, direc) for direc in os.listdir(new_folder_path)]:
                if current_id in f:
                    newname = f.split(current_id)[0] + str(desired_id_list[id_ind]).zfill(8) + f.split(current_id)[1]
                    os.rename(f, newname)
