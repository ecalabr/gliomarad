""" script for renaming data directories to a different alternate ID using an ID csv file """

import os
import csv
import argparse


# define functions
# walk through directories and change names
def change_dir_id(data_dir, folders, current_id_list, desired_id_list):

    # initialize outputs
    out_files = []

    # loop through folders
    for folder in folders:

        # match current ID to current id list
        current_id = folder
        id_ind = [ind for ind, x in enumerate(current_id_list) if int(current_id) == int(x)]
        if len(id_ind) > 1:
            raise ValueError("More than one ID match in ID list for current ID " + current_id)
        else:
            try:  # if no index is found this will error
                id_ind = id_ind[0]
            except:
                pass

        # only do something if ID index is found
        if id_ind:
            # move folder
            current_folder_path = os.path.join(data_dir, folder)
            new_folder_path = os.path.join(data_dir, str(desired_id_list[id_ind]).zfill(8))
            try:
                os.rename(current_folder_path, new_folder_path)
            except:
                pass

            # change names of all files in new folder
            for f in [os.path.join(new_folder_path, d) for d in os.listdir(new_folder_path)]:
                if current_id in f:
                    newname = f.split(current_id)[0] + str(desired_id_list[id_ind]).zfill(8) + f.split(current_id)[1]
                    os.rename(f, newname)
                    out_files.append(newname)


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Path to data directory")
    parser.add_argument('--id_csv', default=None,
                        help="Path to ID CSV")
    parser.add_argument('--anonymize', default="True",
                        help="True if anonymizing, False if deanonymizing")

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert args.data_dir, "No data directory specified. Use --data_dir"
    assert os.path.isdir(args.data_dir), "No data directory found at {}".format(args.data_dir)
    assert args.id_csv, "No patient CSV specified. Use --id_csv="
    assert os.path.isfile(args.id_csv), "No CSV file found at {}".format(args.id_csv)
    assert args.anonymize in ['True', 'False'], "Anonymize option must be either 'True' or 'False'"

    # load csv and get accessions and IDs
    access_list = []
    id_list = []
    with open(args.id_csv, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        id_csv = list(reader)
    for line in id_csv[1:]:
        access_list.append(int(line[2]))
        id_list.append(int(line[0]))

    # handle anonymization direction
    if args.anonymize == 'True':
        cur_id_list = access_list
        des_id_list = id_list
    elif args.anonymize == 'False':
        des_id_list = access_list
        cur_id_list = id_list
    else:
        raise ValueError("Anonymize option must be either 'True' or 'False'")

    # get list of directories in data dir
    dirs = [direc for direc in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, direc))]

    # do work
    output_files = change_dir_id(args.data_dir, dirs, cur_id_list, des_id_list)
