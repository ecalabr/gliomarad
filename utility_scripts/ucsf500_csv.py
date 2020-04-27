""" separates UCSF500 result text copied from PDFs into specific CSV columns based on a list of gene keywords"""

import os
import csv
import argparse


# define functions
def ucsf500_csv(csv_path, outfile):
    # define genes
    genes = {'EGFR': ['EGFR'], 'PTEN': ['PTEN'], 'TERT': ['TERT'],
             'ATRX': ['ATRX'], 'IDH': ['IDH1', 'IDH2'], 'TP53': ['TP53'], 'CDKN2': ['CDKN2'],
             '7/10 Aneuploidy': ['Trisomy', 'Monosomy', 'Polysomy'], '1p/19q': ['1p', '19q']}

    # load csv with ucsf 500s
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        ucsf500 = list(reader)

    # loop through genes in alphabeticle order, combining results when appropriate
    newsheet = [[item for item in sorted(genes.keys())]]
    for line in ucsf500:
        if not line:
            newline = [""] * len(genes)
        else:
            newline = ["negative"] * len(genes)
            if line:
                for item in line[0].split('; '):
                    gene = item.split(' ')[0]
                    for key in genes.keys():
                        if any([gene.startswith(x) for x in genes[key]]):
                            index = list(genes.keys()).index(key)
                            if not newline[index] == 'negative':
                                newline[index] = newline[index] + '; ' + item
                            else:
                                newline[index] = item
        newsheet.append(newline)

    # write output
    with open(outfile, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        writer.writerows(newsheet)

    return outfile


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None,
                        help="Path to UCSF500 csv")
    parser.add_argument('--out', default="ucsf500",
                        help="Output prefix")

    # parse arguments and make sure csv exists
    args = parser.parse_args()
    assert csv, "Must specify CSV using --csv"
    assert os.path.isfile(args.csv), "No csv found at {}".format(args.data_dir)

    # generate output name
    basename = os.path.basename(args.csv).split('.csv')[0]
    my_outfile = os.path.join(os.path.dirname(args.csv), basename + '_' + args.out + '.csv')

    # do work
    csv_outfile = ucsf500_csv(args.csv, my_outfile)
