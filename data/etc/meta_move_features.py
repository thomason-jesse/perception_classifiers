#!/usr/bin/env python
__author__ = 'jesse'

import sys
import os


# python meta_move_features.py [data_dir] [modalities] [behaviors] [dest_dir]
def main():

    data_dir = sys.argv[1]
    modalities = sys.argv[2].split(',')
    behaviors = sys.argv[3].split(',')
    dest = sys.argv[4]

    # loop through modality directories
    for m in modalities:
        # loop through behavior directories
        for b in behaviors:
            csv_dir = os.path.join(data_dir, m, b+"_"+m)
            for root, dirs, files in os.walk(csv_dir):
                csv_files = [f for f in files if len(f.split('.')) > 1 and f.split('.')[1] == 'csv']
                if len(files) > 1:
                    sys.exit("ERROR: Expected single CSV in "+csv_dir+" but found "+str(len(csv_files)))
                cmd = " ".join(["python", "move_features.py", os.path.join(root, csv_files[0]),
                          os.path.join(data_dir, 'etc', 'object_list.csv'), b, m, dest])
                print cmd
                os.system(cmd)

if __name__ == "__main__":
        main()
