#!/usr/bin/env python
__author__ = 'jesse'

import sys
import os


# python write_config.py [obj_dir] [dest_fn]
def main():

    obj_dir = sys.argv[1]
    dest = sys.argv[2]

    obj_items = os.listdir(obj_dir)
    obj_b = [item for item in obj_items if os.path.isdir(os.path.join(obj_dir, item))]
    behaviors = obj_b
    modalities = []
    num_features = {}
    for b in obj_b:
        num_features[b] = {}
        b_items = os.listdir(os.path.join(obj_dir, b))
        obj_m = [item for item in b_items if os.path.isdir(os.path.join(obj_dir, b, item))]
        for m in obj_m:
            if m not in modalities:
                modalities.append(m)
            f = open(os.path.join(obj_dir, b, m, 'features.csv'), 'r')
            nf = len(f.readline().split(','))-1
            num_features[b][m] = nf
            f.close()

    print behaviors
    print modalities
    print num_features

    f = open(dest, 'w')
    f.write("," + ",".join(modalities) + "\n")
    lines = []
    for b in behaviors:
        nfs = [str(num_features[b][m]) if m in num_features[b] else '0' for m in modalities]
        lines.append(b+"," + ",".join(nfs))
    f.write("\n".join(lines))
    f.close()

if __name__ == "__main__":
        main()
