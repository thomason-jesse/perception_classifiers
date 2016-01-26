#!/usr/bin/env python
__author__ = 'jesse'

import sys
import os


# python move_features.py [original features csv] [objects list csv] [behavior] [modality] [dest dir]
def main():

    feat_fn = sys.argv[1]
    obj_id_fn = sys.argv[2]
    beh = sys.argv[3]
    mod = sys.argv[4]
    dest = sys.argv[5]

    # get mapping from object names to IDs
    oname_to_oid = {}
    oname_to_obs = {}
    f = open(obj_id_fn, 'r')
    for line in f:
        if len(line.strip()) == 0:
            continue
        oname, oid = line.strip().split(',')
        oname_to_oid[oname] = int(oid)
        oname_to_obs[oname] = []
    f.close()

    # read in features
    f = open(feat_fn, 'r')
    for line in f:
        oname_obs = line.strip().split(',')[0].split('_')
        oname_p = []
        for p in oname_obs:
            if '.' in p:
                p = ".".join(p.split('.')[:-1])
            try:
                _ = int(p)
            except ValueError:
                if len(p) > 0:
                    p_wos = p.split('/')[-1]
                    oname_p.append(p_wos)
        oname = '_'.join(oname_p)
        oname_to_obs[oname].append(line)
    f.close()

    # write out object directories
    for oname in oname_to_oid:
        oid = oname_to_oid[oname]
        loc_dir = os.path.join(dest, 'obj'+str(oid), beh, mod)
        if not os.path.isdir(loc_dir):
            os.system("mkdir -p "+loc_dir)
        fn = os.path.join(loc_dir, 'features.csv')
        f = open(fn, 'w')
        for obs in oname_to_obs[oname]:
            f.write(obs)
        f.close()


if __name__ == "__main__":
        main()
