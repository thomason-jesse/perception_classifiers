#!/usr/bin/env python
__author__ = 'jesse'

import sys
import os
import time


# python grab_pictures.py [object_IDs_fn]
def main():

    obj_id_fn = sys.argv[1]

    # get mapping from object names to IDs
    oid_to_oname = {}
    f = open(obj_id_fn, 'r')
    for line in f:
        if len(line.strip()) == 0:
            continue
        oname, oid = line.strip().split(',')
        oid_to_oname[int(oid)] = oname
    f.close()

    loc = "/home/users/max/ordinal_dataset/"
    dest = "~/Downloads/pics/"

    # gather images for each object
    pics = []
    for oidx in range(1, 33):
        # get a representative image for each trial
        for trial in range(1, 6):
            p = os.path.join(loc, "obj_"+str(oidx), "trial_"+str(trial), "look", "vision_data", "test0*.jpg")
            pics.append(p)

    scp_cmd = "scp -v -P 40 jthomason@nixons-head.csres.utexas.edu:\""
    scp_cmd += " ".join(pics)
    scp_cmd += "\" " + dest + " > tmp.txt 2>&1"

    print scp_cmd
    os.system("touch tmp.txt")
    os.system("chmod a+rwx tmp.txt")
    os.system(scp_cmd)

    f = open("tmp.txt", 'r')
    for oidx in range(1, 33):
        for trial in range(1, 6):
            pon = None
            while pon is None:
                line = f.readline()
                if line[:5] == "Sink:":
                    pon = line.split(" ")[3].strip()
            pnn = oid_to_oname[oidx]+"_"+str(trial)+".JPG"
            mv_cmd = "mv "+os.path.join(dest, pon)+" "+os.path.join(dest, pnn)
            print mv_cmd
            os.system(mv_cmd)
    f.close()

if __name__ == "__main__":
        main()
