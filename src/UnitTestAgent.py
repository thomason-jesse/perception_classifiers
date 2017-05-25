#!/usr/bin/env python
__author__ = 'jesse'

import sys


class UnitTestAgent:

    def __init__(self, io, tidx, table_oidxs):
        assert 0 <= tidx <= 2
        assert len(table_oidxs) == 3

        self.io = io
        self.table_oidxs = table_oidxs
        self.tidx = tidx

    def face_table(self, tidx):
        assert 0 <= tidx <= 2

        if self.io.face_table(tidx):
            self.tidx = tidx
        else:
            sys.exit("ERROR: failed to face table " + str(tidx))
