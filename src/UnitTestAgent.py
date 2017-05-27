#!/usr/bin/env python
__author__ = 'jesse'

import sys


class UnitTestAgent:

    def __init__(self, io, tid, table_oidxs):
        assert 1 <= tid <= 3
        assert len(table_oidxs) == 3

        self.io = io
        self.table_oidxs = table_oidxs
        self.tid = tid

    def face_table(self, tid):
        assert 1 <= tid <= 3

        if self.io.face_table(tid, self.table_oidxs[tid - 1]):
            self.tid = tid
        else:
            sys.exit("ERROR: failed to face table " + str(tid))
