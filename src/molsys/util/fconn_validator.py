# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:17:19 2017

@author: rochus
"""

import csv
import string

class fconn_validator:
    """
    class to validate fragment connectivities

    depending on the mode either use a local csv file or talk to MOF+
    via the API using json to answer requests on the vlaidity of a bond
    connecting two fragments
    """

    def __init__(self, mode="csv", csv_file="allowed_frags.csv"):
        """
        TBI: a mode "api" to request info from server
        """
        self.frag_pairs = {}
        self.mode = mode
        if mode == "api":
            raise ValueError("Not implemented yet...")
        elif mode == "csv":
            csvf = open(csv_file, "r")
            csvreader = csv.reader(csvf, delimiter=",")
            for line in csvreader:
                if len(line)> 0:
                    pair = [line[0],line[1]]
                    pair = map(string.strip,pair)
                    pair.sort()
                    spair = string.join(pair, ":")
                    if not spair in self.frag_pairs.keys():
                        self.frag_pairs[spair] = line[2].strip()
                    else:
                        raise ValueError("A pair appears twice in the csv file")
            csvf.close()
        return

    def report(self):
        print(self.frag_pairs)
        return

    def __call__(self, fbond):
        if fbond in self.frag_pairs.keys():
            return self.frag_pairs[fbond]
        else:
            return False
