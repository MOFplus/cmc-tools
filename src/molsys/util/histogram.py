#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
import sys


class histogram(object):
    def __init__(self, start, stop, bins=200):
        self.nbins = bins
        self.data = [0.0] * bins
        self.start = float(start)
        self.stop = float(stop)
        self.dr = (self.stop - self.start) / float(bins)
        self.total = 0.0

    def add(self, point):
        if not (self.start <= point <= self.stop):
            raise Exception("out of bounds point: {}".format(point))
        bin = int((point - self.start) / self.dr)
        self.data[bin] += 1.0
        self.total += 1

    def normalize(self):
        for i in range(self.nbins):
            self.data[i] /= self.total

    def write(self, path):
        with open(path, "w") as fd:
            fd.write("# value histogram\n")
            for (i, data) in enumerate(self.data):
                fd.write("{} {}\n".format(i * self.dr + self.start, data))
