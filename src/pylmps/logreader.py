import os
import numpy
import scipy
import scipy.io
import pandas
import sys
if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO 
import matplotlib.pyplot as plt





class logreader(object):
    def __init__(self):
        self.logs = []



class log(object):
    def __init__(self,fname=None):
        self.fname = fname
        if fname is not None:
            self.read_log()
        else:
            self.data = None

    def read_log(self,fname=None):
        if fname is not None:
            self.fname = fname
        assert self.fname is not None
        f = open(self.fname,'r')
        self.logtext = self.format_logtxt_main(f.read())
        self.data = pandas.read_table(StringIO(self.logtext),delim_whitespace=True)
        f.close()
        return

    def format_logtxt_main(self,txt):
        newtext = ''
        for xx in txt.split('\n'):
            x = xx.split()
            if xx.count ('Volume Cella Cellb Cellc CellAlpha CellBeta CellGamma ') != 0: newtext += xx+'\n'
            if len(x) < 10: continue
            try:
                float(x[0]) # sry for that shitty code
                float(x[1])
                float(x[2])
                float(x[3])
                float(x[4])
                float(x[5])
            except:
                continue
            newtext += xx+'\n'
        return newtext

    def text_to_data(self,method='all'):
        if method == 'all':
            # try all possibilities and use the one that works
            text = self._format_text1(self.logtext)
            data = pandas.read_table(StringIO(text),delim_whitespace=True)
            text2 = f.rsplit('Loop ',1)[0].rsplit('Mbytes',1)[-1]
            data2 = pandas.read_table(StringIO(text2),delim_whitespace=True)
            
    def _format_text1(self,text):
        return text.rsplit('WARNING',1)[0].rsplit('Mbytes',1)[-1]
    
    def _format_text1(self,text):
        return text.rsplit('Loop ',1)[0].rsplit('Mbytes',1)[-1]
