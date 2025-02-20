STOW_VERSION='1.mwe' ###MINIMAL WORKING EXAMPLE
###import sys, getopt, traceback, imp ###ORIGINAL SCRIPT IMPORTS
import sys, getopt ###MWE SCRIPT IMPORTS
#N.B.: argparse instead of optparse somehow recommended.
#argparse allows you to write less code :http://stackoverflow.com/a/29006699
#argparse docu for mutually exclusive options:
#https://docs.python.org/3.4/library/argparse.html?highlight=argparse#mutual-exclusion
#fileinput somehow recommended too: https://docs.python.org/2/library/fileinput.html

#RS added field4 for explanation
#RS warning: short options are single characters!!! the second character will be taken as the argument

def main(argv, option):
    """ Input method to read shell positional arguments
    
    :Parameters:
        - argv      (list): list of command line arguments passed to a Python script. btw: Python docu suggests fileinput instead of sys.argv. https://docs.python.org/2/library/sys.html
        - option    (arr) : array of array of options. Each row is an option-specific entry. The columns are: default value, short option, long option (consecutively).
        
    :Returns:
        - field     (arr) : array of parameters extracted according option (row-major order).
    """
    if len(option[0]) > 3:
        helpmessage = 'python '+sys.argv[0]+''.join(['\n -'+field[1]+' <'+field[2]+'> [default: '+str(field[0])+']'+field[3] for field in option])
    else:
        helpmessage = 'python '+sys.argv[0]+''.join(['\n -'+field[1]+' <'+field[2]+'> [default: '+str(field[0])+']' for field in option])        
    shortoptions = 'h'+''.join([field[1]+':' for field in option]) # as str
    longoptions = [field[2]+'=' for field in option] # as arr
    try:
          opts, args = getopt.getopt(argv, shortoptions, longoptions) # Similar to GNU version of getopt: options do not have to appear before all the operands
    except getopt.GetoptError:
          print('*** INPUT ERROR:\n'+helpmessage)
          sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('*** INPUT HELP :\n'+helpmessage)
            sys.exit()
        for field in option:
            if opt in ("-"+field[1], "--"+field[2]):
                field[0] = arg
    return [field[0] for field in option] # as arr

