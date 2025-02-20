# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os.path
import sys
import importlib
import traceback

"""
HOW TO IMPLEMENT A NEW ADDON
Given addonname as addon's name (a string), add:
- "addonname" (as a string) to the list __all__ (do not forget the comma!);
- addonname.py module in __file__ (e.g. this) directory;
- a class in addonname.py named addonname, e.g. the same name as the module
    to which the class belongs;
For a standard addon importing behaviour, the interface takes care of the rest.
For a custom addon import, see further.

HOW TO CUSTOMIZE ADDON IMPORT
Given addonname as addon name
- define an importing function (e.g. _importfunc_addonname);
- AFTER THAT, set _importfunc[addonname] = _importfunc_addonname.
If _importfunc[addonname] is None: standard addon import is used.
USE CUSTOM ADDON IMPORT FOR
- a different module addonname2.py
- a different directory that contains addonname.py
- a class in addonname.py that is NOT named addnoname
    - N.B. must be named addonname in locals() (i.e. here) for instance:
        from .addonname import classname as addonname
- in general, a particular importing behaviour (see e.g. zmat)
N.B. BY DESIGN: if addonname in __all__, locals()[addonname] can be only:
- None (addon is known but not imported)
- a class named addonname
    - if standard: must be named addonname in the addonname module too!
"""

### importing order is detrimental for addon inter-dependencies
### there is a comment for dependency/ies (feel free to improve!)
### TBI: addon dependency
__all__ = [
    "base", ### N.B: many derive from base
    "graph",
    "fragments", ### graph
    "ff", ### fragments, base
    "ric", ### ff, base
    "acab", ### base
    "spg", ### base
    "ptg", ### base
    "bb",
    "molecules",
    "groups",
    "zmat",
    "wannier",
    "topo",
    "obabel",
    "traj",
    "charge",
    "jaxff"
]

_importfunc = {k:None for k in __all__}
_errortrace = {k:None for k in __all__}

def _importfunc_zmat():
    """custom import for zmat prevents annoying warning message in pandas"""
    try:
        import os
        import sys
        f = open(os.devnull, 'w') #cross-platform and version
        sys.stderr = f
        import pandas
        import chemcoord
    except ImportError:
        zmat = None
    else:
        from .zmat import zmat 
        globals()["zmat"] = zmat ### N.B.: implicit locals()["zmat"] = zmat.zmat
    finally:
        sys.stderr = sys.__stderr__
_importfunc["zmat"] = _importfunc_zmat

### cf. https://stackoverflow.com/a/38804371
### reason: importlib 2.7 cannot inherit __future__.absolute import
###     e.g. importlib.import_module("addonname", ".")
_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dirname) ### deleted after the loop
for _addon in __all__:
    try:
        if _importfunc[_addon]: ### custom addon import
            _importfunc[_addon]()
        else: ### standard addon import
            ### simulates: from .addon import addon (python2.7 and python3)
            _module = importlib.import_module(_addon, _dirname)
            globals()[_addon] = getattr(_module, _addon) ### its class
    except Exception as e: ### stores errors raised by _addon.py module
        _errortrace[_addon] = traceback.format_exc()
        globals()[_addon] = None ### means import failed

for _addon in __all__: 
    try:
        sys.modules['__'+_addon] = sys.modules[_addon] # to keep addons working
        del sys.modules[_addon] # prevent import addons in any directory
    except KeyError as e:
        pass

del sys.path[0] # prevent import modules of that directory

