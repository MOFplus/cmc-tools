# -*- coding: utf-8 -*-
import os
import subprocess

from .user import user_api
from .ff import FF_api
from .admin import admin_api

from .napi import api

def git_revision_hash():
    work_dir = os.getcwd()
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(src_dir)
        rev_no = len(subprocess.check_output(['git', 'rev-list', 'HEAD'], universal_newlines=True).strip().split("\n"))
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True).strip()
    except:
        # catchall in case the code is downloaded via a zip file
        rev_no = 0.0
        commit = "abcdefghijklmnop"
    finally:
        os.chdir(work_dir)
    return (rev_no, commit)

rev_no, commit = git_revision_hash()
__version_info__ = (0, 0, rev_no, "%s"%commit)
__version__ = "%i.%i.%i.%s"%__version_info__

__all__ = ["user_api", "FF_api", "admin_api", "api"]
