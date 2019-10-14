

You need to assign path to exact file (f) to filescr variable on each loop iteration, but not path to files (files - is a list!)

Try below code

import os
from os import path
import shutil

src = "home/alex/galvanize/capstones/capstone2/MURA-v1.1/train/XR_ELBOW/"
dst = "home/alex/galvanize/capstones/capstone2/MURA-v1.1/train/XR_ELBOW

files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
for f in files:
    shutil.copy(path.join(src, f), dst)

