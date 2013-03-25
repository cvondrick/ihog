signature = ""
accesskey = ""
sandbox   = True
localhost = "http://localhost:8080/"
database  = "mysql://root@localhost/ihog"
geolocation = ""

# probably no need to mess below this line

import multiprocessing
processes = multiprocessing.cpu_count()

import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
