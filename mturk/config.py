signature = "umtn2ZUWgNy50Gxmq6g+iqNnWuQtnIgEQmq4W7Pa"
accesskey = "1M30EJTD3F4Q4XN3TKR2"
sandbox   = True
localhost = "http://reason.csail.mit.edu:8080/"
database  = "mysql://root@localhost/ihog"
geolocation = ""

# probably no need to mess below this line

import multiprocessing
processes = multiprocessing.cpu_count()

import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
