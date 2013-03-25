import os.path, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from turkic.server import handler, application
from turkic.database import session
from models import *

@handler()
def getjob(id):
    job = session.query(Job).get(id)
    windows = []
    for interconnect in job.interconnect:
        windows.append(interconnect.window.id)
    return {"windows": windows}

@handler("image/jpeg", False)
def getwindow(id):
    window = session.query(DetectionWindow).get(id)
    f = open(window.filepath)
    return str(f.read())
