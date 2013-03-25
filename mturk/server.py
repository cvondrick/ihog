import os.path, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from turkic.server import handler, application

@handler()
def helloworld(name):
    return {"response": "Hello, {0}!".format(name)}
