import config
import argparse
import os, os.path
import random
from itertools import izip_longest
from turkic.cli import handler, importparser, Command, LoadCommand
from turkic.database import session
from models import *

@handler("Imports a set of detection windows")
class load(LoadCommand):
    def setup(self):
        parser = argparse.ArgumentParser(parents = [importparser])
        parser.add_argument("dirpath")
        parser.add_argument("--trials", type=int, default = 10)
        parser.add_argument("--pertask", type=int, default = 100)
        return parser

    def title(self, args):
        return "Classify images"

    def description(self, args):
        return "Look at images and tell us if an object appears inside it."

    def cost(self, args):
        return 0.05

    def duration(self, args):
        return 7200 * 3

    def keyboards(self, args):
        return "image, classification, compute, vision, fun"

    def __call__(self, args, group):
        print "Loading windows..."

        windows = []
        for file in os.listdir(args.dirpath):
            if not file.endswith(".jpg"):
                continue
            window = DetectionWindow(filepath = os.path.join(args.dirpath, file))
            windows.append(window)
            session.add(window)

        if len(windows) == 0:
            print "No windows found."
            return
        print "Found {0} windows".format(len(windows))

        windows = windows * args.trials
        random.shuffle(windows)

        print "Creating jobs..."
        counter = 0
        for chunk in chunker(windows, args.pertask):
            job = Job(group = group)
            session.add(job)
            for window in chunk:
                ic = Interconnect(window=window, job=job)
                session.add(ic)
            counter += 1
        print "Created {0} jobs".format(counter)

        session.commit()

def chunker(iterable, chunksize):
    return izip_longest(*[iter(iterable)]*chunksize)
