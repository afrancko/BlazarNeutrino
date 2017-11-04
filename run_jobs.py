#!/usr/bin/env python

import pydag
import itertools
import sys
import os
import argparse
import time


def parseArguments():
    # Create argument parser
    # Positional mandatory arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--Rescue",
                        help="If true, run in rescue mode ",
                        action="store_true")
    parser.add_argument("--version",
                        action="version",
                        version='%(prog)s - Version 1.0')

    args = parser.parse_args()
    return args


args = parseArguments().__dict__
if __name__ == '__main__':
    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args:
        print(str(a) + ": " + str(args[a]))
    print"############################################\n "

Resc = args["Rescue"]
user = "tglauch"
num_jobs = 200
sample = 'EHE'
PROCESS_DIR = "/data/user/{}/EHE/BlazarNeutrino/dagman_files/{}/".format(user,sample)
WORKDIR = PROCESS_DIR + "jobs/"
script = "/data/user/{}/EHE/BlazarNeutrino/llh.py".format(user)
dagFile = WORKDIR + "job.dag"
submitFile = WORKDIR + "job.sub"

if not Resc:
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print "Created New Folder in:  " + WORKDIR

    path = os.path.join(PROCESS_DIR, "logs/")
    if not os.path.exists(path):
        os.makedirs(path)
        print "Created New Folder in:  " + path
    print "Write Dagman Files to: " + submitFile

    arguments = " $(ARG0)"

    submitFileContent = {"getenv": True,
                         "universe": "vanilla",
                         "notification": "Error",
                         "log": "$(LOGFILE).log",
                         "output": "$(LOGFILE).out",
                         "error": "$(LOGFILE).err",
                         "request_memory": "2GB",
                         "arguments": arguments}

    submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
                                               script,
                                               **submitFileContent)
    submitFile.dump()

    nodes = []
    for i in range(num_jobs):
        logfile = path + '/' + str(i)
        print logfile
        dagArgs = pydag.dagman.Macros(
            LOGFILE=logfile,
            ARG0=i)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)

    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()

os.system("condor_submit_dag -f " + dagFile)
time.sleep(1)
