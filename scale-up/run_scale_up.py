#========================================================================
#         Author:  Ang Li, PNNL
#        Website:  http://www.angliphd.com  
#        Created:  03/19/2018 03:09:45 PM, Richland, WA, USA.
#========================================================================

# This script is used to test multiple applications with different scale and GPU configurations using different schemas.
# It outputs the execution time of each application at each configuration to separate result files for each schema.

import string
import subprocess
import os
import math

#============================= CONFIG ===============================
TIMES = 5
gpus = [1,2,4,8]
#gpus = [1,2] # avoid running with higher number of GPUs until required
scale = ["strong","weak"]
#schms = ["scale-up", "scale-up-nvlink"] # commented since only scale-up will be used
schms = ["scale-up"]
#====================================================================

apps = []
#============================== APP =================================
montercarlo = ["montecarlo", "MTC"]

apps.append(montercarlo)
#====================================================================

# Define a function to run an application with different scales and GPUs
def run_one_app(app, outfile, schm):
    os.chdir(app[0])
    os.system("make clean")
    os.system("make")

    # Iterate over different scale and GPU configurations
    for s in scale:
        for g in gpus:
            # Construct command string to run the shell script and measure execution time
            cmd = str("/usr/bin/time -f '%e' ./run_") + str(g) + str("g_") + str(s) + ".sh"
            # Print command and application info
            print(str('$Run ') + app[0] + ':' + cmd)
            time = 0.0
            # Run the command 5 times and compute the average execution time
            for t in range(0,TIMES):
                time += float(subprocess.getoutput(cmd).split('\n')[-1])
            time /= TIMES
            # Format output line with schema, app name, scale, GPU, and execution time
            line = str(schm) + "," + str(app[1]) + "," + str(s) + "," + str(g) + "," + str(time)
            print(line)
            outfile.write(line + "\n")

    os.chdir("..")


for schm in schms:
    # Create an output file for the current schema
    outfile_name = str("res_") + str(schm) + ".txt"
    outfile_path = str("./result/") + outfile_name
    if not os.path.exists("./result"):
        os.mkdir("./result")
    outfile = open(outfile_path, "w")

    os.chdir(schm)
    # Iterate over the list of applications
    for app in apps:
        # Run the current application with different scales and GPUs
        run_one_app(app, outfile, schm)
        
    os.chdir("..")
    outfile.close()
