# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
import csv

def write_to_csv(scores, params_dict, outputfile):
    """This function writes the parameters and the scores with their names in a
    csv file."""

    
    outputdir = os.path.dirname(outputfile)  
    # Create the directory if it doesn't exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    file = open(outputfile, 'a', newline='')
  
    if os.stat(outputfile).st_size == 0:
        # Writes the configuration parameters
        for key in params_dict.keys():
            file.write(key+",")
        for i, key in enumerate(scores.keys()):
            ending = "," if i < len(scores.keys())-1 else ""
            file.write(key+ending)
        file.write("\n")
       

    file.close()

    # Writes the values to each corresponding column.
    with open(outputfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)

    # Iterates over the header names and write the corresponding values.
    with open(outputfile, 'a') as f:
        for i, key in enumerate(headers):
            ending = "," if i < len(headers)-1 else ""
            if key in params_dict:
                f.write(str(params_dict[key])+ending)
            elif key in scores:
                f.write(str(scores[key])+ending)
            else:
                #raise AssertionError("Key not found in the given dictionary")
                f.write(""+ending)
        f.write("\n")




