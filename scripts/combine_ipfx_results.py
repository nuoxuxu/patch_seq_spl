import numpy as np
from tqdm import tqdm
import pandas as pd
# read all npz file in a directiory, they are all python dictionaries, and combine them into one python dictionary

def combine_npz_files(directory):
    import os
    import numpy as np
    import re
    import pandas as pd

    # get all files in the directory
    files = os.listdir(directory)
    files = [f for f in files if re.search('.npz', f)]
    
    # read all files and combine them into one dictionary
    all_data = {}
    for f in tqdm(files):
        data = np.load(directory + f)
        for key in data.keys():
            if key in all_data.keys():
                all_data[key] = np.append(all_data[key], data[key])
            else:
                all_data[key] = data[key]
    return all_data

ipfx_results = combine_npz_files("proc/ipfx/")
pd.DataFrame(ipfx_results).to_csv("proc/ipfx_results.csv", index=False)