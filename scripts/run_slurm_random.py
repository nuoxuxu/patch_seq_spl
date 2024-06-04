from itertools import product
import pandas as pd
from pathlib import Path
import pickle

######### updating params list ##############
ephyts_prop_list = pd.read_csv("/nethome/kcni/nxu/CIHR/proc/ephys_data_tt.csv", index_col=0).columns

with open("/nethome/kcni/nxu/CIHR/proc/iterable_list.pkl", "rb") as f:
    iterable_list = pickle.load(f)

if len(iterable_list) == 1:
    iterable_list = iterable_list[0]

seed_list = range(50)
params =  product(ephyts_prop_list, seed_list)
params = [p for p in params]

with open("/nethome/kcni/nxu/CIHR/proc/params.pkl", "wb") as f:
    pickle.dump(params, f)

processed = [file.name.removesuffix(".csv") for file in Path("/external/rprshnas01/netdata_kcni/stlab/Nuo/random_GLM_results").iterdir() if file.name.endswith(".csv")]

if len(processed) == 0:
    unprocessed = params
else:
    processed = [tuple(name.rsplit("_", 1)) for name in processed]
    processed = [i for i in zip([my_tuple[0] for my_tuple in processed], [int(my_tuple[1]) for my_tuple in processed])]
    unprocessed = list(set(params) - set(processed))

with open("/nethome/kcni/nxu/CIHR/proc/params.pkl", "wb") as f:
    pickle.dump(unprocessed, f)


chunksize=1000
chunk_size = len(unprocessed) // chunksize
iterable_list = [unprocessed[i*chunk_size:(i+1)*chunk_size] for i in range(chunksize)]
iterable_list[-1] += unprocessed[chunksize*chunk_size:]

with open("/nethome/kcni/nxu/CIHR/proc/params_list.pkl", "wb") as f:
    pickle.dump(iterable_list, f)



# unprocessed = unprocessed[:1000]

# with open("/nethome/kcni/nxu/CIHR/proc/params.pkl", "wb") as f:
#     pickle.dump(unprocessed, f)