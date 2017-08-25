
# coding: utf-8

# In[1]:

import pandas
import os
import glob
import tqdm
import re


# In[2]:

csv = glob.glob("/data1/santiago/BBBC021/deep_profiler/features/cnn-segmentation/*.csv")
dapi = list(filter(lambda f: "DAPI" in f, csv))
tubulin = list(filter(lambda f: "Tubulin" in f, csv))
actin = list(filter(lambda f: "Actin" in f, csv))
filtered = list(filter(lambda f: "s1" in f, dapi))
out_file = "/data1/santiago/BBBC021/deep_profiler/collapsed/cnn-segmentation/Wells.csv"


# In[3]:

def get_tubulin(dap):
    return dap[:-8] + "Tubulin.csv"


# In[4]:

def get_actin(dap):
    return dap[:-8] + "Actin.csv"


# In[5]:

def get_all(dap):
    return os.path.basename(dap[:-11] + "All.csv")


# In[6]:

def get_site(dap, site):
    return re.sub('s\d', 's' + str(site), dap)


# In[7]:

everything = []
for dap in tqdm.tqdm(filtered):
    df = pandas.DataFrame()
    for site in range(1, 5):
        dap = get_site(dap, site)
        if os.path.isfile(dap):
            tub = get_tubulin(dap)
            act = get_actin(dap)
            dap_df = pandas.read_csv(dap).add_prefix("DAPI_")
            tub_df = pandas.read_csv(tub).add_prefix("Tubulin_")
            act_df = pandas.read_csv(act).add_prefix("Actin_")
            all_df = pandas.concat([dap_df, tub_df, act_df], axis=1)
            df = pandas.concat([df, all_df])
    df = df.dropna(axis=0)
    collapsed = df.mean()
    collapsed["Plate"] = os.path.basename(dap[:-16])
    collapsed["Well"] = dap.split('_')[-3]
    everything.append(collapsed)
everythingdf = pandas.DataFrame(everything)
everythingdf.to_csv(out_file, index=False)


# In[ ]:



