import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import spatial
from tensorboardX import SummaryWriter
import bottleneck as bn

def cf(data, uid):
    dist=[]
    idx=0
    for i in range(500):
        if idx==uid: 
            idx += 1
            dist.append(-1)
            continue
        
        else: 
           
            dist.append(spatial.distance.cosine(data[uid], data[idx]))
        idx += 1
    
    return dist
DATA_DIR = 'C:/Users/SH420/vae_cf/recommendation/ml-20m'

recon = pd.read_csv(os.path.join(DATA_DIR, "reconstructed/recon.csv"))
recon = recon.replace(-np.inf, -100.0)
recon = np.array(recon)


result_dist = cf(recon, 0)
closest = result_dist.index(max(result_dist))

result_dist = pd.DataFrame(result_dist, columns=['dist'])
result_dist.to_csv("C:/Users/SH420/vae_cf/recommendation/cf_list.csv", index=False)
