#!/bin/bash

N_WORKER=${1:-12}

# python src/main.py task=official func=proc data=DGN_2k
# python src/main.py task=scene_cfg func=proc data=DGN_2k n_worker=$N_WORKER
# python src/main.py func=stat data=DGN_2k n_worker=$N_WORKER
# python src/main.py func=split data=DGN_2k n_worker=$N_WORKER
python src/main.py func=render data=DGN_2k n_worker=$N_WORKER func.gpu_lst=[2,3,4,5,6,7]
