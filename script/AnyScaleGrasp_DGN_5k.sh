#!/bin/bash

N_WORKER=${1:-12}
DGN5K_SCALE_LST="[0.020,0.028,0.042,0.060,0.081,0.106,0.133,0.162,0.194,0.227,0.263,0.300]"
DGN5K_FLOATING_SCALE_OVERRIDE="task={1-export_floating_scene_cfg:{scale_lst:${DGN5K_SCALE_LST}}}"
DGN5K_TABLETOP_SCALE_OVERRIDE="task={1-export_tabletop_scene_cfg:{scale_lst:${DGN5K_SCALE_LST}}}"

# python src/main.py task=bodex func=proc data=DGN_5k n_worker=$N_WORKER

# DGN_5k uses the HumanGraspData 12-anchor power-spaced scales without changing the global scene_cfg defaults.
python src/main.py task=scene_cfg func=proc data=DGN_5k n_worker=$N_WORKER "$DGN5K_FLOATING_SCALE_OVERRIDE" "$DGN5K_TABLETOP_SCALE_OVERRIDE"
python src/main.py func=stat data=DGN_5k n_worker=$N_WORKER
python src/main.py func=split data=DGN_5k n_worker=$N_WORKER
# python src/main.py func=render data=DGN_5k n_worker=$N_WORKER func.gpu_lst=[0,1,2,3,4,5,6,7]
