# MeshProcess 

This repository **does not propose any new methods**, but provides easy-to-use code that leverages advanced tools for efficiently preparing large-scale, high-quality 3D mesh assets.

## Introduction

### Main Features 
- Generate simplified 2-manifold meshes using [OpenVDB](https://www.openvdb.org/) and [ACVD](https://github.com/valette/ACVD).
- Perform convex decomposition with [CoACD](https://github.com/JYChen18/CoACD).
- Render single-view point clouds using [Warp](https://github.com/NVIDIA/warp).

### Highlights
1. **Fast**: Processes each mesh in seconds, with parallel processing support for multiple meshes.
2. **Robust**: Achieves over 95% success rate for processing messy mesh data, including datasets like [Objaverse](https://objaverse.allenai.org).
3. **Easy to Use**: Customizable processing tasks that can be executed with a single command.

### Projects Using MeshProcess
- [BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization](https://pku-epic.github.io/BODex/), by Jiayi Chen*, Yubin Ke*, and He Wang. ICRA 2025.

## Getting Started
### Installation
1. Clone the third-party dependencies.
```
cd MeshProcess
git submodule update --init --recursive --progress
```

2. Create and set up the Python environment using [Conda](https://docs.anaconda.com/miniconda/).
```
conda create -n meshproc python=3.10    
conda activate meshproc

conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia  -y
pip install mkl==2024.0.0

pip install mujoco==3.2.7
pip install trimesh
pip install hydra-core
pip install lxml
pip install imageio
pip install tqdm
pip install scipy

# For partial point cloud rendering
pip install warp-lang
pip install opencv-python
pip install pyglet

pip uninstall numpy
pip install numpy==1.26.4
```

3. Build the third-party package ACVD following their [installation guide](https://github.com/valette/ACVD/tree/master?tab=readme-ov-file#simple-compilation-howto-under-linux). To install the [VTK](https://www.vtk.org/) dependencies of ACVD,
```bash
sudo apt-get update
sudo apt install -y build-essential cmake git unzip qt5-default libqt5opengl5-dev libqt5x11extras5-dev libeigen3-dev libboost-all-dev libglew-dev libglvnd-dev

git clone https://gitlab.kitware.com/vtk/vtk.git
cd vtk
git checkout v9.2.0     
mkdir build
cd build

# If you have sudo permission
cmake ..
make -j12
sudo make install
export VTK_DIR=/usr/local/include/vtk-9.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# else
cmake -DCMAKE_INSTALL_PREFIX=<your_path> ..
make -j12
make install
export VTK_DIR=<your_path>/lib/cmake/vtk-9.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your_path>/lib
``` 

4. Build our customized CoACD by following the [compiling instructions](https://github.com/JYChen18/CoACD?tab=readme-ov-file#3-compile). Please note that we do not support CoACD installed via `pip`. OpenVDB will be automatically included after compiling CoACD.


### Running Example Data
For a quick start, we provide example mesh data in the `assets/object/example_obj/raw_mesh` directory, which can be processed by our all-in-one script.
```
bash script/example.sh
```

### Debug
If you want to debug, you can specify an object id (string) in `src/config/base.yaml` and set the `quiet` in `src/config/task/bodex/official.yaml` to True to see the intermediate log.

### Running New Data
1. **Downloading datasets**: Please see the [guides](https://github.com/JYChen18/MeshProcess/tree/main/src/dataset#dataset-download) for downloading and processing object assets from [DexGraspNet](https://pku-epic.github.io/DexGraspNet/) and [Objaverse](https://objaverse.allenai.org/objaverse-1.0). You can prepare your raw mesh data similarly.

2. **Processing meshes**: Results will be saved in `assets/object/DGN_obj/processed_data`.
```
python src/main.py func=proc data=DGN
```

3. **(Optional) Getting statistics**: This can be used to monitor the progress during processing meshes.
```
python src/main.py func=stat data=DGN
```

4. **Recording and spliting valid data**: Results will be saved in `assets/object/DGN_obj/valid_split`. 
```
python src/main.py func=split data=DGN
```

5. **Rendering partial point cloud and (optional) images**: Results will be saved in `assets/object/DGN_obj/vision_data`. This is not needed for grasp pose synthesis, but necessary for network learning.
```
python src/main.py func=render data=DGN
```

### AnyScaleGrasp (Mingrui)

The `raw_mesh` and `processed_mesh` are copied from BODex.

The object scales are defined in `src/config/task/scene_cfg.yaml`
```bash
# Check the script to see what is enabled or disabled.
bash script/AnyScaleGrasp_DGN.sh 96 # n_worker
```