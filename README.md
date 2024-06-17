### Installing and Using

- Install the needed packages for python

```shell
  pip install matplotlib scipy open3d PyQt5
```

Install from PyPi

```
pip install gasparse
```

Build and Install from source
```shell
  git clone https://github.com/Numerical-Geometric-Algebra/gasparse.git
  cd gasparse
  pip install .
```

Installing from wheels 
```shell
  git clone https://github.com/Numerical-Geometric-Algebra/gasparse.git
  cd gasparse
  pip install wheelhouse/gasparse-0.0.5a0-cp37-cp37m-manylinux_2_5_i686.manylinux1_i686.manylinux_2_17_i686.manylinux2014_i686.whl
```
See the gasparse/wheelhouse folder to find the appropriate wheel for your system. `cp37` means python version 3.7 `i686` and `x86_64` are the computer arquitectures. 


To test with dcp clone `DCP` inside this folder

```shell
git clone git@github.com:WangYueFt/dcp.git
```
for the Go-ICP install the python package
```shell
pip install py-goicp
```
for teaser follow the instructions in [TEASER++ for Python](https://github.com/MIT-SPARK/TEASER-plusplus?tab=readme-ov-file#minimal-python-3-example)


## Code Structure

There are **two** main runnable scripts the `benchmark.py` script and the `point_cloud_vis.py` script

- `benchmark.py` is used to benchmark multiple different algorithms, for a specific dataset. The user can specify the algorithms and the dataset.
- `point_cloud_vis.py` is a point cloud visualizer which uses the Open3D library. Beyond visualising points it enables
    1. Visualise the primitives of each point cloud
    1. Apply rigid transformations to the point clouds (via axis angle)
    1. Estimate a motor (rigid transformation) from a selected algorithm (aligns the target point cloud with that motor) 

In the script `algorithms_pcviz.py` are the algorithms used for the visualizer, the user can add an algorithm to that script. The specifications of that function are described in that script.

Script `algorithms.py` lists the algorithms for benchmarking. It is used by `benchmark.py` to benchmark those algorithms.
