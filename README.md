
### Installing and Using

 - Install the needed packages for python
```shell
    pip install matplotlib
    pip install scipy
    pip install open3d
    pip install PyQt5
    pip install git+ssh://git@github.com/FranciscoVasconcelos/sparse-multivectors.git@v0.0.1a
```

**NOTE**: Since the repository sparse-multivectors is private to install the gasparse module the user needs to first get the ssh github permissions by [adding an ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

The Stanford Dataset is available [here](https://mega.nz/folder/4VQBwB7Z#Rv5kA9EbcHPWj3-BvYJRFw). 

## Code Structure

There are **two** main runnable scripts the *benchmark.py* script and the *point_cloud_vis.py* script

- `benchmark.py` is used to benchmark multiple different algorithms, for a specific dataset. The user can specify the algorithms and the dataset.
- `point_cloud_vis.py` is a point cloud visualizer which uses the Open3D library. Beyond visualising points it enables
    1. Visualise the primitives of each point cloud
    1. Apply rigid transformations to the point clouds (via axis angle)
    1. Estimate a motor (rigid transformation) from a selected algorithm (aligns the target point cloud with that motor) 

In the script `algorithms_pcviz.py` are the algorithms used for the visualizer, the user can add an algorithm to that script. The specifications of the function are described in that script.

Script `algorithms.py` lists the algorithms for benchmarking. It is used by `benchmark.py` to benchmark those algorithms.


