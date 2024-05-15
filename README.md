
### Installing and Using

- Install the needed packages for python

```shell
    pip install matplotlib scipy open3d PyQt5 gasparse
```

## Code Structure

There are **two** main runnable scripts the `benchmark.py` script and the `point_cloud_vis.py` script

- `benchmark.py` is used to benchmark multiple different algorithms, for a specific dataset. The user can specify the algorithms and the dataset.
- `point_cloud_vis.py` is a point cloud visualizer which uses the Open3D library. Beyond visualising points it enables
    1. Visualise the primitives of each point cloud
    1. Apply rigid transformations to the point clouds (via axis angle)
    1. Estimate a motor (rigid transformation) from a selected algorithm (aligns the target point cloud with that motor) 

In the script `algorithms_pcviz.py` are the algorithms used for the visualizer, the user can add an algorithm to that script. The specifications of that function are described in that script.

Script `algorithms.py` lists the algorithms for benchmarking. It is used by `benchmark.py` to benchmark those algorithms.
