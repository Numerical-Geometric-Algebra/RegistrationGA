# NOTES

### Comparing using the 3DMatch data set

 - Subsample the points from the point cloud
    + Does python-open3d have tools for sampling?
    + If not find other library/tools
 - Go-ICP needs to have the input point clouds between $[0,1]^3$
    + Don't test Go-ICP for 3DMatch?

- Only do sampling if there are more then 50 000

- Add a sampling slider to the visualizer

Allways sample about 50 000 points

if the point cloud has N points then the ratio has to be
ratio = 50 000/N

nbr_sampled = nbr_points/every_k_points

ratio = nbr_sampled/nbr_points

random sampling does not work well because it affects the primitives too much. So we are using uniform down sampling.

Sampling has to be the same for both point clouds otherwise we are truly and surely fucking fucked really bad.


### Comparing VS rotation angle

Plot the algorithms with increased angle of rotation
