# Rigid Transformation Estimation Via Eigenmultivectors

This code describes a set of tools to estimate rotation and translations by solving a multilinear eigenvalue problem. 


### TODO
 - [ ] Test with missing data
    - Use a sphere to remove points
    - Use a plane to remove points
 - [ ] Test with outliers
   - Random points inside a box
   - Superfluous samples for some points
      - For some point xi add a superfluous point by adding noise 
 - [ ] Option to visualize the dual objects, point pairs (and planes?)
 - [ ] Improve code speed (gasparse module)
    - It would be nice if it worked in real time
- [ ] Test in real datasets like in 3DMatch
- [ ] Study the effects of noise on the solution
    - Study how the eigenmultivectors and eigenvalues get affected by noise
    - Study how the estimation gets affected by noisy eigenmultivectors
    - Prove guarantee level of convergence when some amount of noise
    - Study the effects of having more or less points