import gasparse
import numpy as np
import geo_algebra as pyga
import matplotlib.pyplot as plt
from multilinear_algebra import *

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib.animation as animation

from sympy import Matrix

import time


VGA3D = gasparse.GA(3)
DCGA3D = gasparse.GA(metric=[1,1,1,1,-1,1,1,1,1,-1],compute_mode="large")
POW4GA = gasparse.GA(4,11,compute_mode="large")

locals().update(VGA3D.basis())

# Use the precision of the matissa of double 
# exponent: np.log10(2**(53-1))
# precision = 1/2**(53-1)
precision = 1e-100
VGA3D.set_precision(precision)
DCGA3D.set_precision(precision)
POW4GA.set_precision(precision)

basis_POW4GA = list(POW4GA.basis(grades=1).values())
basis_DCGA3D = list(DCGA3D.basis(grades=1).values())

''' Define the vector basis for the two conformal geometric algebras of DCGA'''
e1_1 = basis_DCGA3D[0]
e1_2 = basis_DCGA3D[1]
e1_3 = basis_DCGA3D[2]
e1_4 = basis_DCGA3D[3]
e1_5 = basis_DCGA3D[4]

e2_1 = basis_DCGA3D[5]
e2_2 = basis_DCGA3D[6]
e2_3 = basis_DCGA3D[7]
e2_4 = basis_DCGA3D[8]
e2_5 = basis_DCGA3D[9]

# compute the cga's null basis
einf1 = (1.0*e1_5 - 1.0*e1_4)*(1/np.sqrt(2))
eo1 = (1.0*e1_5 + 1.0*e1_4)*(1/np.sqrt(2))

einf2 = (1.0*e2_5 - 1.0*e2_4)*(1/np.sqrt(2))
eo2 = (1.0*e2_5 + 1.0*e2_4)*(1/np.sqrt(2))

''' Define the bivector basis and the reciprocal for the embedding of DCGA '''

# x1*x^2, x2*x^2, x3*x^2 
f1 = (1/2)*(e1_1*einf2 + einf1*e2_1)
f2 = (1/2)*(e1_2*einf2 + einf1*e2_2)
f3 = (1/2)*(e1_3*einf2 + einf1*e2_3)
f4 = eo1*eo2

# the reciprocal basis
# x1, x2, x3
rf1 = e1_1*eo2 + eo1*e2_1
rf2 = e1_2*eo2 + eo1*e2_2
rf3 = e1_3*eo2 + eo1*e2_3
rf4 = -einf1*einf2

# x1*x2, x2*x3, x3*x1
g1 = (e2_2*e1_1 + e2_1*e1_2)*(1/np.sqrt(2))
g2 = (e2_2*e1_3 + e2_3*e1_2)*(1/np.sqrt(2))
g3 = (e2_3*e1_1 + e2_1*e1_3)*(1/np.sqrt(2))

# x^2 = x1^2 + x2^2 + x3^2
g4 = (eo2*einf1 + einf2*eo1)*(1/np.sqrt(2))

rg1 = -g1
rg2 = -g2
rg3 = -g3
rg4 = -g4

# x1^2, x2^2, x3^2
h1 = e2_1*e1_1
h2 = e2_2*e1_2
h3 = e2_3*e1_3

# reciprocal basis
rh1 = -h1
rh2 = -h2
rh3 = -h3

# the embedding basis
emb_basis_DCGA3D = [f1,f2,f3,f4,rf1,rf2,rf3,rf4,g1,g2,g3,g4,h1,h2,h3]
emb_recbasis_DCGA3D = [rf1,rf2,rf3,rf4,f1,f2,f3,f4,rg1,rg2,rg3,rg4,rh1,rh2,rh3]

''' Define the vector basis for the POW4GA geometric algebra '''

# Positive basis
a1 = basis_POW4GA[0]
a2 = basis_POW4GA[1]
a3 = basis_POW4GA[2]
a4 = basis_POW4GA[3]

# negative basis
a5 = basis_POW4GA[4]
a6 = basis_POW4GA[5]
a7 = basis_POW4GA[6]
a8 = basis_POW4GA[7]
a9 = basis_POW4GA[8]
a10 = basis_POW4GA[9]
a11 = basis_POW4GA[10]
a12 = basis_POW4GA[11]
a13 = basis_POW4GA[12]
a14 = basis_POW4GA[13]
a15 = basis_POW4GA[14]

POW4GA_pss = 1
for i in range(len(basis_POW4GA)):
    POW4GA_pss ^= basis_POW4GA[i]

''' Define the null vector basis for the POW4GA geometric algebra '''

w1 = (1.0*a1 - 1.0*a5)*(1/np.sqrt(2))
w2 = (1.0*a2 - 1.0*a6)*(1/np.sqrt(2))
w3 = (1.0*a3 - 1.0*a7)*(1/np.sqrt(2))
w4 = (1.0*a4 - 1.0*a8)*(1/np.sqrt(2))

rw1 = (1.0*a1 + 1.0*a5)*(1/np.sqrt(2))
rw2 = (1.0*a2 + 1.0*a6)*(1/np.sqrt(2))
rw3 = (1.0*a3 + 1.0*a7)*(1/np.sqrt(2))
rw4 = (1.0*a4 + 1.0*a8)*(1/np.sqrt(2))


emb_basis_POW4GA = [w1,w2,w3,w4,rw1,rw2,rw3,rw4,a9,a10,a11,a12,a13,a14,a15]
emb_recbasis_POW4GA = [rw1,rw2,rw3,rw4,w1,w2,w3,w4,-a9,-a10,-a11,-a12,-a13,-a14,-a15]

basis = list(VGA3D.basis(grades=1).values())

b1 = basis[0]
b2 = basis[1]
b3 = basis[2]

def emb_lin_map(x):
    x1 = (x|b1)(0)
    x2 = (x|b2)(0)
    x3 = (x|b3)(0)
    return (x1*e1_1 + x2*e1_2 + x3*e1_3, x1*e2_1 + x2*e2_2 + x3*e2_3)

def emb_lin_map2d(x):
    x1 = (x|b1)(0)
    x2 = (x|b2)(0)
    return (x1*e1_1 + x2*e1_2, x1*e2_1 + x2*e2_2)

def DCGA_embbeding(x):
    x_1,x_2 = emb_lin_map(x)
    x_sq = (x*x)(0)
    p1 = eo1 + x_1 + (1/2)*x_sq*einf1
    p2 = eo2 + x_2 + (1/2)*x_sq*einf2
    return p1^p2

def DCGA_embbeding2d(x):
    x_1,x_2 = emb_lin_map2d(x)
    x_sq = (x*x)(0)
    p1 = eo1 + x_1 + (1/2)*x_sq*einf1
    p2 = eo2 + x_2 + (1/2)*x_sq*einf2
    return p1^p2

# The mapping from DCGA3d to POW4GA
def gas_mapping(X):
    out = 0
    for i in range(len(emb_recbasis_DCGA3D)):
        out += (X|emb_recbasis_DCGA3D[i])(0)*emb_basis_POW4GA[i]
    return out

def POW4GA_embedding(x):
    P = DCGA_embbeding(x)
    p = gas_mapping(P)
    return p

def POW4GA_embedding2d(x):
    P = DCGA_embbeding2d(x)
    p = gas_mapping(P)
    return p

def do_it_in_3D():
    x1 = np.linspace(-5, 5, 10)
    x2 = np.linspace(-5, 5, 10)
    x3 = np.linspace(-5, 5, 10)

    x1, x2, x3 = np.meshgrid(x1, x2, x3)
    xvga_array = np.vstack((x1.ravel(), x2.ravel(), x3.ravel())).T

    xvga_list = xvga_array.tolist()
    xvga = VGA3D.mvarray(xvga_list,grades=1)

    ''' Create a shape by wedging 14 vectors in the POW4GA embedding space '''

    # Generate 14 vectors in 3D euclidean space
    zvga_list = (np.random.rand(14,3) - 0.5).tolist()
    # Convert them to VGA
    zvga_array = VGA3D.mvarray(zvga_list,grades=1)
    # Apply the embedding of the points into POW4GA
    z_array = POW4GA_embedding(zvga_array)
    # Wedge all points and take the dual  

    z = 1
    for i in range(len(z_array)):
        z ^= z_array[i]*100

    # z = (z.dual())(1)
    # z = (z_array.prod()(14).dual())(1)
    # z = z/(z|rw4)(0)

    x = POW4GA_embedding(xvga)

def iterative_solution(x0,z,n=100,eps=1e-5):
    x = x0
    u = pyga.normalize(x)
    alpha0 = 1e-3
    basis,rbasis = pyga.get_basis(VGA3D,1)
    basis = [basis[0],basis[1]]
    rbasis = [rbasis[0],rbasis[1]]
    for i in range(n):
        v = 0
        for j in range(len(basis)):
            v += rbasis[j]*(POW4GA_embedding(x+basis[j]*eps/2)|z)(0)
            v -= rbasis[j]*(POW4GA_embedding(x-basis[j]*eps/2)|z)(0)
        v /= eps
        u = pyga.normalize((u^v)*pyga.inv(v))
        x += alpha0*u
        print("u|v=",u|v,sep='')
        print("c(x)|z=",POW4GA_embedding(x)|z,sep='')
        print(x)
    return x

# The unit pseudoscalar of two dimensions POW4GA
pss2d = w1^w2^rw1^rw2^a9^a13^a14^a12^w4^rw4

# def if_it_does_not_work_do_it_in_2d():

# ngridpoints = 100
# x1 = np.linspace(-10, 10, ngridpoints)
# x2 = np.linspace(-10, 10, ngridpoints)

ngridpoints = 100
# x1 = np.linspace(0.95, 1.08, ngridpoints)
# x2 = np.linspace(0.94, 1.06, ngridpoints)

# x1 = np.linspace(0.95, 1.08, ngridpoints)
# x2 = np.linspace(0.94, 1.06, ngridpoints)

x1 = np.linspace(0.95, 1.15, ngridpoints)
x2 = np.linspace(1, 1.1, ngridpoints)

# x1 = np.linspace(0.5, 1.5, ngridpoints)
# x2 = np.linspace(0.5, 1.06, ngridpoints)

x1, x2 = np.meshgrid(x1, x2)
xvga_array_stacked = np.vstack((x1.ravel(), x2.ravel())).T

xvga_array = np.c_[x1.reshape(ngridpoints,ngridpoints,1),x2.reshape(ngridpoints,ngridpoints,1)]

xvga_list = xvga_array.tolist()
xvga = VGA3D.mvarray(xvga_list,basis=['e1','e2'])

# Convert the points to the POW4GA
x = POW4GA_embedding(xvga)

''' Create a shape by wedging 9 vectors in the POW4GA embedding space '''

zvga_nparray = np.array(
[[1.01000319, 1.09313873],
 [1.04257296, 1.06897817],
 [1.00051724, 1.0466953 ],
 [1.02002827, 1.07689709],
 [1.01755883, 1.00896712],
 [1.04267048, 1.030304  ],
 [1.09397318, 1.05321543],
 [1.0038194 , 1.05568863],
 [1.0405772 , 1.05677772]])

zvga_nparray = np.array(
[[1.0122189349596487, 1.050056906769775], 
 [1.0685790735905023, 1.0052292835561354], 
 [1.066297986455385, 1.0091334687926636]])

zvga_nparray = np.array(
[[1.009923212650331, 1.0148393786254895], 
 [1.0154027912114847, 1.0557079385643007], 
 [1.067667829203872, 1.0624246413641358], 
 [1.009900642158677, 1.0544771891872315]])

zvga_nparray = np.array(
[[1.0995403862628101, 1.0311711150832747], 
 [1.0587597631937122, 1.0283240723345108], 
 [1.004862254401609, 1.0052129677967139], 
 [1.0848037527720291, 1.0247667129123195], 
 [1.0429377523412613, 1.0928812134402937]])

# zvga_nparray = np.array(
# [[1.0052931535020837, 1.035626318511413], 
#  [1.0838294664436119, 1.0341699735296652], 
#  [1.0370148491357816, 1.0410446109421212], 
#  [1.0512940323123343, 1.0476029234408972], 
#  [1.0500566731197145, 1.0335862378968574], 
#  [1.0921387168292105, 1.060897689354746]])

n_vecs = 20

# Generate 9 vectors in 2D euclidean space
zvga_nparray = 10*(np.random.rand(n_vecs,2) - 0.5)
# zvga_nparray = 0.1*np.random.rand(6,2) + 1
zvga_list = zvga_nparray.tolist()
# Convert them to VGA
zvga_array = VGA3D.mvarray(zvga_list,basis=['e1','e2'])
# Apply the embedding of the points into POW4GA
z_array = POW4GA_embedding(zvga_array)
# Wedge all points and take the dual  

basis = [w1,w2,rw1,rw2,a9,a13,a14,a12,w4,rw4]
rbasis = [rw1,rw2,w1,w2,-a9,-a13,-a14,-a12,rw4,w4]

basis = mv.concat(rbasis)

bas = list(POW4GA.basis(grades=1).values())
rbas = []

for i in range(len(bas)):
    rbas += [pyga.inv(bas[i])]

def f(Z):
    return ((Z|z_array)*z_array).sum()

def pos_norm(x):
    return ((x|rbasis)(0)*(x|rbasis)(0)).sum()(0)


def power_iter_alg(f,x0,niters=100):
    x = basis
    # x = x0
    for i in range(niters):
        x = f(x)
        x = x/np.sqrt(pos_norm(x))

    return x,(f(x)*pyga.inv(x))(0)


f_matrix = get_matrix(f,basis,rbasis)
eigvals,eigvecs = np.linalg.eig(f_matrix)



# A = Matrix(z_array.tolist(1)[0])

# N1 = A.nullspace()
# N2 = list((np.array(N1).reshape(len(N1),15)).astype(np.float64).tolist())
# N = POW4GA.mvarray(N2,basis=rbas)

# n = N[1]
# n = N[2] + 10*N[7]
# # n = N[7]
# # n = N[2]
# # n = N[2] + N[7]
# # n = N[3]
# # n = N[4]
# # n = N[5]
# # n = N[8]
# # n = N[9]
# # n = N[10]

# n = n/np.sqrt(pos_norm(n)) # numerically stable

# inner_prod = abs(np.array((x|n)(0).tolist(0)[0]))
# inner_prod = inner_prod[:,:,0]
# # thrsh = abs(inner_prod).min()*2
# thrsh = 1e-5
# mask = inner_prod <= thrsh
# print(np.count_nonzero(mask))
# x_masked = xvga_array[mask]


# xx = xvga_array[:,:,0]
# yy = xvga_array[:,:,1]
# zz = inner_prod
# levels = MaxNLocator(nbins=100).tick_values(zz.min(), zz.max())
# # cmap = plt.colormaps['plasma']
# # cmap = plt.colormaps['PiYG']
# cmap = plt.colormaps['RdYlBu']
# # cmap = plt.colormaps['PRGn']
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# fig, (ax1) = plt.subplots(nrows=1)
# cf = ax1.contourf(xx, yy, zz, levels=levels, cmap=cmap)
# # c = ax1.contour(xx, yy, zz, colors='k')
# # cf = ax1.pcolormesh(x, y, z)
# ax1.scatter(x_masked.T[0],x_masked.T[1])
# ax1.scatter(zvga_nparray.T[0],zvga_nparray.T[1],color='orange')

# fig.colorbar(cf, ax=ax1)
# # ax1.set_title('contourf with levels')
# fig.tight_layout()
# plt.show()
# # eigvec,eigval = power_iter_alg(f,z_array[0])


# # f_matrix = get_matrix(f,basis,rbasis)
# # f_inverse = np.linalg.inv(f_matrix)



'''
ga2d_size = 10
# z = z_blade|pss2d

# # outer_prod = abs(np.array((x|(z_blade|pss2d)).tolist(0)[0]))
# outer_prod = abs(np.array((z|x).tolist(ga2d_size-grade-1)[0]))
# inner_prod = np.max(outer_prod,axis=-1)


# outer_prod = abs(np.array((100*((100*(x^z_array[0]))^z_array[1])^z_array[2]^z_array[3]^z_array[4]^z_array[5]^z_array[6]).tolist(8)[0]))
# outer_prod = abs(np.array((z_blade|pss2d).tolist(0)[0]))

# inner_prod = abs(np.array((x|z1)(0).tolist(0)[0]))
# inner_prod = abs(np.array((x|z)(0).tolist(0)[0]))
# inner_prod = inner_prod.reshape(inner_prod.shape[:-1])
# a = np.array((z|z_array).tolist(0)[0])
# thrsh = abs(a).min()*2
thrsh = pyga.numpy_max(z|z_array)
# thrsh = inner_prod.min()*50

mask = inner_prod <= thrsh
# mask = mask.reshape(len(mask))
print(np.count_nonzero(mask))
x_masked = xvga_array[mask]

# plt.figure()
# plt.scatter(x_masked.T[0],x_masked.T[1])
# plt.scatter(zvga_nparray.T[0][0:grade],zvga_nparray.T[1][0:grade])
# plt.scatter(zvga_nparray.T[0],zvga_nparray.T[1])


# index = np.argmin(inner_prod)
xx = xvga_array[:,:,0]
yy = xvga_array[:,:,1]
zz = inner_prod
levels = MaxNLocator(nbins=100).tick_values(zz.min(), zz.max())
# cmap = plt.colormaps['plasma']
# cmap = plt.colormaps['PiYG']
cmap = plt.colormaps['RdYlBu']
# cmap = plt.colormaps['PRGn']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig, (ax1) = plt.subplots(nrows=1)
cf = ax1.contourf(xx, yy, zz, levels=levels, cmap=cmap)
# c = ax1.contour(xx, yy, zz, colors='k')
# cf = ax1.pcolormesh(x, y, z)
ax1.scatter(x_masked.T[0],x_masked.T[1])
ax1.scatter(zvga_nparray.T[0][0:grade],zvga_nparray.T[1][0:grade],color='orange')

fig.colorbar(cf, ax=ax1)
# ax1.set_title('contourf with levels')
fig.tight_layout()
# plt.show()

#  Estimate the motor that is applied iteratively 
niters = 50
z1 = pyga.normalize(z1)
z2 = pyga.normalize(z2)

# Compute the rotor that rotates z1 to z2
S = 1 + z2*pyga.inv(z1)
R = pyga.normalize(S)
B = pyga.normalize(R(2))

# Compute the rotor 
if (B*B)(0) > 0:
    # hyperpolic rotations
    theta = np.arctanh((R(2)|B)(0)/R(0))/niters
    U = np.cosh(theta) + B*np.sinh(theta)
else: 
    # elliptical rotations
    theta = np.arctan2((R(2)|B)(0),R(0))/niters
    U = np.cos(theta) + B*np.sin(theta)

w = z1

sol = iterative_solution(1.05*e1+1.05*e2,z,n=500)



def animate(i):
    
    global c,cf,w,xvga_array

    w = (U*w*~U)(1)
    
    timenow = time.time()
    scalar = abs((x|w)(0))
    print("The inner Product took:",time.time() - timenow)
    timenow = time.time()
    inner_prod = np.array(scalar.tolist(0)[0])
    print("Transforming to a list took:",time.time() - timenow)
    zz = inner_prod.reshape(inner_prod.shape[:-1])

    for coll in cf.collections:
        coll.remove()
    # for coll in c.collections:
    #     coll.remove()

    xx = xvga_array[:,:,0]
    yy = xvga_array[:,:,1]
    
    cf = ax1.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    # c = ax1.contour(xx, yy, zz, colors='k')

    # return c, cf
    return c,cf

# anim = animation.FuncAnimation(fig, animate, frames=niters)
# plt.show()


# print(xvga[index])
# print(zvga_array[0])

# Check the reciprocity
# v = np.zeros([len(emb_basis_POW4GA),len(emb_basis_POW4GA)])
# for i in range(len(emb_basis_POW4GA)):
#     for j in range(len(emb_basis_POW4GA)):
#         v[i][j] = (emb_basis_POW4GA[i]*emb_recbasis_POW4GA[j])(0)

# print(v)

# # Print the metric
# a = np.zeros(len(emb_basis_POW4GA))
# for i in range(len(emb_basis_POW4GA)):
#     a[i] = (emb_basis_POW4GA[i]*emb_basis_POW4GA[i])(0)

# print(a)


# v = np.zeros([len(emb_basis_DCGA3D),len(emb_basis_DCGA3D)])
# for i in range(len(emb_basis_DCGA3D)):
#     for j in range(len(emb_basis_DCGA3D)):
#         v[i][j] = (emb_basis_DCGA3D[i]*emb_recbasis_DCGA3D[j])(0)

# print(v)

# # Print the metric
# a = np.zeros(len(emb_basis_DCGA3D))
# for i in range(len(emb_basis_DCGA3D)):
#     a[i] = (emb_basis_DCGA3D[i]*emb_basis_DCGA3D[i])(0)

# print(a)
'''

''' I really enjoy the fact that basis vector elements get converted to ascii symbols
    It really feels like I'm in the 70s or something. 
    I think everyone should do things as in the classic era.
'''