from geo_algebra import *


for i in range(10):
    R = gen_rdn_CGA_rotor()
    R_sq = the_other_rotor_sqrt(R)
    print("Type of rotor:",(R(2)*~R(2))(0))
    print("Normalized R:",(R*~R)(0))
    print("Normalized R_sq:",(R_sq*~R_sq)(0))
    if R(0) < 1:
        print("theta:",np.arccos(R(0))/np.pi*180)
    else:
        print("gamma:",np.arccosh(R(0)))

    print(abs(np.array((R_sq*R_sq - R).tolist()[0])).max())
    print()