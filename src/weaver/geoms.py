# -*- coding: utf-8 -*-
import vector as vec
# import PySymm
#try:
#    import Numeric as num
#except ImportError:

import numpy as num
        
# utility functions

thresh = 15.0

deg2rad = num.pi/180.0

def  soft_comp(v1, v2):
    global thresh
    diff = v1-v2
    if abs(diff) < thresh:
        return 0
    else:
        if diff < 0:
            return -1
        else:
            return 1

def set_equal(s):
    first = s[0]
    for se in s[1:]:
        if (soft_comp(first, se) != 0): return 0
    return 1

############################################################

        
def analyze_geom(span, nvertices, flag=None, oldcom = None):
    #print('FLAG IS', flag )
    #print('safg', self.flag    )
    cn = len(span)
    #print('span is', span)
    # just to see what we get: analyze with PySymm here        
    # se = PySymm.symm(nvertices, num.array(span))
    # print(se.symmetry)
    # print(se.symmetry_codes)
    #
    angles = []
    angle_atoms = []
    for i in range(cn):
        for j in range(i+1, cn):
            angles.append(vec.angle(span[i], span[j]))
            angle_atoms.append((i,j))
    angles = num.array(angles)*180.0/num.pi
    if cn == 0:
        print("\n&&& CN 0 : single atom")
        main_axis = num.array([0,0,1.0], "d")
        #secd_axis = num.array([0,1.0,0.0], "d")
        secd_axis = num.array([0,0,1.0], "d")
    elif cn == 1:
        print("\n&&& CN 1 : a stub")
        main_axis = vec.normalize(span[0])
        ref = num.array([1,0,0],"d")
        secd_axis = vec.normalize(ref - (vec.project(ref, main_axis)))
    elif cn == 2:
        print("\n&&& CN 2 : assuming a linear geometry (linker)")
        main_axis = vec.normalize(span[0]-span[1])
        # ref = num.array([1,0,0],"d")
        #secd_axis = vec.normalize(ref - (vec.project(ref, main_axis)))
        if oldcom != None :
            oldcom_angle1 = num.pi/2 - vec.angle(oldcom,main_axis)
            secd_axis = vec.normalize(oldcom*num.cos(oldcom_angle1))
        else: secd_axis = None
        #secd_axis = None
    elif cn == 3:
        print("\n&&& CN 3 : assuming a trigonal (planar) geometry")
        planar = 1
        for i in range(3):
            if (soft_comp(angles[i], 120.0) != 0) : planar = 0
        if not planar:
            print("    !! CENTER IS NOT TRIGONAL PLANAR")
            main_axis = vec.normalize(span[0]+span[1]+span[2])
        else:
            a1 = vec.cross_prod(span[0],span[1])
            a2 = vec.cross_prod(span[1],span[2])
            a3 = vec.cross_prod(span[2],span[0])
            main_axis = vec.normalize(a1+a2+a3)
        # take orthogonal axis in plane of main axis and 
        #          neighb 0
        secd_axis = vec.normalize(span[0] - (vec.project(span[0],main_axis)))
        # print(vec.angle(main_axis, secd_axis)*180.0/num.pi)
    elif cn == 4:
        print("\n&&& CN 4 : assuming a tetrahedral/square planar geometry")
        #print(angles)
        #print(angle_atoms)
        tetrahedral = 1
        smaller = []
        larger = []
        plan   = []
        exactly_planar = 0
        for i in range(6):
        #print(angles[i])
        #print(angle_atoms)
            comp = soft_comp(angles[i], 109.4712206)
            if (comp < 0) : 
                #print('smaller')
                smaller.append(i)
            if (comp > 0) : 
                #print('larger')
                larger.append(i)
            comp = soft_comp(angles[i], 180.0)
            #print(comp)
            if (comp == 0) : 
                #print('almost equal')
                plan.append(i)
        print('len of larger')
        if len(plan) == 2:
            if (angles[plan[0]] < 181.0 and angles[plan[0]] > 179.0): angles[plan[0]]=180.0
            if (angles[plan[1]] < 181.0 and angles[plan[1]] > 179.0): angles[plan[1]]=180.0
            if (angles[plan[0]]== 180.0) and (angles[plan[1]] == 180.0):
                print("    exactly planar!!!")
                exactly_planar = 1
        if (len(smaller) == 0) and (len(larger) == 0):
            print("    ideal tetrahedron (within numerical accuracy)")
            print("    picking an arbitrary axis")
            # take axis between neighb 0 and 1
            main_axis = vec.normalize(span[0]+span[1])
            # take orthogonal axis in plane of main axis and 
            #          neighb 0
            secd_axis = vec.normalize(span[0] - (vec.project(span[0],main_axis)))
            # print(vec.angle(main_axis, secd_axis)*180.0/num.pi)
        elif len(plan) == 2:
            print("    square or distorted square planar geometry")
            #print(angles)
            ## take main axis orthogonal to plane
            n00 = angle_atoms[plan[0]][0]
            n01 = angle_atoms[plan[0]][1]
            n10 = angle_atoms[plan[1]][0]
            n11 = angle_atoms[plan[1]][1]
            #print(span[n00], span[n01], span[n10], span[n11])
            a1 = vec.cross_prod(span[n00],span[n10])
            # a2 = vec.cross_prod(span[n00],span[n11])
            #a3 = vec.cross_prod(span[n01],span[n10])
            a4 = vec.cross_prod(span[n01],span[n11])
            #print(a1)
            #print(a4)
            if exactly_planar:
                print("    exactly planar")
                main_axis = vec.normalize(a1+a4)
            #print('maix_axis', main_axis)
            else:
                main_axis = vec.normalize(span[0]+span[1]+span[2]+span[3])            
                # take secd axis as center between atoms of the two 180 deg angles
            secd_axis = vec.normalize(span[n00])
#            secd_axis = vec.normalize(span[n00] + span[n10])

            if flag=='other_secd_axis': secd_axis = vec.normalize(span[n00] + span[n11])
            #print('secd axis',secd_axis)
            # orthogonalize to main axis
            secd_axis = vec.normalize(secd_axis - (vec.project(secd_axis,main_axis)))
        elif abs(len(smaller)-len(larger)) == 2:
            if len(smaller) == 2:
                print("    distorted tetrahedron (D_2d symmetry, 2 angles smaller, 4 angles larger)")
                special_set = smaller
                minor_set   = larger
            else:
                print("    distorted tetrahedron (D_2d symmetry, 2 angles larger, 4 angles smaller)")
                print("    distortion towards square planar") 
                special_set = larger
                minor_set   = smaller
            # sanity check
            print(special_set)
            print(minor_set)
            special_angles = []
            for i in special_set: special_angles.append(angles[i])
            minor_angles = []
            for i in minor_set: minor_angles.append(angles[i])
            print("    special set: %s" % str(special_angles))
            print("    minor set: %s" % str(minor_angles))
            if not set_equal(special_angles):
                print("!!! warning: special set is not equal within numerical accuracy")
                print("    %s" % special_angles)
            if not set_equal(minor_angles):
                print("!!! warning: minor set is not equal within numerical accuracy")
                print("    %s" % minor_angles)
            n1, n2 = angle_atoms[special_set[0]]
            print("    picking main axis between neighbors %d and %d (first angle in special set: %8.2f)"\
                            % (n1,n2, angles[special_set[0]]))
            main_axis = vec.normalize(span[n1]+span[n2])
            secd_axis = vec.normalize(span[n1] - (vec.project(span[n1],main_axis)))
            # print(vec.angle(main_axis, secd_axis)*180.0/num.pi)
        else:
            print("!!!!!! unknown (= unimplemented :-))) tetrahedral distortion")
            print(span)
            print(angles)
    elif cn == 6:
        print("\n&&& CN 6 : assuming a octahedral geometry")
        a180 = []
        a90  = []
        for i in range(15):
            # print(angles [i])
            comp = soft_comp(angles[i], 90.0)
            if (comp == 0) : a90.append(i)
            comp = soft_comp(angles[i], 180.0)
            if (comp == 0) : a180.append(i)
        n180 = len(a180)
        n90  = len(a90)
        print("    we have %d 180 deg and %d 90 deg angles" % (n180, n90))
        if (n180 == 3) and (n90 == 12):
            print("    looks like a proper octahedron")
            if flag == "ref_to_zaxis":
                print("    taking z and y axes as reference!!!")
                main_axis = num.array([0,0,1],"d")
                secd_axis = num.array([0,1,0],"d")
            elif flag =="ref_to_first":
                print("    taking first connector as main axis")
                main_axis = vec.normalize(span[0])
                secd_axis = None
                i = 1
                while (secd_axis == None):
                    if (vec.angle(span[0],span[i]) <= num.pi/2.0+0.01):
                        print("second axis to connector %d" % i)
                        secd_axis = vec.normalize(span[i] -(vec.project(span[i],main_axis)))
                    i += 1
                    if i == 7: print("ERROR")
            else:
                print("    taking two spanning vectors with 90 deg angle as axes")
                n0, n1 = angle_atoms[a90[0]]
                main_axis = vec.normalize(span[n0])
                secd_axis = vec.normalize(span[n1] - (vec.project(span[n1],main_axis)))
                # print(main_axis)
                # print(secd_axis)
        elif (n180 == 3):
            print("    not a perfect octahedron, but we have three 180 degree angles")
            print("    take them as primary axes")
            n0 = angle_atoms[a180[0]][0]
            n1 = angle_atoms[a180[1]][0]
            print("    angle betwen axis defining atoms is %10.5f" % (vec.angle(span[n0],span[n1])/deg2rad))
            main_axis = vec.normalize(span[n0])
            secd_axis = vec.normalize(span[n1] - (vec.project(span[n1],main_axis)))
        else:
            print("    not a a regular octahedron ... dont know what to do!!")
            print(angles)
    elif cn==12 :
        print("\n&&& CN 12 : assuming a cuboctahedral geometry")
        # take an arbitrary vertex as main axis
        main_axis = vec.normalize(span[0])
        # the trick is to use one of the vertices connected to the primary 
        # in a square face. this should be already close to 90.0
        angles_sort = num.argsort(angles[0:11])
        # sanity test
        print("    sanity testing first four angles connected to the primary")
        print("    should be close to 60 degrees")
        for i in range(4):
            print("angle %3d is %10.5f" % (i+1, angles[angles_sort[i]]))
        vertices = angle_atoms[angles_sort[4]]
        other_vert = vertices[1]
        if other_vert == 0: other_vert = vertices[0]
        print("using vertex %3d with angle %10.5f as second axis" % (other_vert, angles[angles_sort[4]]))
        secd_axis = vec.normalize(span[other_vert]-(vec.project(span[other_vert], main_axis)))
    else:
        print("currently unsupported CN ... code it!!!!!!")
    return main_axis, secd_axis
