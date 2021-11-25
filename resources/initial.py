import numpy as np
import pandas as pd


def binary_decompose(j,n):
    val_dec = j
    val_bin = []
    for i in range(n):
        k = val_dec//2
        val_bin += [val_dec - 2*k]
        val_dec = k
    val_bin = np.flip(val_bin)
    return val_bin
def binary_to_decical(val_bin):
    n = len(val_bin)
    base = [2**i for i in range(n)]
    base = np.flip(base)
    return np.sum(base*np.array(val_bin))
def make_noeuds(n):
    for i in range(n):
        globals()[f"attr_{i}"]= [[],[]]
    for j in range(2**n):
        b = binary_decompose(j,n)
        for i in range(n):
            if b[i]==0:
                globals()[f"attr_{i}"][0] += [j]
            if b[i]==1:
                globals()[f"attr_{i}"][1] += [j]
    noeud_list = []
    name_space = []
    for i in range(n):
        noeud_list += [globals()[f"attr_{i}"]]
        name_space += [(i,)]
    return noeud_list, name_space
def noeud_compose2(X,Y):
    NN = np.intersect1d(X[0],Y[0])
    NY = np.intersect1d(X[0],Y[1])
    YN = np.intersect1d(X[1],Y[0])
    YY = np.intersect1d(X[1],Y[1])
    return np.array([NN,NY,YN,YY])
def noeud_compose3(X,Y,Z):
    XY = noeud_compose2(X,Y)
    NNN =  np.intersect1d(XY[0],Z[0])
    NNY =  np.intersect1d(XY[0],Z[1])
    NYN =  np.intersect1d(XY[1],Z[0])
    NYY =  np.intersect1d(XY[1],Z[1])
    YNN =  np.intersect1d(XY[2],Z[0])
    YNY =  np.intersect1d(XY[2],Z[1])
    YYN =  np.intersect1d(XY[3],Z[0])
    YYY =  np.intersect1d(XY[3],Z[1])
    return np.array([NNN,NNY,NYN,NYY,YNN,YNY,YYN,YYY])
def double_noeud(list_1):
    n = len(list_1)
    noeud_2_list = []
    name_2_space = []
    for i in range(n-1):
        for j in range(i+1,n):
            noeud_2_list += [noeud_compose2(list_1[i],list_1[j])]
            name_2_space += [(i,j)]
    return noeud_2_list, name_2_space
def triple_noeud(list_1):
    n = len(list_1)
    noeud_3_list = []
    name_3_space = []
    for i in range(n-2):
        for j in range(i+1,n-1):
            for k in range(j+1,n):
                noeud_3_list += [noeud_compose3(list_1[i],list_1[j],list_1[k])]
                name_3_space += [(i,j,k)]
    return noeud_3_list, name_3_space
def generate_noeud_dictionary(n):
    noeud_1_list, name_1_list = make_noeuds(n)
    noeud_2_list, name_2_list = double_noeud(noeud_1_list)
    noeud_3_list, name_3_list = triple_noeud(noeud_1_list)
    dictionary = {}
    for i in range(len(name_1_list)):
        dictionary[name_1_list[i]] = noeud_1_list[i]
    for i in range(len(name_2_list)):
        dictionary[name_2_list[i]] = noeud_2_list[i]
    for i in range(len(name_3_list)):
        dictionary[name_3_list[i]] = noeud_3_list[i]
    return dictionary
        
def objlist_to_binary(n):
    obj_list = range(2**n)
    obj_list_bin = []
    for i in obj_list:
        obj_list_bin += [binary_decompose(i,n)]
    return np.array(obj_list_bin)

def score(H, noeud):
    n = len(noeud)
    count = np.array([np.sum(H[noeud[i]]) for i in range(n)])
    sort = np.flip(np.sort(count))
    index = np.arange(n)
    return np.sum(sort*index)/(n-1)

def score_conditional(H, noeud, condition): # notes : set of attrs in condition must bu subset of attrs in noeud
    m = len(condition)
    n = len(noeud)
    index = np.arange(int(n/m))
    count = np.zeros((int(n/m)))
    for c in condition:
        count_c = []
        for x in noeud:
            if len(np.intersect1d(x,c))!=0:
                count_c += [np.sum(H[x])]
        count_c = np.flip(np.sort(count_c))   
        count += count_c
    return np.sum(count*index)/(n/m-1)
    
