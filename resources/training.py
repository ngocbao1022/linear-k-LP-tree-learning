import numpy as np
from resources.testing import all_trees_no_CP_respect_order,eligible_trees,recursive_no_CP_respect_order
from resources.initial import score,score_conditional,objlist_to_binary
from resources.tree_classes import *

# TRAINING - tree from algo
def order(H, noeuddict):
    s_list = []
    for x in noeuddict.values():
        s_list += [score(H,x)]
    sort = np.argsort(s_list)
    tree = []
    keys = list(noeuddict.keys())
    for i in sort:
        tree += [keys[i]]
    return tree

def best_tree(H, noeuddict, n_attrs):
    s_list = []
    for x in noeuddict.values():
        s_list += [score(H,x)]
    sort = np.argsort(s_list)
    keys = list(noeuddict.keys())
    order =[]
    for i in sort:
        order += [keys[i]]
    tree = []
    attrs = []
    i=0
    while len(attrs)!= n_attrs:
        x = np.setdiff1d(list(order[i]),attrs)
        if len(x) != 0:
            tree += [order[i]]
            attrs += list(x)
        i+=1
    return tree

def best_tree_v2(H, noeuddict, n_attrs):
    #build first node
    s_list = []
    keys = list(noeuddict.keys())
    noeuds = list(noeuddict.values())
    for x in noeuds:
        s_list += [score(H,x)]
    i = np.argmin(s_list)
    tree = [keys[i]]
    keys = keys[:i]+keys[i+1:]
    noeuds = noeuds[:i]+noeuds[i+1:]
    attrs = list(tree[0])
    found = (len(attrs)==n_attrs)
    #build conditional node
    while not found:
        s_list = []
        for i in range(len(noeuds)):
            condition = tuple(np.intersect1d(attrs, list(keys[i])))
            if len(condition)==0:
                s_list += [score(H,noeuds[i])]
            elif len(condition)!=len(keys[i]):
                s_list += [score_conditional(H,noeuds[i],noeuddict[condition])]
            else:
                s_list += [float('inf')]
        i = np.argmin(s_list)
        tree += [keys[i]]
        keys = keys[:i]+keys[i+1:]
        noeuds = noeuds[:i]+noeuds[i+1:]
        attrs =  list(np.union1d(attrs,list(tree[-1])))
        found = (len(attrs)==n_attrs)
    return tree


# TRAINING - tree no rep from algo
def best_tree_no_rep(H, noeuddict):
    s_list = []
    for x in noeuddict.values():
        s_list += [score(H,x)]
    sort = np.argsort(s_list)
    tree = []
    keys = list(noeuddict.keys())
    attrs = []
    for i in sort:
        x = np.intersect1d(np.array(attrs),np.array(list(keys[i])))
        if len(x) == 0:
            tree += [keys[i]]
            attrs += list(keys[i])
    return tree

# TRAINING - tree no rep from algo
def best_tree_no_rep_v2(H, noeuddict, n_attrs):
    obj_list_bin = objlist_to_binary(n_attrs)
    s_list = []
    for x in noeuddict.values():
        s_list += [score(H,x)]
    sort = np.argsort(s_list)
    noeud_list = []
    keys = list(noeuddict.keys())
    for i in sort:
        noeud_list += [keys[i]]
    tree_list = all_trees_no_CP_respect_order(noeud_list,n_attrs)
    mean_rank_list = []
    for tree in tree_list:
        tree_obj = kLPtree_classic(H,tree,num_of_attrs = n_attrs, noeuddict = noeuddict, obj_list = obj_list_bin)
        mean_rank_list += [tree_obj.mean_empirical_rank(H)]
    arg = np.argmin(mean_rank_list)
    return tree_list[arg]