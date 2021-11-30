import numpy as np
from resources.initial import score,score_conditional,objlist_to_binary
from resources.tree_classes import *

def remove_all_noeuds_contain(x, noeud_list):
    new_list = []
    for N in noeud_list:
        not_in = True 
        for e in x:
            not_in *= (list(N).count(e)==0)
        if not_in: new_list += [N]
    return new_list

def recursive_no_CP_respect_order(sequence, noeud_list):
    # case 1 : x = [a] => trees = [[a]]
    # case 2 : x = [ab, a, b] => trees = [[ab], [a,b]]
    # case 3 : for noeud in list : chon vi tri bat dau va right sequence, lua chon thuat toan de cut list
    sequence_list = []
    if len(noeud_list)==0:
        sequence_list = [sequence]
    elif len(noeud_list)==1:
        sequence_list = [sequence + noeud_list]
    else:
        for i in range(len(noeud_list)):
            base_sequence = sequence + [noeud_list[i]]
            sequence_list += recursive_no_CP_respect_order(base_sequence, remove_all_noeuds_contain(noeud_list[i],noeud_list[i+1:]))
    return sequence_list

def eligible_trees(tree_list, num_of_attrs):
    new_tree_list = []
    for tree in tree_list:
        attrs = set({})
        for N in tree:
            attrs = attrs|set(N)
        if len(attrs)==num_of_attrs:
            new_tree_list += [tree]
    return new_tree_list

def all_trees_no_CP_respect_order(noeud_list, num_of_attrs):
    tree_list = recursive_no_CP_respect_order([],noeud_list)
    return eligible_trees(tree_list, num_of_attrs)

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
def best_tree_no_rep_v2(H, noeuddict, num_of_attrs):
    obj_list_bin = objlist_to_binary(num_of_attrs)
    s_list = []
    for x in noeuddict.values():
        s_list += [score(H,x)]
    sort = np.argsort(s_list)
    noeud_list = []
    keys = list(noeuddict.keys())
    for i in sort:
        noeud_list += [keys[i]]
    tree_list = all_trees_no_CP_respect_order(noeud_list,num_of_attrs)
    mean_rank_list = []
    for tree in tree_list:
        tree_obj = kLPtree_classic(H,tree,num_of_attrs = num_of_attrs, noeuddict = noeuddict, obj_list_bin = obj_list_bin)
        mean_rank_list += [tree_obj.mean_empirical_rank(H)]
    arg = np.argmin(mean_rank_list)
    return tree_list[arg]