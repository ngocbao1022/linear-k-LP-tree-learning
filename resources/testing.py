import numpy as np
from resources.tree_classes import kLPtree_classic,kLPtree_total
import time


import itertools as it
def permut(t, origin_attr = (0,1,2,3)):
    # generate permutation
    attrs = list(origin_attr)
    permut = it.permutations(attrs)
    permut_list = list(permut)
    # replace each char by their permutation
    result = []
    for new_permut in permut_list:
        new_t = []
        for x in t:
            noeud = []
            for i in range(len(x)):
                if x[i] == attrs[0]: noeud += [new_permut[0]]
                elif x[i] == attrs[1]: noeud += [new_permut[1]]
                elif x[i] == attrs[2]: noeud += [new_permut[2]]
                else: noeud += [new_permut[3]]
            new_t += [tuple(sorted(noeud))]
        result += [new_t]
    # remove duplicates
    new_array = [tuple(row) for row in result]
    uniques = np.unique(new_array,axis=0)
    return list(uniques)

def insert_new_to_tree_no_CP(tree, new, k):
    inserted_list = [[new]+tree]
    for i in range(len(tree)):
        inserted_list += [tree[0:i+1]+[new]+tree[i+1:]]
        if len(tree[i])<k : 
            noeud = tuple(sorted(list(tree[i])+list(new)))
            inserted_list += [tree[0:i]+[noeud]+tree[i+1:]]
    return inserted_list
  
def all_tree_no_CP(n = 4, k = 3):
    tree_list = [[]]
    for i in range(n):
        new_tree_list = []
        for t in tree_list:
            new_tree_list += insert_new_to_tree_no_CP(t, (i,), k)
        tree_list = new_tree_list
    return tree_list

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



def generate_sample_with_counting_time(ranks,N=2000,p=0.3):
    n = len(ranks)
    rank_sample = np.random.geometric(p, N)
    reroll = rank_sample[rank_sample>n]
    N_r = len(reroll)
    while N_r!=0:
        rank_sample[rank_sample>n] = np.random.geometric(p, N_r)
        reroll = rank_sample[rank_sample>n]
        N_r = len(reroll)
    rank_sample -= 1
    start = time.time()
    Rs,count = np.unique(rank_sample, return_counts = True)
    count_adapted_to_rank = np.zeros((n))
    Rr,inverse_index = np.unique(ranks, return_index=True)
    for i in range(len(Rs)):
        count_adapted_to_rank[inverse_index[int(Rs[i])]] = count[i]
    ti = time.time()-start 
    return count_adapted_to_rank,ti

def generate_sample(ranks,N=2000,p=0.3):
    n = len(ranks)
    rank_sample = np.random.geometric(p, N)
    reroll = rank_sample[rank_sample>n]
    N_r = len(reroll)
    while N_r!=0:
        rank_sample[rank_sample>n] = np.random.geometric(p, N_r)
        reroll = rank_sample[rank_sample>n]
        N_r = len(reroll)
    rank_sample -= 1
    Rs,count = np.unique(rank_sample, return_counts = True)
    count_adapted_to_rank = np.zeros((n))
    Rr,inverse_index = np.unique(ranks, return_index=True)
    for i in range(len(Rs)):
        count_adapted_to_rank[inverse_index[int(Rs[i])]] = count[i]
    return count_adapted_to_rank

def rloss_montecarlo(T_learned,T_origin,sample_size=3000,p=0.3,MCsize = 100000):
    ranks = T_origin.item_ranks_from_tree
    mean_rank_origin = []
    mean_rank_learned = []
    for i in range(MCsize):
        sample = generate_sample(ranks,sample_size,p)
        mean_rank_learned += [T_learned.mean_empirical_rank(sample)]
        mean_rank_origin  += [T_origin.mean_empirical_rank(sample)]
    return np.mean(mean_rank_learned)-np.mean(mean_rank_origin)

def rloss_montecarlo_parallel(T_learned_1,T_learned_2,T_origin,sample_size=3000,p=0.3,MCsize = 100000):
    ranks = T_origin.item_ranks_from_tree
    mean_rank_origin = []
    mean_rank_learned_1 = []
    mean_rank_learned_2 = []
    for i in range(MCsize):
        sample = generate_sample(ranks,sample_size,p)
        mean_rank_learned_1 += [T_learned_1.mean_empirical_rank(sample)]
        mean_rank_learned_2 += [T_learned_2.mean_empirical_rank(sample)]
        mean_rank_origin  += [T_origin.mean_empirical_rank(sample)]
    return np.mean(mean_rank_learned_1)-np.mean(mean_rank_origin),np.mean(mean_rank_learned_2)-np.mean(mean_rank_origin)
    
    
# generate random tree
def generate_kLPtree_classic(n,k):
    attrs = np.arange(n)
    attrs = np.random.permutation(attrs)
    tree = []
    PT = []
    i = 0
    while i < n:
        noeud_size = min(n-i,np.random.randint(k)+1)
        tree += [tuple(np.sort(attrs[i:i+noeud_size]))]
        PT += [np.random.permutation(np.arange(2**noeud_size))]
        i = i+noeud_size
    return tree, PT
def generate_kLPtree_total(n,noeuddict):
    keys = list(noeuddict.keys())
    tree_origin = np.random.permutation(keys)
    tree = []
    attrs = []
    PT = []
    i = 0
    while len(attrs)!=n:
        diff = np.setdiff1d(np.array(tree_origin[i]),np.array(attrs))
        if len(diff) != 0:
            tree += [tree_origin[i]]
            PT += [np.random.permutation(np.arange(2**len(tree_origin[i])))]
            attrs += list(diff)
        i += 1
    return tree, PT
def random_sample_from_tree(tree, N = 10000, p = 0.3):
    ranks = tree.item_ranks_from_tree
    sample = generate_sample(ranks, N, p)
    return sample
def test_random_sample_from_random_simulated_tree(N=10000, p=0.3):
    tree, PT = generate_kLPtree_classic(4,3)
    tree_obj = kLPtree_classic([],tree)
    tree_obj.modify_PT(PT)
    print(tree_obj.item_ranks_from_tree)
    print(random_sample_from_tree(tree_obj,N,p))
def learn_from_random_tree(n, k, noeuddict, obj_list_bin, sample_size = 3000, p=0.3):

    origin_tree, PT = generate_kLPtree_classic(n, k)
    origin_tree_obj = kLPtree_classic([],origin_tree)
    origin_tree_obj.modify_PT(PT)

    sample = random_sample_from_tree(origin_tree_obj, sample_size, p)
    learned_tree_1 = best_tree_no_rep_v2(sample, num_of_attrs = n)
    learned_tree_2 = best_tree_no_rep(sample, num_of_attrs = n)
    learned_tree_obj_1 = kLPtree_classic(sample,learned_tree_1)
    learned_tree_obj_2 = kLPtree_classic(sample,learned_tree_2)
    

    rloss_1 = rloss_montecarlo(learned_tree_obj_1, origin_tree_obj, sample_size=sample_size, p=0.3, MCsize=100000)

    rloss_2 = rloss_montecarlo(learned_tree_obj_2, origin_tree_obj, sample_size=sample_size, p=0.3, MCsize=100000)
    return [rloss_1,rloss_2]

def learn_from_random_tree_parallel(n, k,  noeuddict, obj_list_bin, sample_size = 3000, p=0.3):

    origin_tree_1, PT_1 = generate_kLPtree_classic(n, k)
    origin_tree_obj_1 = kLPtree_classic([],origin_tree_1)
    origin_tree_obj_1.modify_PT(PT_1)
    
    origin_tree_2, PT_2 = generate_kLPtree_total()
    origin_tree_obj_2 = kLPtree_total([],origin_tree_2)
    origin_tree_obj_2.modify_PT(PT_2)

    sample12 = random_sample_from_tree(origin_tree_obj_1, sample_size, p)
    sample34 = random_sample_from_tree(origin_tree_obj_2, sample_size, p)
    
    learned_tree_1 = best_tree_no_rep_v2(sample12, num_of_attrs = n)
    learned_tree_2 = best_tree_no_rep(sample12)
    learned_tree_3 = best_tree(sample34)
    learned_tree_4 = best_tree_v2(sample34,n_attrs = n)
    learned_tree_obj_1 = kLPtree_classic(sample12,learned_tree_1)
    learned_tree_obj_2 = kLPtree_classic(sample12,learned_tree_2)
    learned_tree_obj_3 = kLPtree_total(sample34,learned_tree_3)
    learned_tree_obj_4 = kLPtree_total(sample34,learned_tree_4)
    
    rloss_1,rloss_2 = rloss_montecarlo_parallel(learned_tree_obj_1,learned_tree_obj_2, origin_tree_obj_1, sample_size=sample_size, p=0.3, MCsize=2000)
    rloss_3,rloss_4 = rloss_montecarlo_parallel(learned_tree_obj_3, learned_tree_obj_4, origin_tree_obj_2, sample_size=sample_size, p=0.3, MCsize=2000)
    return [rloss_1,rloss_2,rloss_3,rloss_4]

def learning_time(sample_size,n,k,noeuddict):
    ranks = np.random.permutation(range(2**n))
    sample, ti = generate_sample_with_counting_time(ranks)

    start_time = time.time()
    a = best_tree_no_rep(sample, noeuddict)
    a_obj = kLPtree_classic(sample, a)
    exe_time_1 = time.time()-start_time+ti

    start_time = time.time()
    b = best_tree_no_rep_v2(sample, noeuddict, n)
    b_obj = kLPtree_classic(sample,b)
    exe_time_2 = time.time()-start_time+ti

    start_time = time.time()
    c = best_tree(sample, noeuddict)
    c_obj = kLPtree_total(sample,c)
    exe_time_3 = time.time()-start_time+ti

    start_time = time.time()
    d = best_tree(sample, noeuddict)
    d_obj = kLPtree_total(sample,d)
    exe_time_4 = time.time()-start_time+ti
    
    return exe_time_1,exe_time_2,exe_time_3,exe_time_4
    
