import numpy as np


class kLPtree_classic:
    def __init__(self, H, noeudnamelist, num_of_attrs, noeuddict, obj_list_bin):
        if len(H)!=0:
            self.noeud_dict = noeuddict
            self.noeud_names = noeudnamelist
            self.noeud_list = [noeuddict[x] for x in noeudnamelist]
            self.preference_count = np.array([np.array([np.sum(H[self.noeud_list[j][i]]) for i in range(len(self.noeud_list[j]))]) for j in range(len(self.noeud_list))], dtype=object)
            self.preference_table = np.array([np.flip(np.argsort(self.preference_count[j])) for j in range(len(noeudnamelist))], dtype=object)
            self.obj_list = obj_list_bin
            self.n = len(noeudnamelist)
            self.num_attrs = num_of_attrs
            self.rank_from_tree()
        else:
            self.noeud_dict = noeuddict
            self.noeud_names = noeudnamelist
            self.noeud_list = [noeuddict[x] for x in noeudnamelist]
            self.obj_list = obj_list_bin
            self.n = len(noeudnamelist)
            self.num_attrs = num_of_attrs

    def modify_PT(self, new_PT):
        self.preference_table = new_PT
        self.rank_from_tree()

    def show_count(self):
        print(self.preference_count)

    def show_PT(self):
        print(self.preference_table)

    def rank(self, obj): 
        notfound = True
        i = 0
        rank = 0
        local_rank_space = 2**self.num_attrs
        for i in range(len(self.noeud_names)):
            attrs = [int(x) for x in self.noeud_names[i]]
            value = [self.obj_list[obj][i] for i in attrs]
            index = self.valuelist_to_index(value)
            x = np.argwhere(self.preference_table[i]==index)
            local_rank_space /= 2**len(attrs)
            rank += x*local_rank_space
        return rank[0][0]

    def valuelist_to_index(self, vlist):
        n = len(vlist)
        base = [2**i for i in range(n)]
        base = np.flip(base)
        return np.sum(base*np.array(vlist))

    def rank_from_tree(self):
        rank = []
        for i in range(2**self.num_attrs):
            rank += [self.rank(i)]
        self.item_ranks_from_tree =  np.array(rank)

    def mean_empirical_rank(self, H):
        rank = self.item_ranks_from_tree
        S = np.sum(H)
        R_mean = np.sum(np.array(H)*rank/S)
        return R_mean

class kLPtree_total:
    def __init__(self, H, noeudnamelist, num_of_attrs, noeuddict, obj_list_bin):
        if len(H)!=0:
            self.noeud_dict = noeuddict
            self.noeud_names = noeudnamelist
            self.noeud_list = [noeuddict[x] for x in noeudnamelist]
            self.preference_count = np.array([np.array([np.sum(H[self.noeud_list[j][i]]) for i in range(len(self.noeud_list[j]))]) for j in range(len(self.noeud_list))], dtype=object)
            self.preference_table = np.array([np.flip(np.argsort(self.preference_count[j])) for j in range(len(noeudnamelist))], dtype=object)
            self.obj_list = obj_list_bin
            self.n = len(noeudnamelist)
            self.num_attrs = num_of_attrs
            self.rank_from_tree()
        else:
            self.noeud_dict = noeuddict
            self.noeud_names = noeudnamelist
            self.noeud_list = [noeuddict[x] for x in noeudnamelist]
            self.obj_list = obj_list_bin
            self.n = len(noeudnamelist)
            self.num_attrs = num_of_attrs

    def modify_PT(self, new_PT):
        self.preference_table = new_PT
        self.rank_from_tree()

    def show_count(self):
        print(self.preference_count)

    def show_PT(self):
        print(self.preference_table)

    def valuelist_to_index(self, vlist):
        n = len(vlist)
        base = [2**i for i in range(n)]
        base = np.flip(base)
        return np.sum(base*np.array(vlist))

    def index_to_vlist(self, n, index): # n le nombre d'attributs dans ce noeud
        ind = index
        value = []
        for i in range(n):
            k = ind//2
            value += [ind - 2*k]
            ind = k
        value = np.flip(value)
        return value

    def rank(self, obj): 
        notfound = True
        i = 0
        attrs_set = []
        values_set = []
        rank = 0
        local_rank_space = 2**self.num_attrs
        while notfound:
            new_attrs = np.setdiff1d(np.array([int(x) for x in self.noeud_names[i]]), np.array(attrs_set))
            if len(new_attrs) == len(self.noeud_names[i]):
                attrs = [int(x) for x in self.noeud_names[i]]
                value = [self.obj_list[obj][i] for i in attrs]
                index = self.valuelist_to_index(value)
                x = np.argwhere(self.preference_table[i]==index)
                local_rank_space /= 2**len(attrs)
                values_set += value
                attrs_set += attrs
            elif len(new_attrs) == 0:
                x = 0
            else:
                attrs_reduce, new_PT = self.reduce_noeud(i, values_set, attrs_set)               
                value = [self.obj_list[obj][i] for i in attrs_reduce]
                index = self.valuelist_to_index(value)
                x = np.argwhere(new_PT==index)
                local_rank_space /= len(new_PT)
                values_set += list(value)
                attrs_set += list(attrs_reduce)
            rank += x*local_rank_space
            i += 1
            notfound = (len(attrs_set) != self.num_attrs)
        return rank[0][0]

    def rank_binary_to_normal(self, rank_bin):
        n = len(rank_bin)
        rank = 0
        for i in range(n):
            rank = 2*rank + rank_bin[i]
        return rank
        
    def PTbin_to_normal(self, PT_bin):
        new_PT = []
        for r in PT_bin:
            new_PT += [self.rank_binary_to_normal(r)]
        sort = sorted(new_PT)
        return [sort.index(x) for x in new_PT]


    def reduce_noeud(self, i, setvalues, setnames):
        noeudPT = self.preference_table[i]
        noeudname = self.noeud_names[i]
        # indexing structure
        attrs = np.array([int(x) for x in list(noeudname)])
        attrs_set_origin = np.array([int(x) for x in list(setnames)])
        attrs_set = np.intersect1d(attrs_set_origin,attrs)
        attrs_set_index = np.array([np.where(attrs == attrs_set[i])[0][0] for i in range(len(attrs_set))])
        attrs_set_values = np.array([setvalues[np.where(attrs_set_origin==attrs_set[i])[0][0]] for i in range(len(attrs_set))])
        attrs_reduce = np.setdiff1d(attrs,attrs_set)
        PT_bin = []
        for index in noeudPT:
            value = self.index_to_vlist(len(attrs), index)
            PT_bin = PT_bin + [value]
        new_PT = np.array(PT_bin)
        for i in range(len(attrs_set)):
            new_PT = new_PT[new_PT[:,attrs_set_index[i]]==attrs_set_values[i]]
        new_PT = self.PTbin_to_normal(new_PT)
        return attrs_reduce, new_PT

    def test_reduce_noeud(self):
        attrs_reduce, new_PT = self.reduce_noeud(1, np.array([1]),"0")
        return new_PT

    def rank_from_sample(self, H):
        sort = sorted(list(H))
        rank_H = [sort.index(x) for x in H]
        return rank_H

    def rank_from_tree(self):
        rank = []
        for i in range(2**self.num_attrs):
            rank += [self.rank(i)]
        self.item_ranks_from_tree = np.array(rank)

    def mean_empirical_rank(self, H):
        rank = self.item_ranks_from_tree
        S = np.sum(H)
        R_mean = np.sum(np.array(H)*rank/S)
        return R_mean