__author__ = 'aishwarya'

import os
import pickle
import numpy


# Expects name to be a valid relative or absolute path
def save_obj(obj, name):
    print 'Saving log'
    with open(str(name), 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()

# Expects name to be a valid relative or absolute path
def load_obj_general(name):
    try:
        f = open(name, 'r')
        return pickle.load(f)
    except IOError:
        return None


# Source: https://en.wikipedia.org/wiki/Log_probability
def add_log_probs(logprob1, logprob2):
    if logprob1 == float('-inf') and logprob2 == float('-inf'):
        return float('-inf')
    if logprob2 > logprob1:
        temp = logprob1
        logprob1 = logprob2
        logprob2 = temp 
    res = logprob1 + numpy.log1p(numpy.exp(logprob2 - logprob1))
    return res
    
        
def checkLists(list1, list2):
    if list1 is None and list2 is None:
        return True
    elif list1 is None and list2 is not None:
        return False
    elif list1 is not None and list2 is None:
        return False
    else:
        if type(list1) != list or type(list2) != list:
            return False
        elif set(list1) == set(list2):
            return True
        else:
            return False


def checkDicts(dict1, dict2):
    if dict1 is None and dict2 is None:
        return True
    elif dict1 is None and dict2 is not None:
        return False
    elif dict1 is not None and dict2 is None:
        return False
    else:
        if type(dict1) != dict or type(dict2) != dict:
            return False
        if set(dict1.keys()) != set(dict2.keys()):
            return False   
        else:
            for key in dict1.keys():
                if type(dict1[key]) == list:
                    if not checkLists(dict1[key], dict2[key]):
                        return False  
                elif not dict1[key] == dict2[key]:
                    return False
    return True


def arg_max(d):
    lmax = None
    larg_max = None
    for k in d:
        if lmax is None or lmax < d[k]:
            lmax = d[k]
            larg_max = k
    return larg_max, lmax


def get_dict_val(dict_name, key):
    if dict_name is None:
        return None
    elif key not in dict_name:
        return None
    else:
        return dict_name[key]
