__author__ = 'aishwarya'

import os
import pickle
import numpy

# For typechecking
from SemanticNode import SemanticNode


#models_path = '/v/filer4b/v20q001/aish/Documents/Research/Code/catkin_ws/src/nlu_pipeline/src/models/'
models_path = '/v/filer4b/v20q001/aish/Documents/Research/Code/catkin_ws/src/dialog_active_learning/src/models/'


def save_model(obj, name):
    with open(models_path + str(name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_obj_general(obj, name):
    print 'Saving log'
    with open(str(name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load_obj_general(name):
    try:
        f = open(name, 'rb')
        return pickle.load(f)
    except IOError:
        return None


def load_model(name, path=models_path):
    try:
        print os.path.join(path, str(name) + '.pkl')  # DEBUG
        with open(os.path.join(path, str(name) + '.pkl'), 'rb') as f:
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


def predicate_holds(predicate, argument, grounder):
    if argument is None:
        return False
        
    # Special case - things starting with "item_" are items
    if argument.startswith('item_') and predicate == 'item' :
        return True
        
    return grounder.predicate_holds(predicate, argument)


def get_dict_val(dict_name, key):
    if dict_name is None:
        return None
    elif key not in dict_name:
        return None
    else:
        return dict_name[key]
