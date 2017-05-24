__author__ = 'aishwarya'

import copy, numpy 
from utils import *

# NOTE: All probabilities are in log space

class Partition:

    def __init__(self, possible_goals=None, possible_param_values=None, belief=float('-inf')):
        # This partition includes all belief states which are a tuple 
        # of one action in this possible_goals and one value of each 
        # param in possible_param_values which is a dict
        self.possible_goals = possible_goals
        self.possible_param_values = possible_param_values
        self.belief = belief

    def __str__(self):
        strForm = 'States: {' + ','.join(self.possible_goals) + '}\n'
        for param_name in self.possible_param_values :
            strParamValues = [str(v) for v in self.possible_param_values[param_name]]
            strForm = strForm + param_name + ': {' + ','.join(strParamValues) + '}\n'
        strForm = strForm + 'Belief:' + str(numpy.exp(self.belief)) + '\n'
        return strForm
        
    def __hash__(self):
        dict_tuple = ()
        if self.possible_param_values is not None :
            dict_tuple = tuple([(k, tuple(v)) for (k, v) in self.possible_param_values.items()])
        return hash((tuple(self.possible_goals), dict_tuple, self.belief))
        
    # Assumes elements of possible_goals and possible_param_values[key] for all keys
    # are atomic
    def __eq__(self, other) :
        if not checkLists(self.possible_goals, other.possible_goals) :
            return False
        elif not checkDicts(self.possible_param_values, other.possible_param_values) :
            return False
        elif not self.belief == other.belief :
            return False
        return True     
        
    # A partition matches with a system action and utterance if, for each 
    # attribute specified by them, including the action and its parameters,  
    # the partition allows only that value
    # NOTE : Partition.match() is used to determine whether or not to 
    # split the partition. Hence, for a confirm, whether it is affirmed
    # or denied, it is a match only if the partition exactly allows what
    # is mentioned in the system action
    # V.IMP. NOTE: Do not use this to determine whether to boost the 
    # probability of the (partition, utterance) pair
    def match(self, system_action, utterance) :
        # Partition should match info in system action
        if not len(self.possible_goals) == 1 or not system_action.referring_goal == self.possible_goals[0] :
            return False          
        elif not system_action.referring_params == None :
            for param_name in system_action.referring_params :
                if param_name not in self.possible_param_values or len(self.possible_param_values[param_name]) != 1 or self.possible_param_values[param_name][0] != system_action.referring_params[param_name] :
                    return False
            
        # Partition has to match info in utterance
        if utterance.referring_goal != None :
            if len(self.possible_goals) != 1 or self.possible_goals[0] != utterance.referring_goal :
                return False 
        if utterance.referring_params != None :
            for param_name in utterance.referring_params :
                if param_name not in self.possible_param_values or len(self.possible_param_values[param_name]) != 1 or self.possible_param_values[param_name][0] != utterance.referring_params[param_name] :
                    return False           
        return True
    
    def is_terminal(self, knowledge) :
        if self.possible_goals is None or len(self.possible_goals) != 1 :
            return False
        goal = self.possible_goals[0]
        param_order = knowledge.param_order[goal]
        for param in param_order :
            if param not in self.possible_param_values :
                print 'Warning: All partitions must have at least one value for every param but', param, 'not present here'
                return False
            elif len(self.possible_param_values[param]) != 1 :
                return False
        return True
        
    def is_like(self, other) :
        if not checkLists(self.possible_goals, other.possible_goals) :
            return False
        elif not checkDicts(self.possible_param_values, other.possible_param_values) :
            return False
        return True     
                
    def is_valid(self, knowledge, grounder) :
        # Basic sanity check - alteast one goal and all goals are valid
        if self.possible_goals is None or len(self.possible_goals) < 1 :
            return False
        for goal in self.possible_goals :
            if goal not in knowledge.goal_actions :
                return False   
            
        for param_name in self.possible_param_values :
            if len(self.possible_param_values[param_name]) < 1 :
                return False
        
        # If there is only one goal. Check that all 
        # its params have values which are valid
        if len(self.possible_goals) == 1 :
            goal = self.possible_goals[0]
            relevant_params = knowledge.param_order[goal]
            for param_name in relevant_params :
                if param_name not in self.possible_param_values or len(self.possible_param_values[param_name]) < 1 :
                    return False
                if len(self.possible_param_values[param_name]) == 1 :
                    value = self.possible_param_values[param_name][0]
                    if not self.param_value_valid(goal, param_name, value, knowledge, grounder) :
                        return False
            return True
            
        # If control reaches here, there is more than one goal. Check if
        # any param value is fixed. If so, some goal should be compatible
        # with it
        for param_name in self.possible_param_values :
            if len(self.possible_param_values[param_name]) == 1 :            
                # Parameter param_name has exactly one value
                param_value = self.possible_param_values[param_name][0]
                compatible_goals = list()
                for goal in self.possible_goals :
                    # No goal should be incompatible with it
                    if self.param_value_valid(goal, param_name, param_value, knowledge, grounder) :
                        compatible_goals.append(goal)
                if len(compatible_goals) == 0 :
                    return False
        return True
    
    def param_value_valid(self, goal, param_name, value, knowledge, grounder) :
        if param_name not in knowledge.param_order[goal] :
            return True
        if value is None :
            return False
        for true_pred in knowledge.true_constraints[goal][param_name] :
            if not predicate_holds(true_pred, value, grounder) :
                # A predicate that should hold does not
                return False
        for false_pred in knowledge.false_constraints[goal][param_name] :
            if predicate_holds(false_pred, value, grounder) :
                # A predicate that should not hold does
                return False
        return True
    
    def remove_invalid_goals(self, knowledge, grounder) :
        #print 'In remove_invalid_goals with ', str(self)
        
        if len(self.possible_goals) <= 1 :
            # Better to have one wrong goal than a malformed partition 
            # object so return
            return
            
        for param_name in self.possible_param_values :
            if len(self.possible_param_values[param_name]) == 1 :
                value = self.possible_param_values[param_name][0]
                # This partition is certain that 'param_name' has the 
                # value 'value'
                goals_to_remove = list()
                for goal in self.possible_goals :
                    if value is not None and param_name not in knowledge.param_order[goal] :
                        #print 'Removing ', goal, ' because ', param_name, ' = ', value 
                        # This goal doesn't need param name. We don't 
                        # really need to removing it but pruning always 
                        # results in speedup
                        goals_to_remove.append(goal)
                    if not self.param_value_valid(goal, param_name, value, knowledge, grounder) :
                        # This goal is actually incompatible with the 
                        # param value
                        goals_to_remove.append(goal)
                for goal in goals_to_remove :
                    if len(self.possible_goals) > 1 :
                        # Don't remove the goal if there is only one 
                        # because having a logically invalid goal is less
                        # problematic than having an ill-formed one. It
                        # will get removed elsewhere anyway
                        self.possible_goals.remove(goal)
                
    
    def remove_invalid_params(self, knowledge, grounder) :
        #print 'In remove invalid params with '
        #print str(self)
        if len(self.possible_goals) != 1 :
            #print 'no change'
            return 
        goal = self.possible_goals[0]
        for param_name in self.possible_param_values :
            if param_name in knowledge.param_order[goal] :
                current_values = self.possible_param_values[param_name]
                new_values = list()
                for value in current_values :
                    if self.param_value_valid(goal, param_name, value, knowledge, grounder) :
                        new_values.append(value)
                if len(new_values) > 0 :
                    self.possible_param_values[param_name] = new_values
                else :
                    self.possible_param_values[param_name] = [None]
            else :
                self.possible_param_values[param_name] = [None]
        #print 'End of remove invalid params with '
        #print str(self)
        #print '--------------------------------------------'
    
    # Check whether this partition contains all states having some goal 
    # and params        
    def is_superset(self, required_goal, required_params) :
        #print '^^^^^^^^^^^^^^^^^^^^^'
        #print str(self)
        if required_goal != None and required_goal not in self.possible_goals :
            #print "Goal not present"
            #print '^^^^^^^^^^^^^^^^^^^^^'
            # We want a specific goal and that is not in this partition
            return False
        elif required_params != None :
            for param_name in required_params :
                if param_name not in self.possible_param_values :
                    #print param_name, " not present in call to is_superset"
                    # This should not ideally ever happen
                    #print "\nCheck partitions.py for this line as this should not happen\n"
                    #print '^^^^^^^^^^^^^^^^^^^^^'
                    return False
                else :
                    if not set(required_params[param_name]).issubset(set(self.possible_param_values[param_name])) :
                        #print param_name, " values not a subset"
                        #print "Available values: ", self.possible_param_values[param_name]
                        #print "Queried value: ", required_params[param_name]
                        #print '^^^^^^^^^^^^^^^^^^^^^'
                        # The values we want for this param are not a subset 
                        # of the values allowed by the partition
                        return False
        #print 'Superset!'
        #print '^^^^^^^^^^^^^^^^^^^^^'
        return True
    
    # Returns true if either the goal or some param has only one possible 
    # value
    def something_certain(self) :
        if len(self.possible_goals) == 1 :
            return True
        for param_name in self.possible_param_values :
            if len(self.possible_param_values[param_name]) == 1 :
                return True
        return False
    
    # Check whether this partition has only the goal and param values indicated
    def is_equal(self, required_goal, required_params) :
        #print 'required_params = ', str(required_params)
        #print 'Partition.possible_params: ', str(self.possible_param_values)
        if required_goal != None : # We want a specific goal
            if required_goal not in self.possible_goals : 
                # Required goal is not in this partition
                return False
            elif len(self.possible_goals) > 1 :
                # Required goal is not the only goal in this partition
                return False
        if required_params != None :
            # We have some specified params
            for param_name in required_params :
                if param_name not in self.possible_param_values :
                    print param_name, " not present in call to is_equal"
                    # This should not ideally ever happen
                    print "\nCheck partitions.py for this line as this should not happen\n"
                    return False
                else :
                    #print 'set(required_params[', param_name, ']) = ', set(required_params[param_name])
                    #print 'set(self.possible_param_values[', param_name, ']) = ', set(self.possible_param_values[param_name])
                    if not set(required_params[param_name]) == set(self.possible_param_values[param_name]) :
                        # The values we want for this param are the same  
                        # as the values allowed by the partition
                        return False
        return True
            
    def split_by_goal(self, new_goal, knowledge, grounder) :
        #print 'In split_by_goal'
        if new_goal not in self.possible_goals or len(self.possible_goals) == 1 :
            return [self]
        else :
            p1 = Partition([new_goal], copy.deepcopy(self.possible_param_values))
            split_prob = knowledge.partition_split_goal_probs[new_goal]
            p1.belief = self.belief + numpy.log(split_prob)
            p1.remove_invalid_params(knowledge, grounder)    
            #print 'p1 - ', str(p1)
                
            other_goals = [goal for goal in self.possible_goals if goal != new_goal]
            p2 = Partition(other_goals, copy.deepcopy(self.possible_param_values))
            p2.belief = self.belief + numpy.log(1 - split_prob)
            p2.remove_invalid_params(knowledge, grounder)
            #print 'p2 - ', str(p2)
            
            # Check that both partitions are valid. If not, return only 
            # the one which is    
            if not p1.is_valid(knowledge, grounder) :
                p2.belief = self.belief
                #print 'p1 is not valid'
                return [p2]
            elif not p2.is_valid(knowledge, grounder) :
                p1.belief = self.belief
                #print 'p2 is not valid'
                return [p1]
            else :
                return [p1, p2]

    def split_by_param(self, split_param_name, split_param_value, knowledge, grounder) :
        if self.possible_param_values == None or \
        split_param_name not in self.possible_param_values or \
        split_param_value not in self.possible_param_values[split_param_name] or \
        len(self.possible_param_values[split_param_name]) == 1 :
            return [self]
        else :
            p1_param_values = dict()
            p2_param_values = dict()
            for param_name in self.possible_param_values :
                if param_name != split_param_name :
                    p1_param_values[param_name] = self.possible_param_values[param_name]
                    p2_param_values[param_name] = self.possible_param_values[param_name]
                else :
                    p1_param_values[param_name] = [split_param_value]
                    p2_param_values[param_name] = [param_value for param_value in self.possible_param_values[split_param_name] if param_value != split_param_value]
            p1 = Partition(copy.deepcopy(self.possible_goals), p1_param_values)
            p2 = Partition(copy.deepcopy(self.possible_goals), p2_param_values)

            split_prob = knowledge.partition_split_param_probs[split_param_name][split_param_value]
            p1.belief = self.belief + numpy.log(split_prob)
            p2.belief = self.belief + numpy.log(1 - split_prob)
                
            p1.remove_invalid_goals(knowledge, grounder)
            p2.remove_invalid_goals(knowledge, grounder)
            
            # Check that both partitions are valid. If not, return only 
            # the one which is    
            if not p1.is_valid(knowledge, grounder) :
                p2.belief = self.belief
                return [p2]
            elif not p2.is_valid(knowledge, grounder) :
                p1.belief = self.belief
                return [p1]
            else :
                return [p1, p2]
            
