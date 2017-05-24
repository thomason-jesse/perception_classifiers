__author__ = 'aishwarya'

import math
from utils import *

class Utterance:

    def __init__(self, action_type, referring_goal=None, 
            referring_params=None, parse_prob=float('-inf')):
        # This gives the type of dialog action
        # Possible types - inform_full, inform_param, affirm, deny
        # In case of affirm and deny, the goal and params will be None
        self.action_type = action_type
        
        # Best guess of goal action indicated
        self.referring_goal = referring_goal   

        # Best guess of params indicated
        self.referring_params = referring_params 
        
        # Probability of the utterance given the most recent observation
        # in log space
        self.parse_prob = parse_prob
        
        self.parse = None
        
    def __str__(self):
        str_form = 'Utterance: '+ str(self.action_type) + '\n'
        str_form = str_form + '\tGoal: ' + str(self.referring_goal) + '\n'
        if self.referring_params is not None :
            str_form = str_form + '\tParams: ' + ','.join([str(k) + ':' + str(v) for (k, v) in self.referring_params.items()]) + '\n'
        else:
            str_form = str_form + '\tParams:None\n'
        str_form = str_form + '\tProb:' + str(math.exp(self.parse_prob)) + '\n'
        return str_form        

    # assumes elements of referring_params list are atomic
    def __eq__(self, other) :
        if not isinstance(other, Utterance) :
            return False
        if self.action_type != other.action_type :
            return False
        if self.referring_goal != other.referring_goal :
            return False
        if not checkDicts(self.referring_params, other.referring_params) :
            return False
        if self.parse_prob != other.parse_prob :
            return False
        return True
        
    def is_like(self, other) :
        if self.action_type != other.action_type :
            return False
        if self.referring_goal != other.referring_goal :
            return False
        if not checkDicts(self.referring_params, other.referring_params) :
            return False
        return True
        
    def __hash__(self):
        hash_tuple = ()
        if self.referring_params is not None :
            hash_tuple = (self.action_type, self.referring_goal, tuple(self.referring_params.items()), self.parse_prob)
        else :
            hash_tuple = (self.action_type, self.referring_goal, (), self.parse_prob)
        return hash(hash_tuple)
    
    # NOTE: Utterance.match() is used to determine whether or not to 
    # boost the probability of the (partition, utterance) hypothesis.
    # Hence for a denial, it looks for partitions that differ in some 
    # way from the system action
    # NOTE THAT THIS FUNCTIONALITY IS DIFFERENT FROM Partition.match()    
    def match(self, partition, system_action) :
        if self.action_type == 'deny' :
            # Match all partitions that in some way do not match the 
            # system action
            match_system_action = True
            if system_action.referring_goal != None and system_action.referring_goal not in partition.possible_goals :
                match_system_action = False
            if system_action.referring_params != None :
                for param_name in system_action.referring_params :
                    if param_name not in partition.possible_param_values or system_action.referring_params[param_name] not in partition.possible_param_values[param_name] :
                        match_system_action = False
            return not match_system_action
            
        if self.action_type == 'affirm' :
            # Match all partitions that have the goal and params which
            # the system action mentions
            if system_action.referring_goal != None and system_action.referring_goal not in partition.possible_goals :
                return False
            if system_action.referring_params != None :
                for param_name in system_action.referring_params :
                    if param_name not in partition.possible_param_values or system_action.referring_params[param_name] not in partition.possible_param_values[param_name] :
                        return False
            return True
            
        # Control comes here if the utterance is information providing (not affirm/deny)
        #print 'Inform utterance'
        if self.referring_goal != None :
            if system_action.referring_goal != None and self.referring_goal != system_action.referring_goal :
                # Normally utterance and system_action will not both have
                # a goal but if they do, they should match
                #print 'Goal does not match in utterance and system action' # DEBUG
                return False
            if partition.possible_goals == None or self.referring_goal not in partition.possible_goals :
                # The partition does not allow the desired goal
                #print 'Partition does not allow the desired goal'  # DEBUG
                return False
        if self.referring_params != None :
            for param_name in self.referring_params :
                if param_name not in partition.possible_param_values or self.referring_params[param_name] not in partition.possible_param_values[param_name] :
                    # Typically all partitions have some set of values 
                    # for all param names. The partition must allow 
                    # the values specified in the utterance
                    #print 'Parition does not allow ', self.referring_params[param_name], ' for ', param_name   # DEBUG
                    return False
        return True
        
    def is_valid(self, knowledge, grounder) :
        #print 'In is_valid'
        # Sanity check - inform_full must have a goal and affirm and deny do not
        if self.action_type not in ['inform_full', 'inform_param', 'affirm', 'deny'] :
            return False
        if self.action_type == 'inform_full' and self.referring_goal is None :
            return False
        if self.action_type == 'affirm' and self.referring_goal is not None :
            return False         
        if self.action_type == 'deny' and self.referring_goal is not None :
            return False
        
        # Check that if a goal is present, it is valid
        goal = self.referring_goal
        if goal is not None and goal not in knowledge.goal_actions :
            return False
        
        # Check that if a relevant param has a value, it is valid
        if goal is not None :
            relevant_params = knowledge.param_order[goal]
            for param_name in relevant_params :
                if param_name in self.referring_params :
                    argument = self.referring_params[param_name]
                    #print 'param_name = ', param_name, ', argument = ', argument 
                    for true_pred in knowledge.true_constraints[goal][param_name] :
                        #print 'Checking predicate ', true_pred, ' for', argument
                        if not predicate_holds(true_pred, argument, grounder) :
                            # A predicate that should hold does not
                            return False
                    for false_pred in knowledge.false_constraints[goal][param_name] :
                        #print 'Checking predicate ', false_pred, ' for', argument
                        if predicate_holds(false_pred, argument, grounder) :
                            # A predicate that should not hold does
                            return False
        return True
        
