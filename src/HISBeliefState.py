# Important usage notes: 
#   1.  learner_update needs to be called with the learner
#       after every question is asked so that it picks a new question
#   2. self.learner_question_asked should be 'yes' or 'no'

__author__ = 'aishwarya'

import itertools, numpy
from Partition import Partition
from utils import *

class HISBeliefState:

    def __init__(self, knowledge):

        self.knowledge = knowledge
        self.num_user_turns = 0

        # Number of possible actions the goal could be referring to
        self.num_goal_actions = len(knowledge.goal_actions) 
        self.num_utterance_actions = self.num_goal_actions + 1 # As action is utterance is allowed to be None  

        # n - Maximum possible number of parameters an action can have
        self.max_goal_params = len(knowledge.goal_params)     

        # Number of possible values each parameter can take
        self.num_goal_param_values = len(knowledge.goal_params_values) 

        # Create a partition with all possible states
        possible_actions = knowledge.goal_actions
        possible_param_values = dict()
        for param_name in knowledge.goal_params :
            possible_param_values[param_name] = knowledge.goal_params_values
        complete_partition = Partition(possible_actions, possible_param_values, 0.0)
        self.partitions = [complete_partition]	
        
        # Hypotheses currently being tracked
        self.hypothesis_beliefs = None
        
        self.learner_question_asked = 'no'  # Has a question ever been asked in 
                                            # the current dialog
                                        
        # Information about the potential next question from the learner
        # Needs to be explicitly set by calling learner_update with a 
        # learner after every turn
        self.learner_current_candidate_idx = None
        self.learner_current_candidate_label_name = None
        self.learner_current_candidate_margin = 0
        self.learner_current_candidate_other_score = 0
        
        self.num_dialog_turns = 0
        
    def __str__(self) :
        return 'Partitions:\n\t' + '\n\t'.join([str(partition) for partition in self.partitions])

    def increment_dialog_turns(self) :
        self.num_dialog_turns += 1
        
    def get_matching_partitions(self, system_action, utterance) :
        matching_partitions = []
        for partition in self.partitions :
            if partition.match(system_action, utterance) :
                matching_partitions.append(partition)
        return matching_partitions
    
    def learner_update(self, learner) :
        (label_name, candidate_idx, margin, other_score) = learner.get_candidate()
        self.learner_current_candidate_idx = candidate_idx
        self.learner_current_candidate_label_name = label_name
        self.learner_current_candidate_margin = margin
        self.learner_current_candidate_other_score = other_score
    
    # WARNING: this is very hacky
    # TODO (even if not making it less hacky): Add a check that removes invalid partitions
    def set_domain_of_discourse(self, domain_of_discourse):
        for partition in self.partitions:
            for param_name in partition.possible_param_values:
                param_values = partition.possible_param_values[param_name]
                consistent_param_values = list()
                for value in param_values:
                    if value is None or not value.startswith('item_'):
                        consistent_param_values.append(value)
                    else:
                        item_num = int(value[5:])
                        if item_num in domain_of_discourse:
                            consistent_param_values.append(value)
                partition.possible_param_values[param_name] = consistent_param_values
            
    
    # Warning: This function only creates matching partitions in the sense
    # that they match in goal and params. In the case of a confirm system
    # action, it does not check whether the user said yes or no    
    def make_all_matching_partitions(self, system_action, utterance, grounder) :
        #print 'In make_all_matching_partitions'    # DEBUG
        #print '**********************************************' # DEBUG
        required_goal = None
        if system_action.referring_goal != None :
            required_goal = system_action.referring_goal
        elif utterance.referring_goal != None :
            required_goal = utterance.referring_goal    
        required_params = dict()
        if utterance.referring_params != None :
            for param_name in utterance.referring_params :
                if param_name in required_params :
                    required_params[param_name] = required_params[param_name] + [utterance.referring_params[param_name]]
                else :
                    required_params[param_name] = [utterance.referring_params[param_name]]
                required_params[param_name] = list(set(required_params[param_name]))
        if system_action.referring_params != None :
            for param_name in system_action.referring_params :
                if param_name not in required_params :
                    required_params[param_name] = [system_action.referring_params[param_name]]
                required_params[param_name] = list(set(required_params[param_name]))
                
        #print "required_goal = ", required_goal        # DEBUG
        #print "required_params = ", required_params    # DEBUG
        
        # Iterate over the partitions and split any partition that is a 
        # superset of what is required
        new_partitions = list()
        for partition in self.partitions :
            #print '#########################'  # DEBUG
            #print str(partition)   # DEBUG
            #print 'Superset : ', partition.is_superset(required_goal, required_params) # DEBUG
            #print 'Equal : ', partition.is_equal(required_goal, required_params)   # DEBUG
            #print '#########################'  # DEBUG
            if partition.is_superset(required_goal, required_params) and not partition.is_equal(required_goal, required_params) :
                #print 'Partition to be split' # DEBUG
                #print '#########################'  # DEBUG
                # This is a partition to be split
                goal_split_partitions = list()
                if required_goal != None :  # Split based on goal
                    goal_split_partitions = partition.split_by_goal(required_goal, self.knowledge, grounder)
                else :
                    goal_split_partitions = [partition]
                    
                #print 'After split by goal - '
                #for partition in goal_split_partitions :
                    #print str(partition)
                #print '#########################'

                if required_params != None :
                    param_value_pairs = list()
                    for param_name in required_params :
                        param_value_pairs = param_value_pairs + [(param_name, value) for value in required_params[param_name]]
                    #print "param_value_pairs = ", param_value_pairs, '\n' # DEBUG
                    if len(param_value_pairs) == 0 :
                        new_partitions = new_partitions + goal_split_partitions
                        continue
                    partitions_to_split = goal_split_partitions
                    resultant_partitions = list()
                    # For each (param_name, param_value), take all the splits
                    # you have already made for the current partition and
                    # split each one of them acc to this pair
                    for (param_name, param_value) in param_value_pairs :
                        resultant_partitions = list()
                        for partition in partitions_to_split :
                            resultant_partitions = resultant_partitions + partition.split_by_param(param_name, param_value, self.knowledge, grounder)
                        partitions_to_split = resultant_partitions
                        
                    new_partitions = new_partitions + resultant_partitions
                else :
                    # There were no params to split on. So only add the 
                    # partitions obtained by splitting the goal 
                    new_partitions = new_partitions + goal_split_partitions
            else :
                # Either this partition is not a superset of the specified 
                # goal or params, or it is already an exact match of the 
                # required goal and params so don't split it
                new_partitions = new_partitions + [partition]    
        self.partitions = new_partitions
        #print 'After all splits - '
        #for partition in self.partitions :
            #print str(partition)
        #print '#########################'
        #print '**********************************************' # DEBUG

    # Perform belief monitoring update using the system action and n best 
    # parses
    def update(self, system_action, n_best_utterances, grounder) :
        #print 'In update'  # DEBUG
        #print '----------------------------------------------' # DEBUG
        #print str(system_action)   # DEBUG
        for utterance in n_best_utterances :
            #print str(utterance)   # DEBUG
            # Check whether there are partitions that exactly match the 
            # system action-utterance pair
            matching_partitions = self.get_matching_partitions(system_action, utterance)
            #print 'len(matching_partitions) = ', len(matching_partitions)  # DEBUG
            if len(matching_partitions) == 0 :
                # if there are no matching partitions, split all partitions
                # that are supersets of the goal and param values of this 
                # pair to obtain matching partitions
                self.make_all_matching_partitions(system_action, utterance, grounder)
            else : # DEBUG
                #for partition in matching_partitions : # DEBUG
                    #print str(partition)   # DEBUG
                pass
        #print '----------------------------------------------' # DEBUG
        hypothesis_beliefs = dict()

        # Clean up partitions
        self.remove_invalid_partitions(grounder)
        self.merge_duplicate_partitions()
        
        #print 'Updation'   # DEBUG
        for partition in self.partitions :
            for utterance in n_best_utterances :
                #print '---------------------------------'  # DEBUG
                #print str(partition)   # DEBUG
                #print str(utterance)   # DEBUG
                if utterance.action_type != '-OTHER-' :
                    hypothesis = (partition, utterance)
                    obs_prob = utterance.parse_prob # Pr(o'/u)
                    #print 'obs_prob = ', numpy.exp(obs_prob)  # DEBUG
                    type_prob = numpy.log(self.knowledge.action_type_probs[system_action.action_type][utterance.action_type])   # Pr(T(u)/T(m))
                    #print 'type_prob = ', numpy.exp(type_prob) # DEBUG
                    param_match_prob = numpy.log(int(utterance.match(partition, system_action))) # Pr(M(u)/p,m)
                    #print 'param_match_prob = ', numpy.exp(param_match_prob)  # DEBUG
                    hypothesis_beliefs[hypothesis] = obs_prob + type_prob + param_match_prob + partition.belief 
                        # b(p,u) = k * Pr(o'/u) * Pr(T(u)/T(m)) * Pr(M(u)/p,m) * b(p)
                    #print 'Belief = ', numpy.exp(hypothesis_beliefs[hypothesis])  # DEBUG
                    #print '---------------------------------'  # DEBUG
                else :
                    # This is a special utterance which indicates that 
                    # the text was generated by something other than all 
                    # the obtained utterances. It does not have goal or 
                    # params but does have an associated probability
                    hypothesis = (partition, '-OTHER-')
                    #print 'Other utterance hypothesis ' # DEBUG
                    obs_prob = utterance.parse_prob 
                    #print 'obs_prob = ', numpy.exp(obs_prob)  # DEBUG
                    match_prob = numpy.log(self.knowledge.non_n_best_match_prob)
                    #print 'match prob = ', numpy.exp(match_prob)  # DEBUG
                    hypothesis_beliefs[hypothesis] = obs_prob + match_prob + partition.belief 
                    #print 'Belief = ', numpy.exp(hypothesis_beliefs[hypothesis])  # DEBUG
                    #print '---------------------------------'  # DEBUG
        
        # Normalize beliefs
        sum_hypothesis_beliefs = float('-inf')
        for hypothesis in hypothesis_beliefs.keys() :
            sum_hypothesis_beliefs = add_log_probs(sum_hypothesis_beliefs, hypothesis_beliefs[hypothesis])
                
        partitionwise_sum = dict()
        #print 'Hypotheses - '  # DEBUG
        for (partition, utterance) in hypothesis_beliefs.keys() :
           hypothesis_beliefs[(partition, utterance)] -= sum_hypothesis_beliefs
           #print '---------------------------------'   # DEBUG
           #print str(partition)    # DEBUG
           #print str(utterance)    # DEBUG
           #print 'Belief = ', hypothesis_beliefs[(partition, utterance)]   # DEBUG
           #print '---------------------------------'   # DEBUG
           if partition not in partitionwise_sum :
               partitionwise_sum[partition] = hypothesis_beliefs[(partition, utterance)]
           else : 
               partitionwise_sum[partition] = add_log_probs(partitionwise_sum[partition], hypothesis_beliefs[(partition, utterance)])
           
        # Reset partition beliefs for next round
        # b(p) = \sum_u b(p,u)
        for partition in self.partitions :
            partition.belief = partitionwise_sum[partition]

        self.hypothesis_beliefs = hypothesis_beliefs
        
    def remove_invalid_partitions(self, grounder) :    
        invalid_partitions = set()  
        for partition in self.partitions :
            partition.remove_invalid_params(self.knowledge, grounder)
            partition.remove_invalid_goals(self.knowledge, grounder) 
            if not partition.is_valid(self.knowledge, grounder) :
                invalid_partitions.add(partition)
        for partition in invalid_partitions :
            self.partitions.remove(partition)        
                    
    def merge_duplicate_partitions(self) :
        i = 0
        while i < len(self.partitions) :
            p1 = self.partitions[i]
            j = i+1
            while j < len(self.partitions) :
                p2 = self.partitions[j]
                if p1.is_like(p2) :
                    p1.belief = add_log_probs(p1.belief, p2.belief)
                    self.partitions.remove(p2)
                else :
                    j += 1
            i += 1
