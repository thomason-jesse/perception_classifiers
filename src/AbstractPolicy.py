__author__ = 'aishwarya'

# This class provides a set of common utilities required by many 
# POMDP policies. Many of its functions are specific to the current 
# state representation

import numpy as np, sys, math, copy, itertools
from utils import *
from SystemAction import SystemAction
from Action import Action
from SummaryState import SummaryState
from Partition import Partition
from Utterance import Utterance

class AbstractPolicy :
    def __init__(self, knowledge) :
        self.knowledge = knowledge
        self.untrained = True   # Has the policy been learned
        self.training = False   # Is the policy to be updated from 
                                # current interactions
        self.ask_questions_prob_in_hand_coded_policy = 1.0
        
        self.low_belief_threshold = numpy.log(0.3)
        self.medium_belief_threshold = numpy.log(0.75)
        self.high_belief_threshold = numpy.log(0.99)
        
    # Convert the summary state feature vector b to the features used by
    # the algorithm and get feature vector for the case when action is a
    def get_feature_vector(self, b, a) :
        orig_feature_vector = b.get_feature_vector()
        new_state_feature_vector = list()
        new_state_feature_vector.append(1)
        
        s1 = orig_feature_vector[0]     # Probability of top hypothesis
        s2 = orig_feature_vector[1]     # Probability of second hypothesis
        for (c1, c2) in self.knowledge.ktdq_rbf_centres :
            dist_to_rbf_centre = (s1 - c1) ** 2 + (s2 - c2) ** 2
            feature_value = math.exp(-dist_to_rbf_centre / 2 * (self.knowledge.ktdq_rbf_sigma ** 2))
            # exp(-((s1 - c1)^2 + (s2 - c2)^2) / 2 * sigma^2)
            new_state_feature_vector.append(feature_value)
        
        # No of goals allowed by the top partition     
        s3 = orig_feature_vector[2]
        # Delta function for each possible value
        for i in range(1, len(self.knowledge.goal_actions) + 1) :
            new_state_feature_vector.append(int(i == s3))

        # No of params in the top partition required by its action that are uncertain    
        s4 = orig_feature_vector[3]
        # Delta function for each possible value
        for i in range(0, len(self.knowledge.goal_params) + 1) :
            new_state_feature_vector.append(int(i == s4))
        
        # Number of dialog turns used so far    
        s5 = orig_feature_vector[4]        
        # 0-1 : Is dialogue too long?
        new_state_feature_vector.append(int(s5 > self.knowledge.ktdq_long_dialogue_thresh))
        
        # Do the top and second hypothesis use the same partition: yes/no  
        s6 = orig_feature_vector[5]
        # 0-1 : Do the top and second hypothesis use the same partition?
        new_state_feature_vector.append(int(s6 == 'yes'))

        # Type of last user utterance - inform_full/inform_param/affirm/deny        
        s7 = orig_feature_vector[6]
        # Delta function for each possible value
        for value in self.knowledge.user_dialog_actions :
            new_state_feature_vector.append(int(s7 == value))    
            
        # Goal in top hypothesis partition or 'None' if this is not unique
        s8 = orig_feature_vector[7]
        # Delta function for each possible value
        for value in self.knowledge.goal_actions + [None] :
            new_state_feature_vector.append(int(s8 == value))
            
        # For each feature in the state feature vectors, create 4 
        # features by multiplying with delta function of each action
        new_feature_vector = list()
        for value in new_state_feature_vector :
            for action in self.knowledge.summary_system_actions :
                new_feature_vector.append(value * int(action == a))        
                
        return np.matrix(new_feature_vector).T
    
    # Pick a random action that is valid for the current state 
    def get_random_action(self, current_state) :
        candidate_actions = self.get_candidate_actions(current_state)
        num_actions = len(candidate_actions)
        sample = int(np.random.uniform(0, num_actions))
        return candidate_actions[sample]
    
    # Return the set of actions allowed for the current state. These
    # are mostly hand-coded rules
    def get_candidate_actions(self, current_state) :
        candidate_actions = copy.deepcopy(self.knowledge.summary_system_actions)
        
        # Allow 'take-action' only if the top hypothesis has a unique 
        # executable action
        if current_state is None or current_state.top_hypothesis is None :
            candidate_actions.remove('take_action')
            return candidate_actions
        elif current_state.top_hypothesis[0] is None or current_state.top_hypothesis[0].possible_goals is None :
            candidate_actions.remove('take_action')
            return candidate_actions
        elif len(current_state.top_hypothesis[0].possible_goals) != 1 :
            candidate_actions.remove('take_action')
            return candidate_actions
        else :
            # Do not allow 'request-missing-param' for speak actions
            unique_goal = current_state.top_hypothesis[0].possible_goals[0]
            if unique_goal == 'speak_t' or unique_goal == 'speak_e' :
                if 'request_missing_param' in candidate_actions :
                    candidate_actions.remove('request_missing_param')
                    
            # Do not allow 'take-action' if some required param is uncertain
            for param_name in self.knowledge.param_order[unique_goal] :
                if param_name not in current_state.top_hypothesis[0].possible_param_values :
                    if 'take_action' in candidate_actions :
                        candidate_actions.remove('take_action')
                elif len(current_state.top_hypothesis[0].possible_param_values[param_name]) != 1 :
                    if 'take_action' in candidate_actions :
                        candidate_actions.remove('take_action')
            return candidate_actions
    
    # Defines the current policy
    def pi(self, b) :
        raise NotImplementedError("Current policy not defined")       
    
    def get_initial_action(self, initial_state) :
        a = 'repeat_goal'
        return (a, self.get_system_action_requirements(a, initial_state))
    
    def get_next_action(self, reward, current_state) :
        if self.training :
            # Derived classes must implement update rule to be used in 
            # training mode
            raise NotImplementedError("Updation based on reward not defined")       
        a = self.pi(current_state)
        return (a, self.get_system_action_requirements(a, current_state))
    
    def update_final_reward(self, reward) :
        if self.training :
            # Derived classes must implement update rule to be used in 
            # training mode
            raise NotImplementedError("Updation based on final reward not defined")       
    
    def check_covariance_matrix(self, m) :
        n = m.shape[0]
        #for i in range(n) :
            #if m[i, i] < 0 :
                #print 'Diagonal element is < 0'
                #print 'C = ', self.C
                #sys.exit(1)
        for i in range(n) :
            for j in range(n) :
                if m[i, j] != m[j, i] :
                    print 'Matrix not symmetric'
                    sys.exit(1)
        (eig_vals, eig_vectors) = np.linalg.eig(m)
        for eig_val in eig_vals :
            if type(eig_val) in [np.complex_, np.complex64, np.complex128] :
                print 'Complex eigen value'
                #print 'C = ', self.C
                sys.exit(1)
            elif eig_val < -0.0001 :
                print 'Negative eigen value'
                sys.exit(1)
   
    # A util to get a zero vector because the command is non-intuitive        
    def get_zero_vector(self, size) :
        return np.matrix(np.zeros(size)).T 
        
    # Converts a state to a single Action object if possible
    # If the partition either allows more than one goal or more than one
    # value for any param that action needs, then return None
    def resolve_state_to_goal(self, state) :
        if state is None or state.top_hypothesis is None :
            return None
        partition = state.top_hypothesis[0]   
        if partition.possible_goals is None or len(partition.possible_goals) != 1 :
            return None
        goal = partition.possible_goals[0]
        action = Action(goal)
        param_order = state.knowledge.param_order[goal]
        params = []
        if partition.possible_param_values is None :
            return None
        for param_name in param_order :
            if param_name not in partition.possible_param_values or len(partition.possible_param_values[param_name]) != 1 :
                return None
            else :
                params.append(partition.possible_param_values[param_name][0])
        action.params = params    
        return action
    
    def get_system_action_requirements(self, action_type, state) :
        #print 'In get_system_action_requirements, action_type = ', action_type
        if action_type == 'repeat_goal' :
            #print 'Created system action'
            return SystemAction(action_type)
        elif action_type == 'take_action' :
            #print 'Created system action'
            return self.resolve_state_to_goal(state)
            
        elif action_type == 'confirm_action' :
            if state.top_hypothesis is None or state.top_hypothesis[0] is None :
                # No hypotheses to verify. Confirm something random
                goal_idx = int(np.random.uniform(0, len(state.knowledge.goal_actions)))
                return SystemAction(action_type, state.knowledge.goal_actions[goal_idx])
            
            goal = None
            if len(state.top_hypothesis[0].possible_goals) > 0 : 
                goal_idx = int(np.random.uniform(0, len(state.top_hypothesis[0].possible_goals)))    
                goal = state.top_hypothesis[0].possible_goals[goal_idx]
            else :
                # Top hypothesis has no goals, verify any random goal
                goal_idx = int(np.random.uniform(0, len(self.knowledge.possible_goals)))    
                goal = state.top_hypothesis[0].possible_goals[goal_idx]

            system_action = SystemAction(action_type, goal)
            param_order = state.knowledge.param_order[goal]
            params = dict()
            partition_params = state.top_hypothesis[0].possible_param_values
            if partition_params is None :
                return system_action
            for param_name in param_order :
                if param_name in partition_params and len(partition_params[param_name]) == 1 :
                    # Only confirm params if the hypothesis thinks there is only one 
                    # possible value for it
                    params[param_name] = partition_params[param_name][0]
                #else :
                    #print 'Uncertain about ', param_name, ' : ', partition_params[param_name]
            system_action.referring_params = params
            #print 'Created system action'
            return system_action
            
        elif action_type == 'request_missing_param' :
            if state.top_hypothesis is None or state.top_hypothesis[0] is None :
                # No hypotheses. Anything can be asked
                goal_idx = int(np.random.uniform(0, len(state.knowledge.goal_actions)))
                goal = state.knowledge.goal_actions[goal_idx]
                system_action = SystemAction(action_type, goal)
                param_idx = int(np.random.uniform(0, len(state.knowledge.param_order[goal])))
                system_action.extra_data['param_name'] = state.knowledge.param_order[goal][param_idx]
                #print 'Created system action'
                return system_action
                    
            goal = state.top_hypothesis[0].possible_goals[0]
            system_action = SystemAction(action_type, goal)
            param_order = state.knowledge.param_order[goal]
            params = dict()
            uncertain_params = list()
            partition_params = state.top_hypothesis[0].possible_param_values
            if partition_params is None :
                return system_action
            for param_name in param_order :
                if param_name not in partition_params or len(partition_params[param_name]) != 1 :
                    uncertain_params.append(param_name)
                else :
                    params[param_name] = partition_params[param_name][0]
            system_action.referring_params = params
            if len(uncertain_params) > 0 :
                # If the top partition is uncertain about a param, confirm it
                param_idx = int(np.random.uniform(0, len(uncertain_params)))
                system_action.extra_data['param_name'] = uncertain_params[param_idx]
                #print 'Created system action'
                return system_action
            else :
                # The top hypothesis partition doesn't have uncertain 
                # params but it is possible it is not of high enough 
                # confidence
                
                # If there is no second hypothesis, just confirm any value
                # This param si chosen at random so that you don't get 
                # stuck in a loop here
                if state.second_hypothesis is None or state.second_hypothesis[0].possible_param_values is None :
                    param_idx = int(np.random.uniform(0, len(param_order)))
                    system_action.extra_data['param_name'] = param_order[param_idx]
                    #print 'Created system action'
                    return system_action
                
                # A good heuristic is to see in what params the first 
                # and second hypotheses differ. Any one of these is 
                # likely to help. 
                second_params = state.second_hypothesis[0].possible_param_values
                for param_name in param_order :
                    top_param_value = partition_params[param_name][0]
                    if param_name not in second_params or top_param_value not in second_params[param_name] or len(second_params[param_name]) == 1 :
                        # This is not a useful param to compare
                        pass
                    else :
                        system_action.extra_data['param_name'] = param_name
                        #print 'Created system action'
                        return system_action
                
                # If you reached here, this is probably an inappropriate 
                # action so just verify a random param. 
                param_idx = int(np.random.uniform(0, len(param_order)))
                system_action.extra_data['param_name'] = param_order[param_idx]
                #print 'Created system action'
                return system_action
            
            #print 'Created system action'    
            return system_action
        
        elif action_type == 'ask_classifier_example' :
            system_action = SystemAction(action_type)
            system_action.extra_data['label_name'] = state.learner_current_candidate_label_name
            system_action.extra_data['candidate_idx'] = state.learner_current_candidate_idx
            system_action.extra_data['margin'] = state.learner_current_candidate_margin
            system_action.extra_data['learner_other_score'] = state.learner_current_candidate_other_score
            #print 'Created system action'
            return system_action
            
                    
    def get_action_from_hand_coded_policy(self, state) :
        #print 'In get_action_from_hand_coded_policy'
        if state.num_dialog_turns > 10 :
            return 'take_action'
        feature_vector = state.get_feature_vector()
        #print 'Got feature vector'
        num_goals = feature_vector[2]
        num_uncertain_params = feature_vector[3]
        last_utterance_type = feature_vector[6]
        learner_question_asked = feature_vector[8]
        #print "Before the if-else's"
        if num_goals == 1 :
            if num_uncertain_params == 0 :
                if feature_vector[0] < self.low_belief_threshold :
                    return 'request_missing_param'
                if last_utterance_type == 'affirm' or last_utterance_type == 'deny' :
                    if feature_vector[0] < self.medium_belief_threshold :
                        return 'repeat_goal'      
                    else :
                        r = numpy.random.uniform()
                        if learner_question_asked == 'no' and r < self.ask_questions_prob_in_hand_coded_policy :
                            return 'ask_classifier_example'
                        else :
                            return 'take_action'
                else :
                    if feature_vector[0] < self.high_belief_threshold :
                        return 'confirm_action'
                    else :
                        r = numpy.random.uniform()
                        if learner_question_asked == 'no' and r < self.ask_questions_prob_in_hand_coded_policy :
                            return 'ask_classifier_example'
                        else :
                            return 'take_action'
            else :
                return 'request_missing_param'
        else :
            return 'repeat_goal'      
        
                   
        
    
        
