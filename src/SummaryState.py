__author__ = 'aishwarya'

import sys, math

class SummaryState :
    discrete_features = [5,6,7,8]
    
    def __init__(self, hisBeliefState=None) :
        # TODO: These default values don't make sense. Think about them
        self.top_hypothesis_prob = 0.0      
        self.second_hypothesis_prob = 0.0

        if hisBeliefState is not None :            
            self.knowledge = hisBeliefState.knowledge
            self.num_dialog_turns =  hisBeliefState.num_dialog_turns
        
        self.top_hypothesis = None
        self.second_hypothesis = None
        
        if hisBeliefState is not None :
            
            self.learner_question_asked = hisBeliefState.learner_question_asked 
                # Has a question ever been asked in the current dialog
            # Information about the potential next question from the learner
            self.learner_current_candidate_idx = hisBeliefState.learner_current_candidate_idx
            self.learner_current_candidate_label_name = hisBeliefState.learner_current_candidate_label_name
            self.learner_current_candidate_margin = hisBeliefState.learner_current_candidate_margin
            self.learner_current_candidate_other_score = hisBeliefState.learner_current_candidate_other_score
            
            if hisBeliefState.hypothesis_beliefs is not None :
                inverse = [(value, key) for (key,value) in hisBeliefState.hypothesis_beliefs.items()]
                if inverse is not None and len(inverse) > 0 :
                    self.top_hypothesis_prob = max(inverse)[0]
                    self.top_hypothesis = max(inverse)[1]
                    inverse.remove(max(inverse))
                if len(inverse) > 0 :
                    self.second_hypothesis_prob = max(inverse)[0]
                    self.second_hypothesis = max(inverse)[1]
                
    def __str__(self) :
        return str(self.get_feature_vector())

    # Ordered list of features
    #     [0] Probability of top hypothesis
    #     [1] Probability of second hypothesis
    #     [2] No of goals allowed by the top partition 
    #     [3] No of params in the top partition required by its action that are uncertain
    #     [4] Number of dialog turns used so far
    #     [5] Do the top and second hypothesis use the same partition: yes/no  
    #     [6] Type of last user utterance - inform_full/inform_param/affirm/deny
    #     [7] Goal in top hypothesis partition or 'None' if this is not unique
    #     [8] Has a learner question been asked in this dialog: yes/no
    #     [9] Margin of potential learner question
    #     [10] Other score of potential learner question
    # NOTE: FILES THAT DEPEND ON THIS ORDERING - 
    #           PomdpKtdqPolicy.py
    def get_feature_vector(self) :
        # Probability of top hypothesis; Probability of second hypothesis
        feature = [self.top_hypothesis_prob, self.second_hypothesis_prob]
        
        # No of goals allowed by the top partition 
        num_goals_allowed = len(self.knowledge.goal_actions)
        if self.top_hypothesis is not None :
            num_goals_allowed = len(self.top_hypothesis[0].possible_goals)
        feature.append(num_goals_allowed)
        
        # No of params in the top partition required by its action that are uncertain
        num_params_uncertain = 0
        if num_goals_allowed == 1 :
            goal = self.top_hypothesis[0].possible_goals[0]
            param_order = self.knowledge.param_order[goal]
            #print 'goal = ', goal, 'param_order = ',param_order    # DEBUG
            for param_name in param_order :
                if len(self.top_hypothesis[0].possible_param_values[param_name]) != 1 :
                    num_params_uncertain += 1
        else :
            num_params_uncertain = len(self.knowledge.goal_params)
        feature.append(num_params_uncertain)           
        
        # Number of dialog turns used so far
        feature.append(self.num_dialog_turns)
        
        # Do the top and second hypothesis use the same partition: Yes/No 
        if self.top_hypothesis is None :
            if self.second_hypothesis is None :
                feature.append('yes')
            else :
                feature.append('no')
        else :
            if self.second_hypothesis is None :
                feature.append('no')
            elif self.top_hypothesis[0] == self.second_hypothesis[0] :
                feature.append('yes')
            else :
                feature.append('no')
            
        # Type of last user utterance
        if self.top_hypothesis is None :
            feature.append(None)
        elif type(self.top_hypothesis[1]) == str :
            feature.append(self.top_hypothesis[1])
        else :
            feature.append(self.top_hypothesis[1].action_type)
            
            
        # Goal in top hypothesis partition or 'None' if this is not unique
        if self.top_hypothesis is None or self.top_hypothesis[0] is None:
            feature.append('None')
        elif len(self.top_hypothesis[0].possible_goals) > 1 :
            feature.append('None')
        elif len(self.top_hypothesis[0].possible_goals) == 1 :
            feature.append(self.top_hypothesis[0].possible_goals[0])
        else :
            feature.append('None')
        
        # Has a learner question been asked in this dialog: yes/no
        feature.append(self.learner_question_asked)
        
        # Margin of potential learner question
        feature.append(self.learner_current_candidate_margin)
        
        # Other score of potential learner question
        feature.append(self.learner_current_candidate_other_score)
        
        return feature
        
    def calc_kernel(self, other_summary_state) :
        p = self.knowledge.kernel_degree
        sigma_k = self.knowledge.kernel_std_dev
        k_disc = 0.0
        norm_cts = 0.0
        self_feature_vector = self.get_feature_vector()
        other_feature_vector = other_summary_state.get_feature_vector()
        for i in xrange(0, len(self_feature_vector)) :
            if i in self.discrete_features :
                k_disc += self.knowledge.kernel_weights[i] * float(self_feature_vector[i] == other_feature_vector[i])
            else :
                norm_cts += self.knowledge.kernel_weights[i] * (self_feature_vector[i] - other_feature_vector[i]) ** 2
        k_cts = p * p * math.exp( - norm_cts / (2 * sigma_k * sigma_k))
        return k_disc + k_cts
     
        
