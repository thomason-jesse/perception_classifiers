__author__ = 'aishwarya'

import itertools, math, numpy
# TODO: Automate filling as many of these as possible from knowledge base

class Knowledge:
    yes = 'yes'
    no = 'no'
    unk = '-UNK-'

    def __init__(self):
        self.goal_actions = ['point']
        self.goal_params = ['patient']
        self.goal_params_values = range(32)
        
        self.summary_system_actions = ["make_guess", "ask_predicate_label", "ask_positive_example"]
        self.system_dialog_actions = ["get_initial_description"] + self.summary_system_actions
        self.user_dialog_actions = ['inform_param', 'affirm', 'deny']

        self.goal_change_prob = 0.0

        # Prob of splitting a partition by specifying a goal
        # partition_split_goal_probs[goal_value] gives the probability
        # when the goal is goal_value
        self.partition_split_goal_probs = dict()
        
        # Prob of splitting a partition by specifying a param
        # partition_split_param_probs[param_name][param_value] gives the
        # probability of splitting by specifying value param_value for 
        # the param param_name
        self.partition_split_param_probs = dict()

        self.set_partition_split_probs()
        
        # Prob that the user says something other than what the system 
        # expects given its previous action
        self.user_wrong_action_prob = 0.01
        
        # action_type_probs[system_action_type][user_action_type] gives 
        # the prob that the utterance has type user_action_type given 
        # that the system action was of type system_action_type
        self.action_type_probs = dict()
        
        self.set_action_type_probs()
        
        # Probability that the obs is due to an utterance not in the 
        # N-best list. 
        self.non_n_best_prob = 0.3
        
        # Probability that an utterance not in the N-best list matches 
        # the partition and system_action - This can probably be 
        # calculated exactly but it will be hard to do so.
        self.non_n_best_match_prob = math.exp(-10)
        
        # Param order doesn't really matter because there is only one 
        # param, but there are too many references to this
        self.param_order = dict()
        for action in self.goal_actions :
            self.param_order[action] = ['patient']

        self.param_relevance = dict()
        self.set_param_relevance()
        
        # Parameters for the RL problem
        self.gamma = 1
        self.correct_action_reward = 100
        self.wrong_action_reward = -100
        self.per_turn_reward = -1
        
        # Set params for the specific RL-algorithm being used
        #self.set_params_for_gp_sarsa()
        self.set_params_for_ktdq()
        
        # The grounder returns possible groundings with the probability 
        # that they satisfy a lambda expression. We retain hypotheses 
        # that have this probability more than the below threshold
        self.grounding_log_prob_threshold = numpy.log(0.5)
    
    # Settings specifically for the KTD-Q algorithm    
    def set_params_for_ktdq(self) :
        self.ktdq_init_theta_std_dev = 0.01
        self.ktdq_lambda = 1
        self.ktdq_eta = 0
        self.ktdq_P_n = 1
        self.ktdq_kappa = 0
        
        rbf_points = [0.25, 0.5, 0.75]
        self.ktdq_prob_bins = rbf_points
        self.ktdq_rbf_centres = list(itertools.product(rbf_points, rbf_points))
        self.ktdq_rbf_sigma = 0.001
        self.ktdq_epsilon = 0.1
        
        # No of turns above which a dialogue is assumed to be long
        self.ktdq_long_dialogue_thresh = 5 
        
        self.ktdq_cleaning_epsilon = 0.01
        self.ktdq_alpha = 0.001
        self.ktdq_beta = 2

    # Settings specifically for the GP-SARSA algorithm
    def set_params_for_gp_sarsa(self) :
        self.gp_sarsa_std_dev = 5
        self.sparsification_param = 1
        
        # Hyperparameters for polynomial kernel - values set from the paper
        self.kernel_std_dev = 5     # sigma_k in the paper
        self.kernel_degree = 4      # p in the paper
        self.kernel_weights = [1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #self.kernel_weights = [1, 1, 1, 1, 1, 1, 1]
        
        # Weight for summary action agreement in GP-SARSA kernel
        # State feature weights in this kernel are the summary space
        # distance weights
        self.summary_action_distance_weight = 0.1
        
        # Distance range in which summary belief points are mapped to
        # the same grid point
        self.summary_space_grid_threshold = 0.2
        
        # Component-wise weights for distance metric in summary space
        # When calculating the distance, a sum of the weighted L2 norm
        # of the continuous components and weighted misclassification
        # distance of discrete components is used. If the weight vector
        # is smaller than the number of features, remaining weights will 
        # be taken as 0
        self.summary_space_distance_weights = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]

    # This gives a 0-1 value for whether a param is relevant for an action. 
    # Values are taken from param_order
    def set_param_relevance(self) :
        self.param_relevance = dict()
        for action in self.goal_actions :
            self.param_relevance[action] = dict()
            for param in self.goal_params :
                if action in self.param_order and param in self.param_order[action] :
                    self.param_relevance[action][param] = 1
                else :
                    self.param_relevance[action][param] = 0
            
    def set_partition_split_probs(self) :
        for goal in self.goal_actions :
            self.partition_split_goal_probs[goal] = 1.0 / len(self.goal_actions)
        for param_name in self.goal_params : 
            self.partition_split_param_probs[param_name] = dict()
            for param_value in self.goal_params_values :
                self.partition_split_param_probs[param_name][param_value] = 1.0 / len(self.goal_params_values)

    def set_action_type_probs(self) :
        self.expected_actions = dict()
        self.expected_actions['get_initial_description'] = ['inform_param']
        self.expected_actions['ask_predicate_label'] = ['affirm', 'deny']
        self.expected_actions['ask_classifier_example'] = ['affirm', 'deny']
        for system_dialog_action in self.system_dialog_actions :
            if system_dialog_action not in self.action_type_probs :
                self.action_type_probs[system_dialog_action] = dict()
            for user_dialog_action in self.user_dialog_actions :
                if system_dialog_action not in self.expected_actions or self.expected_actions[system_dialog_action] == None :
                    # The system doesn't know what user action to expect
                    # So all user actions are equally probable
                    self.action_type_probs[system_dialog_action][user_dialog_action] = 1.0 / len(self.user_dialog_actions)
                else :
                    if user_dialog_action in self.expected_actions[system_dialog_action] :
                        # This is an expected action. Divide the probability 
                        # that the user took an expected action equally 
                        # among all expected actions
                        self.action_type_probs[system_dialog_action][user_dialog_action] = (1.0 - self.user_wrong_action_prob) / len(self.expected_actions[system_dialog_action])
                    else :
                        # This is an unexpected action. Divide the probability
                        # that the user took an unexpected action equally
                        # among all unexpected actions
                        self.action_type_probs[system_dialog_action][user_dialog_action] = self.user_wrong_action_prob / (len(self.user_dialog_actions) - len(self.expected_actions[system_dialog_action]))
]
