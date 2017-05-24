__author__ = 'aishwarya'

import itertools, math, numpy
# TODO: Automate filling as many of these as possible from knowledge base

class Knowledge:
    yes = 'yes'
    no = 'no'
    unk = '-UNK-'

    def __init__(self):
        #self.goal_actions = ['searchroom', 'speak_t', 'speak_e', 'remind', 'askperson', 'bring', 'at']
        #self.goal_actions = ['bring', 'at']
        #self.goal_params = ['patient', 'recipient', 'location']
        self.goal_actions = ['push', 'press', 'shake', 'lift', 'drop', 'point']
        self.goal_params = ['patient']

        # This is kept as a single vector common to all values so that hopefully 
        # some operaitons involving them can be made matrix operations and implemented
        # efficiently using numpy
        #self.goal_params_values = [None, 'peter', 'ray', 'dana', 'kazunori', 'matteo', 'shiqi', 'jivko', 'stacy', 'yuqian', 'max', 'pato', 'bwi', 'bwi_m', 'l3_516', 'l3_508', 'l3_512', 'l3_510', 'l3_402', 'l3_418', 'l3_420', 'l3_432', 'l3_502', 'l3_414b', True, False]
        #self.goal_params_values = [None, 'chips', 'coffee', 'hamburger', 'juice', 'muffin', 'alice', 'frannie', 'bob', 'carol', 'dave', 'george', 'eve', 'mallory', 'peggy', 'walter', 'l3_516', 'l3_508', 'l3_512', 'l3_510', 'l3_402', 'l3_418', 'l3_420', 'l3_432', 'l3_502', 'l3_414b']      
        #self.goal_params_values = [None, 'chips', 'coffee', 'hamburger', 'juice', 'muffin', 'stacy', 'ray', 'peter', 'scott', 'dana', 'jivko', 'shiqi', 'jesse', 'aishwarya', 'rodolfo', 'l3_502', 'l3_420', 'l3_432', 'l3_508', 'l3_510', 'l3_512', 'l3_404', 'l3_414b'] 
        #self.goal_params_values = [None, 'stacy', 'ray', 'peter', 'scott', 'dana', 'jivko', 'shiqi', 'jesse', 'aishwarya', 'rodolfo']

        # TODO: Should ideally use these lists for typechecking
        #self.people = ['stacy', 'ray', 'peter', 'scott', 'dana', 'jivko', 'shiqi', 'jesse', 'aishwarya', 'rodolfo']
        self.items = list()    
        for i in range(0, 148) :
            self.items.append('item_' + str(i)) 
        self.goal_params_values = self.items 
        
        self.system_dialog_actions = ['repeat_goal', 'confirm_action', 'request_missing_param', 'ask_classifier_example']
        self.user_dialog_actions = ['inform_param', 'inform_full', 'affirm', 'deny']
        self.summary_system_actions = self.system_dialog_actions + ['take_action']

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
        # N-best list. The actual probability will be this minus the sum 
        # of probabilities of N-best parses
        self.max_obs_by_non_n_best_prob = math.exp(-56)
        self.min_obs_by_non_n_best_prob = math.exp(-58)
        
        # Probability that an utterance not in the N-best list matches 
        # the partition and system_action - This can probably be 
        # calculated exactly but it will be hard to do so.
        self.non_n_best_match_prob = math.exp(-10)
        
        # The system requires arguments to be passed to ASP in specific
        # orders for each action. This is not an obviously generalizable
        # order so it is hard-coded here. The following is the intended
        # interpretation of the parameters
        
        #   searchroom - patient (who/what to search for), location (where to search)
        #   speak_t - patient (what to say)
        #   speak_e - patient (what to say)
        #   remind - patient (event you are reminding someone of), location (where is the 
        #            event), recipient (the person you are reminding)
        #   askperson - patient (the person you ask), recipient (the person you ask about)
        #   bring - patient (what to bring), recipient (whom to bring it to)
        #   walk - location (where to go)
        
        #self.param_order = dict()
        #self.param_order['searchroom'] = ['patient', 'location']
        #self.param_order['speak_t'] = ['patient']
        #self.param_order['speak_e'] = ['patient']
        #self.param_order['remind'] = ['recipient', 'patient', 'location']
        #self.param_order['askperson'] = ['patient', 'recipient']
        #self.param_order['bring'] = ['patient', 'recipient']
        #self.param_order['at'] = ['location']
        
        self.param_order = dict()
        for action in self.goal_actions :
            self.param_order[action] = ['patient']
        #self.param_order['bring'] = ['patient', 'recipient']
        

        self.param_relevance = dict()
        self.set_param_relevance()

        # Constraints for type checking
        self.set_param_constraints()
        
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
        self.expected_actions['repeat_goal'] = ['inform_full']
        self.expected_actions['request_missing_param'] = ['inform_param']
        self.expected_actions['confirm_action'] = ['affirm', 'deny']
        self.expected_actions['ask_classifier_example'] = ['affirm', 'deny']
        for system_dialog_action in self.system_dialog_actions :
            if system_dialog_action not in self.action_type_probs :
                self.action_type_probs[system_dialog_action] = dict()
            for user_dialog_action in self.user_dialog_actions :
                if self.expected_actions[system_dialog_action] == None :
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

    # The true_constraints and false_constraints are to introduce 
    # type constraints for each param of each action, for example, 
    # the patient of searchroom should be a person and the patient 
    # of bring should not be a room. The lists need to be initialized
    # for each parameter relevant to each goal. By default, they will 
    # be empty so it is sufficient to specify them when they are non 
    # empty                        
    def set_param_constraints(self) :
        self.true_constraints = dict()
        self.false_constraints = dict()
        for action in self.goal_actions :
            self.true_constraints[action] = dict()
            self.false_constraints[action] = dict()
            for param in self.goal_params : 
                self.true_constraints[action][param] = list()
                self.false_constraints[action][param] = list()
                
        #self.true_constraints['searchroom']['patient'] = ['person']
        #self.true_constraints['searchroom']['location'] = ['room']
        #self.true_constraints['remind']['recipient'] = ['person']
        #self.true_constraints['remind']['location'] = ['room']
        #self.true_constraints['askperson']['patient'] = ['person']
        #self.true_constraints['askperson']['recipient'] = ['person']
        #self.true_constraints['bring']['patient'] = ['item']
        #self.true_constraints['bring']['recipient'] = ['person']
        #self.true_constraints['at']['location'] = ['room']
        
        for action in self.goal_actions :
            self.true_constraints[action]['patient'] = ['item']
        
        #self.false_constraints['speak_t']['patient'] = ['person', 'room']
        #self.false_constraints['remind']['patient'] = ['person', 'room']
