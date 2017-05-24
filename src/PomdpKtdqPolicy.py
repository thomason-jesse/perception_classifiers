__author__ = 'aishwarya'

# This implements the KTD-Q algorithm used in 
#       Daubigney, L., Geist, M., Chandramohan, S. & Pietquin, O. (2012), 
#       "A Comprehensive Reinforcement Learning Framework for Dialogue 
#        Management Optimisation", IEEE Journal of Selected Topics in 
#        Signal Processing., December, 2012, Vol. 6(8), pp. 891-902
#
# The algorithm is designed in 
#       Geist, M. & Pietquin, O. (2010), "Kalman Temporal Differences", 
#       Journal of Artificial Intelligence Research (JAIR)., October, 
#       2010, Vol. 39, pp. 483-532.
#
# Variable names imitate the paper but drop superscripts and subscripts 
# when unnecessary for the sake of readibility
# TODO: Include a README with complete algorithm with notation 
# corresponding variable names to code
#
# Implementation has been done using numpy for increased readability and
# in the hope that it scales better. I always use numpy matrices not 
# arrays because arrays don't do vector math in an intuitive way

import numpy as np, sys, math, copy, itertools
from utils import *
from SystemAction import SystemAction
from Action import Action
from SummaryState import SummaryState
from Partition import Partition
from Utterance import Utterance
from AbstractPolicy import AbstractPolicy

class PomdpKtdqPolicy(AbstractPolicy) :
    def __init__(self, knowledge) :
        print 'In policy creation, knowledge class = ', knowledge.__class__.__name__ 
        #x = raw_input()
        
        AbstractPolicy.__init__(self, knowledge)
        self.training = True
        
        # Calculate number of features
        self.n = 6
        #self.n = 1
        #self.n += len(self.knowledge.ktdq_rbf_centres)
        self.n += len(self.knowledge.ktdq_prob_bins)
        self.n += len(self.knowledge.ktdq_prob_bins)
        self.n += len(self.knowledge.goal_actions)
        self.n += len(self.knowledge.goal_params) + 1
        self.n += len(self.knowledge.user_dialog_actions)
        self.n += len(self.knowledge.goal_actions) + 1
        self.n *= len(self.knowledge.summary_system_actions)
        
        # Initialize params
        mu = 0
        sigma = self.knowledge.ktdq_init_theta_std_dev
        self.theta = np.matrix(np.random.normal(mu, sigma, self.n)).T
        self.P = self.knowledge.ktdq_lambda * np.matrix(np.eye(self.n))
        
        # Hyperparameters
        self.eta = self.knowledge.ktdq_eta
        self.P_n = self.knowledge.ktdq_P_n
        self.kappa = self.knowledge.ktdq_kappa
        self.gamma = self.knowledge.gamma 
        
        self.cleaning_epsilon = self.knowledge.ktdq_cleaning_epsilon
        self.alpha = self.knowledge.ktdq_alpha
        self.beta = self.knowledge.ktdq_beta
        
        self.prev_state = None
        self.prev_action = None

    # Convert the summary state feature vector b to the features used by
    # the algorithm and get feature vector for the case when action is a
    def get_feature_vector(self, b, a) :
        orig_feature_vector = b.get_feature_vector()
        new_state_feature_vector = list()
        new_state_feature_vector.append(1)
        
        #s1 = numpy.exp(orig_feature_vector[0])     # Probability of top hypothesis
        #s2 = numpy.exp(orig_feature_vector[1])     # Probability of second hypothesis
        #for (c1, c2) in self.knowledge.ktdq_rbf_centres :
            #dist_to_rbf_centre = (s1 - c1) ** 2 + (s2 - c2) ** 2
            ##feature_value = math.exp(-dist_to_rbf_centre / 2 * (self.knowledge.ktdq_rbf_sigma ** 2))
            #feature_value = dist_to_rbf_centre
            #print 's1 = ', s1, ', s2 = ', s2, ', c1 = ', c1, ', c2 = ', c2, 
            #print ', d = ', dist_to_rbf_centre, ', f = ', feature_value
            ## exp(-((s1 - c1)^2 + (s2 - c2)^2) / 2 * sigma^2)
            #new_state_feature_vector.append(feature_value)
        ##print 'new_state_feature_vector = ', new_state_feature_vector
        #print '----------------'
        
        # Probability of top hypothesis
        s1 = numpy.exp(orig_feature_vector[0])     
        for canonical_prob_value in self.knowledge.ktdq_prob_bins :
            feature_value = numpy.abs(s1 - canonical_prob_value)
            new_state_feature_vector.append(feature_value)
        
        # Probability of second hypothesis
        s2 = numpy.exp(orig_feature_vector[1])     
        for canonical_prob_value in self.knowledge.ktdq_prob_bins :
            feature_value = numpy.abs(s2 - canonical_prob_value)
            new_state_feature_vector.append(feature_value)
        
        # No of goals allowed by the top partition     
        s3 = orig_feature_vector[2]
        # Delta function for each possible value
        for i in range(1, len(self.knowledge.goal_actions) + 1) :
            #print 's3 = ', s3, ', i = ', i
            new_state_feature_vector.append(int(i == s3))
        #print '----------------'

        # No of params in the top partition required by its action that are uncertain    
        s4 = orig_feature_vector[3]
        # Delta function for each possible value
        for i in range(0, len(self.knowledge.goal_params) + 1) :
            #print 's4 = ', s4, ', i = ', i
            new_state_feature_vector.append(int(i == s4))
        #print '----------------'
        
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
            #print 's7 = ', s7, ', value = ', value
            new_state_feature_vector.append(int(s7 == value))    
        #print '----------------'
            
        # Goal in top hypothesis partition or 'None' if this is not unique
        s8 = orig_feature_vector[7]
        # Delta function for each possible value
        for value in self.knowledge.goal_actions + [None] :
            new_state_feature_vector.append(int(s8 == value))
            
        # Has a learner question been asked in this dialog: yes/no
        s9 = orig_feature_vector[8]
        # 0-1 : Has a learner question been asked in this dialog?
        new_state_feature_vector.append(int(s9 == 'yes'))
        
        # Margin of potential learner question
        s10 = orig_feature_vector[9]
        new_state_feature_vector.append(s10)
        
        # Other score of potential learner question
        s11 = orig_feature_vector[10]
        new_state_feature_vector.append(s11)
            
        # For each feature in the state feature vectors, create 4 
        # features by multiplying with delta function of each action
        new_feature_vector = list()
        for value in new_state_feature_vector :
            for action in self.knowledge.summary_system_actions :
                new_feature_vector.append(value * int(action == a))        
                
        return np.matrix(new_feature_vector).T
    
    # Calculate Q(b, a) using current parameters
    def calc_q(self, b, a) :
        f = self.get_feature_vector(b, a)
        return (f.T * self.theta).item(0, 0)
                
    # Pick a random action that is valid for the current state 
    def get_random_action(self, current_state) :
        candidate_actions = self.get_candidate_actions(current_state)
        num_actions = len(candidate_actions)
        sample = int(np.random.uniform(0, num_actions))
        return candidate_actions[sample]
    
    # Defines the current policy
    def pi(self, b) :
        r = np.random.uniform()
        if r < self.knowledge.ktdq_epsilon :
            # Pick random action
            return self.get_random_action(b)
        else :
            candidate_actions = self.get_candidate_actions(b)
            max_q = -sys.maxint
            best_action = None
            for a in candidate_actions :
                q = self.calc_q(b, a)
                print 'a = ', a, ', q = ', q
                if q >= max_q :
                    max_q = q
                    best_action = a
                    
            if best_action is None :
                # This happens if the policy gives Nans for everything 
                return self.get_random_action(b)
                    
            # Greedy action            
            return best_action        
    
    def get_initial_action(self, initial_state) :
        #print 'In get_initial_action'
        self.prev_state = initial_state
        a = 'repeat_goal'
        self.prev_action = a
        return (a, self.get_system_action_requirements(a, initial_state))
    
    def get_next_action(self, reward, current_state) :
        #print 'b = ', current_state.get_feature_vector()
        if self.untrained :
            a_prime = self.get_action_from_hand_coded_policy(current_state)
        else :
            a_prime = self.pi(current_state)
        if self.training :
            if self.prev_state is None or self.prev_action is None :
                raise RuntimeError("Training called without setting previous state or action")
            self.train(self.prev_state, self.prev_action, current_state, reward)
            #self.untrained = False
        self.prev_state = current_state
        self.prev_action = a_prime
        return (a_prime, self.get_system_action_requirements(a_prime, current_state))
    
    def update_final_reward(self, reward) :
        if self.training :
            if self.prev_state is None or self.prev_action is None :
                raise RuntimeError("Training called without setting previous state or action")
            self.train(self.prev_state, self.prev_action, None, reward)
    
    def compute_sigma_points(self) :
        theta = list()
        w = list()
        
        ktdq_lambda = (self.alpha ** 2) * (self.n + self.kappa) - self.n
        
        matrix_to_decompose = (self.n + ktdq_lambda) * self.P
        #print 'matrix_to_decompose = ', matrix_to_decompose
        #self.check_covariance_matrix(matrix_to_decompose)
        cholesky_decomposition = np.linalg.cholesky(matrix_to_decompose)
        
        for j in range(0, 2 * self.n + 1) :
            if j == 0 :
                theta_j = self.theta
                #w_j = self.kappa / (self.n + self.kappa)
                #w_j = (w_j / (self.alpha * self.alpha)) + (1.0 / (self.alpha * self.alpha)) - 1
                w_j = ktdq_lambda / self.n + ktdq_lambda
            elif j <= self.n :
                # Add j^th column of cholesky decomposition
                theta_j = self.theta + cholesky_decomposition[:, j-1]
                #w_j = 1.0 / (self.n + self.kappa)
                #w_j = (w_j / (self.alpha * self.alpha))
                w_j = 1.0 / 2 * (self.n + ktdq_lambda)
            else :
                # Subtract (j-n)^th column of cholesky decomposition
                theta_j = self.theta - cholesky_decomposition[:, j-self.n-1]
                #w_j = 1.0 / 2 * (self.n + self.kappa)
                #w_j = (w_j / (self.alpha * self.alpha))
                w_j = 1.0 / 2 * (self.n + ktdq_lambda)
                
            #theta_j = self.theta + self.alpha * (theta_j - self.theta)
                
            theta.append(theta_j)
            w.append(w_j)
                
            #a = np.random.multivariate_normal(np.array(self.theta.flatten()).flatten(), self.P)
            #theta.append(np.matrix(np.reshape(a, (a.size, 1))))
            #w.append(1)    
        return (theta, w)
    
    # Train on a single example (b, a, b', r)
    def train(self, b, a, b_prime, r) :
        #print 'In train'
        #print 'theta = ', self.theta
        #print 'P = ', self.P
        #raw_input()
        
        # Prediction step
        P_v = self.eta * self.P     
        self.P = self.P + P_v       
        
        # Calculate sigma points - thetas, weights - w for unscented transform
        # Calculate statistics at the same time
        r_hat = 0
        P_theta_r = self.get_zero_vector(self.n)
        P_r = self.P_n
        (theta, w) = self.compute_sigma_points()
        for j in range(0, 2 * self.n + 1) :
            f = self.get_feature_vector(b, a)
            r_j = (f.T * theta[j]).item(0, 0)
            
            if b_prime is not None :
                # If it is a non-terminal state account for future rewards
                max_q = -sys.maxint
                for a_prime in self.get_candidate_actions(b_prime) :
                    f = self.get_feature_vector(b_prime, a_prime)            
                    q = (f.T * theta[j]).item(0, 0)
                    if q > max_q :
                        max_q = q
                f = self.get_feature_vector(b, a)        
                r_j = r_j - self.gamma * max_q
            
            r_hat = r_hat + r_j * w[j]

            if j == 0 :
                P_r = P_r + (w[j] + 1 - (self.alpha ** 2) + self.beta) * (r_j - r_hat) * (r_j - r_hat)
                P_theta_r = P_theta_r + ((w[j] + 1 - (self.alpha ** 2) + self.beta) * (r_j - r_hat)) * (theta[j] - self.theta)
            else :
                P_r = P_r + w[j] * (r_j - r_hat) * (r_j - r_hat)
                P_theta_r = P_theta_r + (w[j] * (r_j - r_hat)) * (theta[j] - self.theta)
        
        #print 'P_r = ', P_r
        #raw_input()
        #print 'P_theta_r = ', P_theta_r
        #raw_input()
        # Correction step    
        K = (1.0 / P_r) * P_theta_r
        #print 'K = ', K
        #raw_input()
        old_theta = self.theta
        self.theta = self.theta + K * (r - r_hat)
        self.P = self.P - P_r * (K * K.T)
        #print 'theta_diff = ', numpy.sum(numpy.absolute(K * (r - r_hat)))
        #print 'theta_diff = ', numpy.sum(numpy.absolute(old_theta - self.theta))
        #raw_input()
        #print 'P = ', self.P
        #raw_input()
        
        # Clean up to keep P positive definite
        #self.P = 0.5 * self.P + 0.5 * self.P.T
        #self.P = self.P + self.cleaning_epsilon * np.eye(self.n)
        
        self.save_vars()

    # A util to get a zero vector because the command is non-intuitive        
    def get_zero_vector(self, size) :
        return np.matrix(np.zeros(size)).T 
        
    def save_vars(self) :
        save_model(self.theta, 'theta')
        save_model(self.P, 'P')

    def print_vars(self) :
        print '------------------------------------------------'
        print 'self.theta = ', self.theta
        print 'self.P = ', self.P
        print '------------------------------------------------'
       
                   
        
    
        
