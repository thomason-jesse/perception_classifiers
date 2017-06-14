#!/usr/bin/env python
__author__ = 'aishwarya'


class Policy:
    # policy_type - One of 'guess', 'yes_no', 'example'
    # max_questions - Maximum number of questions the robot is allowed
    #   to ask in a dialog
    # min_confidence_threshold - Minimum average confidence threshold
    #   across candidate yes-no questions to allow asking such a question
    # min_num_unknown_predicates - Minimum number of unknown predicates 
    #   required before asking for example
    def __init__(self, policy_type, max_questions, min_confidence_threshold,
                 min_num_unknown_predicates):
        self.debug_print_level = 2
        self.policy_type = policy_type
        self.max_questions = max_questions
        self.min_confidence_threshold = min_confidence_threshold
        self.min_num_unknown_predicates = min_num_unknown_predicates

    # A util to control how much debug stuff is printed
    def debug_print(self, message, debug_level=2):
        if debug_level <= self.debug_print_level:
            print 'Policy:', message

    def get_next_action(self, state):
        self.debug_print('policy_type = ' + str(self.policy_type), 2)
        if self.policy_type == 'example':
            self.debug_print('len(state[\'unknown_predicates\']) = ' + str(len(state['unknown_predicates'])), 2)
            self.debug_print('self.min_num_unknown_predicates = ' + str(self.min_num_unknown_predicates), 2)
            if (len(state['unknown_predicates']) > self.min_num_unknown_predicates
                    and state['num_dialog_turns'] < self.max_questions):
                return 'ask_positive_example'
                    
        if self.policy_type != 'guess':
            # TODO: You shouldn't get any None's here. Once that is fixed, change the following line
            confidences = [confidence for (predicate, (obj_idx, confidence)) in state['min_confidence_objects'].items()
                           if confidence is not None]
            self.debug_print('confidences = ' + str(confidences), 2)
            self.debug_print('state[\'unknown_predicates\'] = ' + str(state['unknown_predicates']), 2)
            num_candidates = len(state['min_confidence_objects'].items()) + len(state['unknown_predicates'])
            avg_confidence = sum(confidences) / float(num_candidates)
            self.debug_print('avg_confidence = ' + str(avg_confidence), 1)
            self.debug_print('self.min_confidence_threshold = ' + str(self.min_confidence_threshold), 1)
            if (avg_confidence < self.min_confidence_threshold
                    and state['num_dialog_turns'] < self.max_questions):
                return 'ask_predicate_label'

        return 'make_guess'
