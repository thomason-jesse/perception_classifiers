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
    def __init__(self, policy_type, max_questions=5, min_confidence_threshold=0.0001, \
                 min_num_unknown_predicates=3):
        self.policy_type = policy_type
        self.max_questions = max_questions
        self.min_confidence_threshold = min_confidence_threshold
        self.min_num_unknown_predicates = min_num_unknown_predicates


    def get_next_action(self, state):
        if self.policy_type == 'example':
            if len(state['unknown_predicates']) > min_num_unknown_predicates:
                return 'ask_positive_example'
        elif self.policy_type != 'guess':
            confidences = [confidence for (predicate, (obj_idx, confidence)) in state['min_confidence_objects'].items()]
            num_candidates = len(confidences) + len(state['unknown_predicates'])
            avg_confidence = confidences / float(num_candidates)
            if avg_confidence < min_confidence_threshold:
                return 'ask_predicate_label'

        return 'make_guess'
