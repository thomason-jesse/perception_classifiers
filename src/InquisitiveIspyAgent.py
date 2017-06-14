#!/usr/bin/env python
__author__ = 'aishwarya'

import operator
import numpy as np
import traceback
from UnitTestAgent import UnitTestAgent
from perception_classifiers.srv import *


class InquisitiveIspyAgent(UnitTestAgent):
    # initial_predicates - The agent needs to track which classifiers 
    #       it can call. This gives the initial list. This has to match 
    #       the list in python classifier services
    def __init__(self, io, table_oidxs, stopwords_fn, policy, log_fn=None, initial_predicates=None):
        self.debug_print_level = 2  # Print messages with debug_level <= this
        UnitTestAgent.__init__(self, io, 2, table_oidxs)
        
        self.io = io
        self.log_fn = log_fn
        self.policy = policy
        
        # Read stopwords
        self.stopwords = []
        if stopwords_fn is not None:
            fin = open(stopwords_fn, 'r')
            for line in fin.readlines():
                self.stopwords.append(line.strip())
            fin.close()
        
        # Known predicates - Those for which you can ask the classifiers for a label
        # Unknown predicates - Those for which you can't
        self.known_predicates = list()
        self.unknown_predicates = list()
        if initial_predicates is not None:
            self.known_predicates = initial_predicates
        self.debug_print('self.known_predicates = ' + str(self.known_predicates), 1)
        self.debug_print('self.unknown_predicates = ' + str(self.unknown_predicates), 1)
        
        self.objects_for_guessing = self.table_oidxs[1]
        self.objects_for_questions = self.table_oidxs[0] + self.table_oidxs[2]
        
        # Caching useful classifier info    
        self.classifiers_changed = list() 
        
        # key - predicate, key - oidx, value [result, confidence]
        self.current_classifier_results = None
        
        # Key: predicate; Value: (obj_idx, confidence)
        self.min_confidence_objects = dict()
        self.io.say("I am thinking about the objects around me.")
        self.update_min_confidence_objects()
        
        # Some additional state info
        self.num_dialog_turns = 0
        self.cur_dialog_predicates = None
        self.cur_match_scores = None
        
        self.blacklisted_predicates_for_example = None
        

    # A util to control how much debug stuff is printed
    def debug_print(self, message, debug_level=2):
        if debug_level <= self.debug_print_level:
            print 'PomdpIspyAgent:', message

    def log(self, log_str):
        self.debug_print(log_str.strip(), 5)
        if self.log_fn is not None:
            f = open(self.log_fn, 'a')
            f.write(log_str)
            f.close()

    def get_dialog_state(self):
        dialog_state = dict()
        dialog_state['num_dialog_turns'] = self.num_dialog_turns
        dialog_state['match_scores'] = self.cur_match_scores
        candidate_predicates_for_example = set(self.unknown_predicates).difference(self.blacklisted_predicates_for_example)
        dialog_state['unknown_predicates'] = self.candidate_predicates_for_example
        dialog_state['cur_dialog_predicates'] = self.cur_dialog_predicates
        dialog_state['min_confidence_objects'] = self.min_confidence_objects
        self.debug_print('dialog_state = ' + str(dialog_state), 2)
        return dialog_state

    def make_guess(self, match_scores):
        # match_scores.items() gives (obj_idx, score) tuples
        # Shuffle to randomize order of equally scored items on sorting
        obj_indices_with_scores = np.random.permutation(match_scores.items())
        # Sort by match score to identify what the robot should guess
        sorted_match_scores = sorted(obj_indices_with_scores, key=operator.itemgetter(1), reverse=True)
        guess_idx = int(sorted_match_scores[0][0]) 

        # Point to best guess
        self.debug_print('Trying to point to ' + str(guess_idx), 1)
        point_success = self.point_to_object(guess_idx)
        self.debug_print('point_success = ' + str(point_success), 1)

        # Ask if the guess was right
        got_confirmation = False
        correct = False
        while not got_confirmation:
            got_confirmation = True
            self.io.say("Is this the object you have in mind?")
            confirmation = self.io.get()
            if self.is_yes(confirmation):
                correct = True
            elif not self.is_no(confirmation):
                got_confirmation = False
                self.io.say("I didn't catch that.")
                
        # Stop pointing
        self.retract_arm()

        # Add required classifier updates
        if correct:
            # Give a label of +1 to all predicates in current dialog with 
            # the object guess_idx
            new_preds = [predicate for predicate in self.cur_dialog_predicates
                         if predicate not in self.known_predicates]
            self.debug_print('new_preds = ' + str(new_preds), 2)
            self.known_predicates.extend(new_preds)  # Needed here to get correct indices of new predicates
            pidxs = [self.known_predicates.index(predicate) for predicate in self.cur_dialog_predicates]
            oidxs = [guess_idx] * len(pidxs)
            labels = [True] * len(pidxs)
            success = self.update_classifiers(new_preds, pidxs, oidxs, labels)
            if success:
                for predicate in new_preds:
                    if predicate in self.unknown_predicates:
                        self.unknown_predicates.remove(predicate)
            else:
                # Update didn't happen so undo the extension
                for predicate in new_preds:
                    self.known_predicates.remove(predicate)
            self.classifiers_changed = self.classifiers_changed + self.cur_dialog_predicates
            
            self.debug_print('self.known_predicates = ' + str(self.known_predicates), 2)
            self.debug_print('self.unknown_predicates = ' + str(self.unknown_predicates), 2)

    # Identify the object and predicate for which a label should be obtained    
    def get_label_question_details(self):
        # TODO: Once the classifiers stop returning None, you shouldn't need the None check
        predicates = self.unknown_predicates + [predicate for predicate in self.min_confidence_objects.keys()
                                                if self.min_confidence_objects[predicate][1] is not None]
        # Sample a predicate with probability proportional to 1 - confidence in lowest confidence object
        prob_numerators = [1.0] * len(self.unknown_predicates) + [(1.0 - self.min_confidence_objects[predicate][1])
                                                                  for predicate in self.min_confidence_objects.keys()
                                                                  if self.min_confidence_objects[predicate][1]
                                                                  is not None]
        probs = [(v / sum(prob_numerators)) for v in prob_numerators]
        predicate = np.random.choice(predicates, p=probs)
        
        if predicate in self.unknown_predicates:
            candidate_object = np.random.choice(self.objects_for_questions)
        else:
            candidate_object = self.min_confidence_objects[predicate][0]
        
        return predicate, candidate_object
    
    def ask_predicate_label(self):
        predicate, candidate_object = self.get_label_question_details()
        
        # Point to the object about which the question is
        self.point_to_object(candidate_object)
                    
        question_str = 'Would you use the word ' + predicate + ' to describe this object?'
        
        # Get response
        got_answer = False
        label_value = 0
        while not got_answer:
            got_answer = True
            self.io.say(question_str)
            answer = self.io.get()
            if self.is_repeat(answer):
                continue
            elif self.is_yes(answer):
                label_value = 1
            elif not self.is_no(answer):
                got_answer = False
                self.io.say("I didn't catch that.")
                
        # Stop pointing
        self.retract_arm()
        
        # Add required classifier update
        if predicate in self.unknown_predicates:
            new_preds = [predicate]
        else:
            new_preds = []
        self.known_predicates.extend(new_preds)  # Needed here to get correct indices of new predicates
        pidxs = [self.known_predicates.index(predicate)]
        oidxs = [candidate_object]
        labels = [label_value]
        self.debug_print('Updating in ask_predicate_label new_preds = ' + str(new_preds), 1)
        self.debug_print('Updating in ask_predicate_label pidxs = ' + str(pidxs), 1)
        self.debug_print('Updating in ask_predicate_label oidxs = ' + str(oidxs), 1)
        self.debug_print('Updating in ask_predicate_label labels = ' + str(labels), 1)
        success = self.update_classifiers(new_preds, pidxs, oidxs, labels)
        if success:
            if predicate in self.unknown_predicates:
                self.unknown_predicates.remove(predicate)
            self.classifiers_changed.append(predicate)
        else:
            # Update didn't happen so undo the extension
            if predicate in self.unknown_predicates:
                self.known_predicates.remove(predicate)

    def ask_positive_example(self):
        candidate_predicates = set(self.unknown_predicates).difference(self.blacklisted_predicates_for_example)
        predicate = np.random.choice(candidate_predicates)
        
        question_str = 'Could you show me an object that you would describe as ' + predicate + '?'
        self.io.say(question_str)

        # Loop while user reorients the robot.
        self.debug_print('Waiting for user direction to table for touch detection', 1)
        ready_to_detect = False
        no_such_object = False
        while not ready_to_detect:
            cmd = self.io.get()
            self.debug_print('cmd = ' + str(cmd), 1)
            if 'none' in cmd:
                no_such_object = True
                break
            tid = self.table_turn_command(cmd)
            self.debug_print('Identified target table ID ' + str(tid), 1)
            if tid is not None:
                self.face_table(tid, report=True)
                self.debug_print('Faced table ' + str(tid), 1)
            elif self.is_detect(cmd) and self.tid != 2:  # don't detect test objects
                ready_to_detect = True
            else:
                self.io.say("I didn't catch that.")

        if no_such_object:
            self.blacklisted_predicates_for_example.append(predicate)
            self.io.say("I see.")
            return
        
        # Detect touch
        self.debug_print('Waiting to detect touch', 1)
        touch_str = 'I am waiting to detect the object your touch.'
        self.io.say(touch_str)
        pos_detected, obj_idx_detected = self.detect_touch()
        
        # Add required classifier update
        new_preds = [predicate]
        self.known_predicates.extend(new_preds)  # Needed here to get correct indices of new predicates
        pidxs = [self.known_predicates.index(predicate)]
        oidxs = [obj_idx_detected]
        labels = [1]
        success = self.update_classifiers(new_preds, pidxs, oidxs, labels)
        if success:
            self.unknown_predicates.remove(predicate)
            self.classifiers_changed.append(predicate)
        else:
            # Update didn't happen so undo the extension
            self.known_predicates.remove(predicate)

    # Start of human_take_turn of IspyAgent. Re-coded here because
    # only part of that function is to be used here
    def get_initial_description(self):
        self.io.say("Please pick an object that you see and describe it to me in one phrase.")

        understood = False
        predicates = None
        
        while not understood:
            user_response = self.io.get().strip()
            self.log('Get : ' + user_response + '\n')
            if self.is_repeat(user_response):
                self.repeat_self()
                continue
            predicates = self.get_predicates(user_response)
            self.log("Predicates : " + str(predicates) + "\n")

            # If we get some predicates, repeat back the whole phrase to the user to confirm it.
            if len(predicates) > 0:
                while not understood:
                    self.io.say("Did you say: ' " + user_response + " ' ?")
                    user_conf = self.io.get()
                    if self.is_repeat(user_conf):
                        self.repeat_self()
                    elif self.is_yes(user_conf):
                        understood = True
                    elif self.is_no(user_conf):
                        self.io.say("My hearing is not so good. Could you repeat your description?")
                        break
                    else:
                        self.io.say("I didn't catch that.")
            # Failed to parse user response, so get a new one
            else:
                self.io.say("Sorry; I didn't catch that. Could you re-word your description?")

        self.cur_dialog_predicates = predicates

    def get_predicates(self, user_response):
        response_parts = user_response.split()
        predicates = [w for w in response_parts if w not in self.stopwords]
        predicates = list(set(predicates))  # remove duplicates
        unknown_predicates = [predicate for predicate in predicates if predicate not in self.known_predicates]
        self.unknown_predicates.extend(unknown_predicates)
        return predicates
    
    # Given predicates and object idxs, return a map of results
    def get_classifier_results(self, predicates, obj_indices):
        if self.current_classifier_results is None:
            results = dict()
        else:
            results = self.current_classifier_results

        for predicate in predicates:
            pred_results = dict()
            for obj_idx in obj_indices:
                if predicate in self.known_predicates:
                        if predicate not in results or predicate in self.classifiers_changed:
                                self.debug_print('In get_classifier_results fetching result for predicate '
                                                 + predicate + ' for object ' + str(obj_idx), 3)
                                classifier_idx = self.known_predicates.index(predicate)
                                [result, confidence] = self.run_classifier_on_object(classifier_idx, obj_idx)
                        else:
                            [result, confidence] = results[predicate][obj_idx]
                else:
                    [result, confidence] = [0, 0.0]
                pred_results[obj_idx] = [result, confidence]
            results[predicate] = pred_results

        self.current_classifier_results = results
        return results

    # Calculate match scores from an object given a set of cnf clauses of predicates
    def get_match_scores(self):
        predicates = self.cur_dialog_predicates
        classifier_results = self.get_classifier_results(predicates, self.objects_for_guessing)
        self.debug_print('classifier_results = ' + str(classifier_results), 3)

        match_scores = dict()
        sum_match_scores = 0.0
        for obj_idx in self.objects_for_guessing:
            predicate_scores = list()
            for predicate in predicates:
                # Compute per-predicate score as just result * confidence
                # Since result is 0-1, this is basically equal to the 
                # confidence that this predicate holds for this object
                [result, confidence] = classifier_results[predicate][obj_idx]
                predicate_scores.append(result * confidence)

            # Sum the predicate scores across predicates
            match_scores[obj_idx] = sum(predicate_scores)
            sum_match_scores += match_scores[obj_idx]
        
        # We need match scores that work like probabilities so normalize these
        if np.isclose([sum_match_scores], [0.0]):
            for obj_idx in self.objects_for_guessing:
                match_scores[obj_idx] = 1.0 / len(self.objects_for_guessing)
        else:
            for obj_idx in self.objects_for_guessing:
                match_scores[obj_idx] /= sum_match_scores

        return match_scores

    def update_min_confidence_objects(self):
        for predicate in self.known_predicates:
            if predicate not in self.min_confidence_objects or predicate in self.classifiers_changed:
                for obj_idx in self.objects_for_questions:
                    self.debug_print('In update_min_confidence_objects fetching result of predicate '
                                     + predicate + ' for object ' + str(obj_idx), 3)
                    classifier_idx = self.known_predicates.index(predicate)
                    _, confidence = self.run_classifier_on_object(classifier_idx, obj_idx)
                    if (predicate not in self.min_confidence_objects
                            or confidence < self.min_confidence_objects[predicate][1]):
                        self.min_confidence_objects[predicate] = (obj_idx, confidence)

    def run_dialog(self):
        self.debug_print('In run_dialog', 2)
        
        self.current_classifier_results = None
        self.classifiers_changed = list()
        self.blacklisted_predicates_for_example = list()
        self.num_dialog_turns = 0
        
        # Get initial user description and update state
        self.log('Action : get_initial_description')
        self.cur_dialog_predicates = None
        self.get_initial_description()  # Sets self.cur_dialog_predicates

        self.cur_match_scores = self.get_match_scores()
        self.log("Match scores :" + str(self.cur_match_scores) + "\n\n")
        
        dialog_state = self.get_dialog_state()
        dialog_action = self.policy.get_next_action(dialog_state)
        
        object_guessed = False
        try:
            while not object_guessed:
                self.log('\n\nAction : ' + dialog_action)
                self.debug_print('Action : ' + dialog_action)
                self.debug_print('self.classifiers_changed = ' + str(self.classifiers_changed), 1)
                self.num_dialog_turns += 1
                
                if dialog_action == 'make_guess':
                    self.make_guess(self.cur_match_scores)
                    object_guessed = True
                    
                elif dialog_action == 'ask_predicate_label':
                    self.ask_predicate_label()
                    
                elif dialog_action == 'ask_positive_example':
                    self.ask_positive_example()
                
                self.debug_print('Dialog action executed', 1)
                self.debug_print('object_guessed = ' + str(object_guessed), 1)
                self.debug_print('self.classifiers_changed = ' + str(self.classifiers_changed), 1)
                
                # Update cached data for questions
                self.update_min_confidence_objects()
                
                if not object_guessed:
                    # Recompute match scores because relevant classifiers may
                    # have changed
                    self.cur_match_scores = self.get_match_scores()
                    self.log("Match scores : " + str(self.cur_match_scores) + "\n")
                    
                    self.classifiers_changed = list()

                    dialog_state = self.get_dialog_state()
                    dialog_action = self.policy.get_next_action(dialog_state)
                    
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            # Log the exception
            self.debug_print('\n' + traceback.format_exc() + '\n', 0)
            self.log('\n' + traceback.format_exc() + '\n')
            
            # TODO: Might want to block exceptions during actual experiment
            raise

    # process an utterance and return the table id to turn to if it seems to be a turning command
    def table_turn_command(self, u):
        tid = None
        if u == 'turn left':
            tid = self.tid - 1
        elif u == 'turn right':
            tid = self.tid + 1
        elif u == 'turn around' and self.tid != 2:
            tid = self.tid + 2
        elif 'turn to table' in u or 'face table' in u or 'turn table' in u:
            if 'one' in u or '1' in u:
                tid = 1
            elif 'two' in u or '2' in u:
                tid = 2
            elif 'three' in u or '3' in u:
                tid = 3
        
        self.debug_print('tid before mod = ' + str(tid), 2)
                
        if tid is not None and tid <= 0:
            tid = 3
        elif tid is not None and tid >= 4:
            tid = 1
        
        self.debug_print('tid after mod = ' + str(tid), 2)
                
        return tid

    # determine whether an utterance is basically 'yes' or 'no'
    def is_yes(self, u):
        return self.check_for_word(u, ['yes', 'sure', 'yeah'])

    def is_no(self, u):
        return self.check_for_word(u, ['no', 'nope', 'nah'])

    def is_repeat(self, u):
        return self.check_for_word(u, ['what', 'huh', 'repeat'])

    def is_detect(self, u):
        return self.check_for_word(u, ['ready', 'watch', 'look'])

    def check_for_word(self, u, l):
        ws = u.split()
        for w in l:
            if w in ws:
                return True
        return False
