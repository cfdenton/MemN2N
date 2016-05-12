#!/usr/bin/env python
"""bAbI Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re

def make_dict(file_list):
    word_to_idx = {}
    max_length = 0
    max_idx = 0
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                words = parts[0].split(' ')
                if "?" in parts[0]:
                    a = parts[1].split(' ')
                    words.append(a[0])
                    max_idx = max(max_idx, len(parts[2].split(' ')))
                max_length = max(max_length, len(words[1:]))
                for w in words[1:]:
                    if clean_str(w) not in word_to_idx:
                        word_to_idx[clean_str(w)] = idx
                        idx += 1
    return word_to_idx, max_length, max_idx

def load_dataset(filename):
    idxs = []
    words = []
    query_idx = []
    ans_idx = []
    ans_words = []

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            tokens = parts[0].strip(' ').split(' ')
            if "?" in parts[0]:
                query_idx.append(tokens[0])
                q = [word_to_idx[clean_str(w)] for w in tokens[1:]]
                q.extend((max_length-len(q))*[1])
                words.append(q)
                a = parts[2].split(' ')
                a.extend((max_idx - len(a))*[1])
                ans_idx.append(a)
                ans_words.append(word_to_idx[clean_str(parts[1])])
            else:
                c = [word_to_idx[clean_str(w)] for w in tokens[1:]]
                c.extend((max_length-len(c))*[1])
                words.append(c)
            idxs.append(tokens[0])

    return words, idxs, query_idx, ans_idx, ans_words

def clean_str(string):
    string = string.strip(' ')
    string = re.sub(r"[\.\?]", '', string)
    string = string.lower()
    return string

FILE_PATHS_SPLIT = {"bAbI-1": ("cs287_bAbI_splits/qa1single-supporting-fact-rultrain.txt",
                        "cs287_bAbI_splits/qa1single-supporting-fact-dev.txt"),
              "bAbI-2": ("cs287_bAbI_splits/qa2two-supporting-facts-rultrain.txt",
                         "cs287_bAbI_splits/qa2two-supporting-facts-dev.txt"),
              "bAbI-3": ("cs287_bAbI_splits/qa3three-supporting-facts-rultrain.txt",
                         "cs287_bAbI_splits/qa3three-supporting-facts-dev.txt"),
              "bAbI-4": ("cs287_bAbI_splits/qa4two-arg-relations-rultrain.txt",
                         "cs287_bAbI_splits/qa4two-arg-relations-dev.txt"),
              "bAbI-5": ("cs287_bAbI_splits/qa5three-arg-relations-rultrain.txt",
                         "cs287_bAbI_splits/qa5three-arg-relations-dev.txt"),
              "bAbI-6": ("cs287_bAbI_splits/qa6yes-no-questions-rultrain.txt",
                         "cs287_bAbI_splits/qa6yes-no-questions-dev.txt"),
              "bAbI-7": ("cs287_bAbI_splits/qa7counting-rultrain.txt",
                         "cs287_bAbI_splits/qa7counting-dev.txt"),
              "bAbI-8": ("cs287_bAbI_splits/qa8lists-sets-rultrain.txt",
                         "cs287_bAbI_splits/qa8lists-sets-dev.txt"),
              "bAbI-9": ("cs287_bAbI_splits/qa9simple-negation-rultrain.txt",
                         "cs287_bAbI_splits/qa9simple-negation-dev.txt"),
              "bAbI-10": ("cs287_bAbI_splits/qa10indefinite-knowledge-rultrain.txt",
                         "cs287_bAbI_splits/qa10indefinite-knowledge-dev.txt"),
              "bAbI-11": ("cs287_bAbI_splits/qa11basic-coreference-rultrain.txt",
                         "cs287_bAbI_splits/qa11basic-coreference-dev.txt"),
              "bAbI-12": ("cs287_bAbI_splits/qa12conjunction-rultrain.txt",
                         "cs287_bAbI_splits/qa12conjunction-dev.txt"),
              "bAbI-13": ("cs287_bAbI_splits/qa13compound-coreference-rultrain.txt",
                         "cs287_bAbI_splits/qa13compound-coreference-dev.txt"),
              "bAbI-14": ("cs287_bAbI_splits/qa14time-reasoning-rultrain.txt",
                         "cs287_bAbI_splits/qa14time-reasoning-dev.txt"),
              "bAbI-15": ("cs287_bAbI_splits/qa15basic-deduction-rultrain.txt",
                         "cs287_bAbI_splits/qa15basic-deduction-dev.txt"),
              "bAbI-16": ("cs287_bAbI_splits/qa16basic-induction-rultrain.txt",
                         "cs287_bAbI_splits/qa16basic-induction-dev.txt"),
              "bAbI-17": ("cs287_bAbI_splits/qa17positional-reasoning-rultrain.txt",
                         "cs287_bAbI_splits/qa17positional-reasoning-dev.txt"),
              "bAbI-18": ("cs287_bAbI_splits/qa18size-reasoning-rultrain.txt",
                         "cs287_bAbI_splits/qa18size-reasoning-dev.txt"),
              "bAbI-19": ("cs287_bAbI_splits/qa19path-finding-rultrain.txt",
                         "cs287_bAbI_splits/qa19path-finding-dev.txt"),
              "bAbI-20": ("cs287_bAbI_splits/qa20agents-motivations-rultrain.txt",
                         "cs287_bAbI_splits/qa20agents-motivations-dev.txt")}

FILE_PATHS = {"bAbI-1": ("tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt",
                         "tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt"),
              "bAbI-2": ("tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt",
                         "tasks_1-20_v1-2/en/qa2_two-supporting-facts_test.txt"),
              "bAbI-3": ("tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt",
                         "tasks_1-20_v1-2/en/qa3_three-supporting-facts_test.txt"),
              "bAbI-4": ("tasks_1-20_v1-2/en/qa4_two-arg-relations_train.txt",
                         "tasks_1-20_v1-2/en/qa4_two-arg-relations_test.txt"),
              "bAbI-5": ("tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt",
                         "tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt"),
              "bAbI-6": ("tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt",
                         "tasks_1-20_v1-2/en/qa6_yes-no-questions_test.txt"),
              "bAbI-7": ("tasks_1-20_v1-2/en/qa7_counting_train.txt",
                         "tasks_1-20_v1-2/en/qa7_counting_test.txt"),
              "bAbI-8": ("tasks_1-20_v1-2/en/qa8_lists-sets_train.txt",
                         "tasks_1-20_v1-2/en/qa8_lists-sets_test.txt"),
              "bAbI-9": ("tasks_1-20_v1-2/en/qa9_simple-negation_train.txt",
                         "tasks_1-20_v1-2/en/qa9_simple-negation_test.txt"),
              "bAbI-10": ("tasks_1-20_v1-2/en/qa10_indefinite-knowledge_train.txt",
                         "tasks_1-20_v1-2/en/qa10_indefinite-knowledge_test.txt"),
              "bAbI-11": ("tasks_1-20_v1-2/en/qa11_basic-coreference_train.txt",
                         "tasks_1-20_v1-2/en/qa11_basic-coreference_test.txt"),
              "bAbI-12": ("tasks_1-20_v1-2/en/qa12_conjunction_train.txt",
                         "tasks_1-20_v1-2/en/qa12_conjunction_test.txt"),
              "bAbI-13": ("tasks_1-20_v1-2/en/qa13_compound-coreference_train.txt",
                         "tasks_1-20_v1-2/en/qa13_compound-coreference_test.txt"),
              "bAbI-14": ("tasks_1-20_v1-2/en/qa14_time-reasoning_train.txt",
                          "tasks_1-20_v1-2/en/qa14_time-reasoning_test.txt"),
              "bAbI-15": ("tasks_1-20_v1-2/en/qa15_basic-deduction_train.txt",
                          "tasks_1-20_v1-2/en/qa15_basic-deduction_test.txt"),
              "bAbI-16": ("tasks_1-20_v1-2/en/qa16_basic-induction_train.txt",
                          "tasks_1-20_v1-2/en/qa16_basic-induction_test.txt"),
              "bAbI-17": ("tasks_1-20_v1-2/en/qa17_positional-reasoning_train.txt",
                          "tasks_1-20_v1-2/en/qa17_positional-reasoning_test.txt"),
              "bAbI-18": ("tasks_1-20_v1-2/en/qa18_size-reasoning_train.txt",
                          "tasks_1-20_v1-2/en/qa18_size-reasoning_test.txt"),
              "bAbI-19": ("tasks_1-20_v1-2/en/qa19_path-finding_train.txt",
                          "tasks_1-20_v1-2/en/qa19_path-finding_test.txt"),
              "bAbI-20": ("tasks_1-20_v1-2/en/qa20_agents-motivations_train.txt",
                          "tasks_1-20_v1-2/en/qa20_agents-motivations_test.txt")}

FILE_PATHS_v11 = {"bAbI-1": ("tasksv11en/qa1_single-supporting-fact_train.txt",
                         "tasksv11en/qa1_single-supporting-fact_test.txt"),
              "bAbI-2": ("tasksv11en/qa2_two-supporting-facts_train.txt",
                         "tasksv11en/qa2_two-supporting-facts_test.txt"),
              "bAbI-3": ("tasksv11en/qa3_three-supporting-facts_train.txt",
                         "tasksv11en/qa3_three-supporting-facts_test.txt"),
              "bAbI-4": ("tasksv11en/qa4_two-arg-relations_train.txt",
                         "tasksv11en/qa4_two-arg-relations_test.txt"),
              "bAbI-5": ("tasksv11en/qa5_three-arg-relations_train.txt",
                         "tasksv11en/qa5_three-arg-relations_test.txt"),
              "bAbI-6": ("tasksv11en/qa6_yes-no-questions_train.txt",
                         "tasksv11en/qa6_yes-no-questions_test.txt"),
              "bAbI-7": ("tasksv11en/qa7_counting_train.txt",
                         "tasksv11en/qa7_counting_test.txt"),
              "bAbI-8": ("tasksv11en/qa8_lists-sets_train.txt",
                         "tasksv11en/qa8_lists-sets_test.txt"),
              "bAbI-9": ("tasksv11en/qa9_simple-negation_train.txt",
                         "tasksv11en/qa9_simple-negation_test.txt"),
              "bAbI-10": ("tasksv11en/qa10_indefinite-knowledge_train.txt",
                         "tasksv11en/qa10_indefinite-knowledge_test.txt"),
              "bAbI-11": ("tasksv11en/qa11_basic-coreference_train.txt",
                         "tasksv11en/qa11_basic-coreference_test.txt"),
              "bAbI-12": ("tasksv11en/qa12_conjunction_train.txt",
                         "tasksv11en/qa12_conjunction_test.txt"),
              "bAbI-13": ("tasksv11en/qa13_compound-coreference_train.txt",
                         "tasksv11en/qa13_compound-coreference_test.txt"),
              "bAbI-14": ("tasksv11en/qa14_time-reasoning_train.txt",
                          "tasksv11en/qa14_time-reasoning_test.txt"),
              "bAbI-15": ("tasksv11en/qa15_basic-deduction_train.txt",
                          "tasksv11en/qa15_basic-deduction_test.txt"),
              "bAbI-16": ("tasksv11en/qa16_basic-induction_train.txt",
                          "tasksv11en/qa16_basic-induction_test.txt"),
              "bAbI-17": ("tasksv11en/qa17_positional-reasoning_train.txt",
                          "tasksv11en/qa17_positional-reasoning_test.txt"),
              "bAbI-18": ("tasksv11en/qa18_size-reasoning_train.txt",
                          "tasksv11en/qa18_size-reasoning_test.txt"),
              "bAbI-19": ("tasksv11en/qa19_path-finding_train.txt",
                          "tasksv11en/qa19_path-finding_test.txt"),
              "bAbI-20": ("tasksv11en/qa20_agents-motivations_train.txt",
                          "tasksv11en/qa20_agents-motivations_test.txt")}

args = {}


def main(arguments):
    global args

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('version', help="which version of the data",
                        type=str)
    parser.set_defaults(split=False)
    args = parser.parse_args(arguments)
    paths = {}
    if args.version == '1.2':
        paths = FILE_PATHS
    elif args.version == '1.1':
        paths = FILE_PATHS_v11
    elif args.version == 'split':
        paths = FILE_PATHS_SPLIT
    global word_to_idx, max_length, max_idx
    for task in paths.keys():
        word_to_idx, max_length, max_idx = make_dict(FILE_PATHS_SPLIT[task])
        train, valid = FILE_PATHS_SPLIT[task]
        train_words, train_idxs, train_query_idx, train_ans_idx, train_ans_words = load_dataset(train)
        valid_words, valid_idxs, valid_query_idx, valid_ans_idx, valid_ans_words = load_dataset(valid)
        #test_words, test_context_idx, test_query_idx, test_ans_idx, test_ans_words = load_dataset(test)

        V = len(word_to_idx)+1
        print("vocabsize:", V)
        filename = task + '.hdf5'
        with h5py.File(filename, "w") as f:
            f['train_idxs'] = np.array(train_idxs, dtype=np.int32)
            f['train_query_idx'] = np.array(train_query_idx, dtype=np.int32)
            f['train_words'] = np.array(train_words, dtype=np.int32)
            f['train_ans_idx'] = np.array(train_ans_idx, dtype=np.int32)
            f['train_ans_words'] = np.array(train_ans_words, dtype=np.int32)
            f['valid_idx'] = np.array(valid_idxs, dtype=np.int32)
            f['valid_query_idx'] = np.array(valid_query_idx, dtype=np.int32)
            f['valid_words'] = np.array(valid_words, dtype=np.int32)
            f['valid_ans_idx'] = np.array(valid_ans_idx, dtype=np.int32)
            f['valid_ans_words'] = np.array(valid_ans_words, dtype=np.int32)
            #f['test_context_idx'] = np.array(test_context_idx, dtype=np.int32)
            #f['test_query_idx'] = np.array(test_query_idx, dtype=np.int32)
            #f['test_words'] = np.array(test_words, dtype=np.int32)
            #f['test_ans_idx'] = np.array(test_ans_idx, dtype=np.int32)
            #f['test_ans_words'] = np.array(test_ans_words, dtype=np.int32)

            f['vocabsize'] = np.array([V], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
