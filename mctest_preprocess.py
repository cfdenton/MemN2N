#!/usr/bin/env python

"""MCTest Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import nltk.data

def make_dict(file_list, common_words_list=None):
    word_to_idx = {}
    max_length = 0
    max_choice_length = 0
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                story = parts[2]
                story = re.sub(r'\\newline', ' ', story)
                story = re.sub(r'\\tab', ' ', story)
                # replace titles
                story = re.sub(r'(Mr|Ms|Mrs|Dr)\.', r'\g<1>', story)
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                sentences = tokenizer.tokenize(story)
                for sent in sentences:
                    words = sent.strip().split(' ')
                    max_length = max(max_length, len(words))
                    for w in words:
                        if clean_str(w, common_words_list) not in word_to_idx:
                            word_to_idx[clean_str(w, common_words_list)] = idx
                            idx += 1
                questions = parts[3:]
                for i in range(4):
                    q = questions[5*i]
                    words = q.strip().split(' ')
                    max_length = max(max_length, len(words))
                    for w in words:
                        if clean_str(w, common_words_list) not in word_to_idx:
                            word_to_idx[clean_str(w, common_words_list)] = idx
                            idx += 1
                    choices = questions[5*i+1:5*i+5]
                    for choice in choices:
                        words = choice.strip().split(' ')
                        max_choice_length = max(max_choice_length, len(words))
                        for w in words:
                            if clean_str(w, common_words_list) not in word_to_idx:
                                word_to_idx[clean_str(w, common_words_list)] = idx
                                idx += 1
    return word_to_idx, max_length, max_choice_length

def load_dataset(tsv, ans, common_words_list):
    idxs = []
    words = []
    query_idx = []
    ans_words = []
    choice_words = []

    tsv_file = open(tsv, "r")
    if ans is not None:
        ans_file = open(ans, "r")
    letter_to_idx = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    for line in tsv_file:
        parts = line.strip().split('\t')
        story = parts[2]
        story = re.sub(r'\\newline', ' ', story)
        story = re.sub(r'\\tab', ' ', story)
        story = re.sub(r'(Mr|Ms|Mrs|Dr)\.', r'\g<1>', story)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(story)
        questions = parts[3:]
        if ans is not None:
            answers = [letter_to_idx[let] for let in ans_file.readline().strip().split('\t')]
        # add each question with the tokenized story as context
        for i in range(4):
            idx = 1
            for sent in sentences:
                s = [word_to_idx[clean_str(w, common_words_list)] for w in sent.strip().split(' ')]
                s.extend((max_length - len(s))*[1])
                words.append(s)
                idxs.append(idx)
                idx += 1
            q = questions[5*i]
            q = [word_to_idx[clean_str(w, common_words_list)] for w in q.strip().split(' ')]
            q.extend((max_length-len(q))*[1])
            words.append(q)
            idxs.append(idx)
            query_idx.append(idx)
            # add the choices
            choices = questions[5*i+1:5*i+5]
            choice_lst = []
            for choice in choices:
                c = [word_to_idx[clean_str(w, common_words_list)] for w in choice.strip().split(' ')]
                c.extend((max_choice_length-len(c))*[1])
                choice_lst.append(c)
            choice_words.append(choice_lst)
            # add the answer
            if ans is not None:
                ans_words.append(answers[i])

    return np.array(words, dtype=np.int32), np.array(idxs, dtype=np.int32), np.array(query_idx, dtype=np.int32), np.array(ans_words, dtype=np.int32), np.array(choice_words, dtype=np.int32)

def get_common_words(filename):
    common_words_list = set()
    with open(filename, "r") as f:
        for line in f:
            word = line.split(' ')[0]
            common_words_list.add(word)
    return common_words_list

def clean_str(string, common_words_list=None):
    string = string.strip()
    string = re.sub(r"[\.\?\,\!\'\\\:\"]", '', string)
    string = string.lower()
    if common_words_list is not None:
        if string not in common_words_list:
            string = '<unk>'
    return string

def load_word_vecs(filename, word_to_idx, common_words_list):
    word_vecs = np.random.uniform(-0.25, 0.25, (len(word_to_idx.keys())+1, 50))
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            word = clean_str(parts[0], common_words_list)
            vec = parts[1:]
            if word in word_to_idx:
                word_vecs[word_to_idx[word]-1] = vec
    word_vecs[0] = np.zeros(50)
    return np.array(word_vecs, dtype=np.int32)

TSV_FILE_PATHS = {"mc160": ("MCTest/mc160.train.tsv",
                            "MCTest/mc160.dev.tsv",
                            "MCTest/mc160.test.tsv"),
                  "mc500": ("MCTest/mc500.train.tsv",
                            "MCTest/mc500.dev.tsv",
                            "MCTest/mc500.test.tsv")}

ANS_FILE_PATHS = {"mc160": ("MCTest/mc160.train.ans",
                            "MCTest/mc160.dev.ans",
                            "MCTest/mc160.test.ans"),
                  "mc500": ("MCTest/mc500.train.ans",
                            "MCTest/mc500.dev.ans",
                            "MCTest/mc500.test.ans")}

WORD_VECS_PATH = "glove.6B.50d.txt"

args = {}


def main(arguments):
    global args

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('dataset', help="Which MCTest dataset?",
                        type=str)
    parser.add_argument('use_glove', help="GLOVE embeddings?",
                        type=bool)
    parser.set_defaults(dataset="mc160", use_glove=False)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    global word_to_idx, max_length, max_choice_length
    train, valid, test = TSV_FILE_PATHS[dataset]
    train_ans, valid_ans, test_ans = ANS_FILE_PATHS[dataset]
    common_words_list = None
    if args.use_glove:
        common_words_list = get_common_words(WORD_VECS_PATH)
    word_to_idx, max_length, max_choice_length = make_dict([train, valid, test], common_words_list)

    V = len(word_to_idx) + 1
    print("vocabsize:", V)

    train_words, train_idxs, train_query_idx, train_ans_words, train_choice_words = load_dataset(train, train_ans, common_words_list)
    if valid:
        valid_words, valid_idxs, valid_query_idx, valid_ans_words, valid_choice_words = load_dataset(valid, valid_ans, common_words_list)
    if test:
        test_words, test_idxs, test_query_idx, test_ans_words, test_choice_words = load_dataset(test, test_ans, common_words_list)

    if args.use_glove:
        word_vecs = load_word_vecs(WORD_VECS_PATH, word_to_idx, common_words_list)
        args.dataset = args.dataset + '_glove'

    filename = args.dataset + ".hdf5"
    with h5py.File(filename, "w") as f:
        f['train_idxs'] = train_idxs
        f['train_query_idx'] = train_query_idx
        f['train_words'] = train_words
        f['train_ans_words'] = train_ans_words
        f['train_choice_words'] = train_choice_words
        f['valid_idxs'] = valid_idxs
        f['valid_query_idx'] = valid_query_idx
        f['valid_words'] = valid_words
        f['valid_ans_words'] = valid_ans_words
        f['valid_choice_words'] = valid_choice_words
        f['test_idxs'] = test_idxs
        f['test_query_idx'] = test_query_idx
        f['test_words'] = test_words
        f['test_ans_words'] = test_ans_words
        f['test_choice_words'] = test_choice_words
        f['vocabsize'] = np.array([V], dtype=np.int32)
        if args.use_glove:
            f['word_vecs'] = word_vecs

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
