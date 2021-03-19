import os
import re
import html
from typing import List, Tuple, T, Iterable, Union, NewType
from params import param


cue_id2label = {0: 'Affix', 1: 'Cue', 2: 'MCue', 3: 'O'}
cue_label2id = {0: 0, 1: 1, 2: 2, 3: 3}
scope_id2label = {0: '<PAD>', 1: 'I', 2:'O', 3: 'C', 4: 'B', 5: 'E', 6: 'S'}

class RawData():
    """
    Wrapper of data. For sfu and bioscope data, splitting of train-dev-test will also be done here.
    class var:
        cues (tuple[T]): cues data.
            (sentences, cue_labels, cue_sep, num_cues)
        scopes (tuple[T]): scopes data.
            (original_sent, sentences[n], cue_labels[n], scope_labels[n]). n=num_cues
        non_cue_sents (tuple[T]): sentences that does not contain negation.
    For detail refer to the local methods for each dataset in constructor
    """
    def __init__(self, file, dataset_name='sfu'):
        """
        params:
            file: The path of the data file.
            dataset_name: The name of the dataset to be preprocessed. Values supported: sfu, bioscope, sherlock.
            frac_no_cue_sents: The fraction of sentences to be included in the data object which have no negation/speculation cues.
        """
        if dataset_name == 'bioscope':
            self.cues, self.scopes, self.non_cue_sents = bioscope(file)
        elif dataset_name == 'sfu':
            sfu_cues = [[], [], [], []]
            sfu_scopes = [[], [], [], []]
            sfu_noncue_sents = []
            for dir_name in os.listdir(file):
                if '.' not in dir_name:
                    for f_name in os.listdir(os.path.join(file, dir_name)):
                        r_val = sfu_review(
                            os.path.join(file, dir_name, f_name))
                        sfu_cues = [a+b for a, b in zip(sfu_cues, r_val[0])]
                        sfu_scopes = [a+b for a,
                                      b in zip(sfu_scopes, r_val[1])]
                        sfu_noncue_sents.extend(r_val[2])
            self.cues = sfu_cues
            self.scopes = sfu_scopes
            self.non_cue_sents = sfu_noncue_sents
        elif dataset_name == 'sherlock':
            self.cues, self.scopes, self.non_cue_sents = sherlock(file)  
        else:
            raise ValueError(
                "Supported Dataset types are:\n\tbioscope\n\tsfu\n\tconll_cue")

def sherlock(f_path) -> Tuple[List, List, List]:
    """
    return: raw data format. (cues, scopes, non_cue_sents)
        cues (list[list[T]]): (sentences, cue_labels, cue_sep, num_cues)
            cue_sep is the seperation label of cues. 
            num_cues notes the number of cues in this sent.
        scopes (list[list[T]]): (original_sent, sentences[n], cue_labels[n], scope_labels[n])
            where n = number of cues in this sentence.
            Note that for senteces[n], the length is different for sent corresponding to
            an affix cue.
        non_cue_sents (list[T]): sentences that does not contain negation.
    """
    raw_data = open(f_path)
    sentence = []
    labels = []
    label = []
    scope_sents = []
    or_sents = []
    data_scope = []
    scope = []
    scope_cues = []
    data = []
    non_cue_data = []
    num_cue_list = []
    cue_sep = []

    for line in raw_data:
        label = []
        sentence = []
        tokens = line.strip().split()
        afsent = []
        noncue_sep = []
        if len(tokens) == 8:  # This line has no cues
            num_cues = 0
            sentence.append(tokens[3])
            label.append(3)  # Not a cue
            noncue_sep.append(0)
            for line in raw_data:
                tokens = line.strip().split()
                if len(tokens) == 0:
                    break
                else:
                    sentence.append(tokens[3])
                    label.append(3)
                    noncue_sep.append(0)
            non_cue_data.append([sentence, label, noncue_sep])

        else:  # The line has 1 or more cues
            num_cues = (len(tokens) - 7) // 3
            affix_num = -1
            # cue_count+=num_cues
            scope = [[] for i in range(num_cues)]
            # First list is the real labels, second list is to modify
            # if it is a multi-word cue.
            label = [[], []]
            # Generally not a cue, if it is will be set ahead.
            label[0].append(3)
            label[1].append(-1)  # Since not a cue, for now.
            aflabel = [[],[]]
            aflabel[0].append(3)
            aflabel[1].append(-1)
            cue_counter = -1
            prev_cue_c = -1
            for i in range(num_cues):
                if tokens[7 + 3 * i] != '_':  # Cue field is active
                    if tokens[8 + 3 * i] != '_':  # Check for affix
                        label[0][-1] = 0  # Affix
                        #affix_list.append(tokens[7 + 3 * i]) # pylint: disable=undefined-variable
                        if i != prev_cue_c:
                            cue_counter += 1
                        prev_cue_c = i
                        label[1][-1] = cue_counter  # Cue number
                        aflabel[0][-1] = 0
                        aflabel[1][-1] = cue_counter
                        # sentence.append(tokens[7+3*i])
                        # new_word = '##'+tokens[8+3*i]
                    else:
                        # Maybe a normal or multiword cue. The next few
                        # words will determine which.
                        label[0][-1] = 1
                        aflabel[0][-1] = 1
                        # Which cue field, for multiword cue altering.
                        if i != prev_cue_c:
                            cue_counter += 1
                        prev_cue_c = i
                        label[1][-1] = cue_counter
                        aflabel[1][-1] = cue_counter

                if tokens[8 + 3 * i] != '_':
                    scope[i].append(1)
                else:
                    scope[i].append(2)
            sentence.append(tokens[3])
            afsent.append(tokens[3])
            for line in raw_data:
                tokens = line.strip().split()
                if len(tokens) == 0:
                    break
                else:
                    #sentence.append(tokens[3])
                    token = tokens[3]
                    affix_flag = False
                    # Generally not a cue, if it is will be set ahead.
                    label[0].append(3)
                    label[1].append(-1)  # Since not a cue, for now.
                    aflabel[0].append(3)
                    aflabel[1].append(-1)
                    for i in range(num_cues):
                        if tokens[7 + 3 *
                                    i] != '_':  # Cue field is active
                            if tokens[8 + 3 *
                                        i] != '_':  # Check for affix
                                label[0][-1] = 0  # Affix
                                aflabel[0][-1] = 0
                                aflabel[0].append(3)
                                if i != prev_cue_c:
                                    cue_counter += 1
                                prev_cue_c = i
                                label[1][-1] = cue_counter  # Cue number
                                aflabel[1][-1] = cue_counter
                                aflabel[1].append(-1)
                                affix_flag = True
                                affix_num = i
                                token = [tokens[3], tokens[7 + 3 * i], tokens[8 + 3 * i]]
                            else:
                                # Maybe a normal or multiword cue. The
                                # next few words will determine which.
                                label[0][-1] = 1
                                aflabel[0][-1] = 1
                                # Which cue field, for multiword cue
                                # altering.
                                if i != prev_cue_c:
                                    cue_counter += 1
                                prev_cue_c = i
                                label[1][-1] = cue_counter
                                aflabel[1][-1] = cue_counter
                                
                        if tokens[8 + 3 * i] != '_':
                            # Detected scope
                            if tokens[7 + 3 * i] != '_' and i == affix_num:
                                # Check if it is affix cue
                                scope[i].append(1)
                                if param.sherlock_seperate_affix:
                                    scope[i].append(1)
                            else:
                                scope[i].append(1)
                        else:
                            scope[i].append(2)
                    if affix_flag is False:
                        sentence.append(token)
                        afsent.append(token)
                    else:
                        sentence.append(token[0])
                        afsent.append(token[1])
                        afsent.append('<AFF>'+token[2])
            for i in range(num_cues):
                indices = []
                for index, j in enumerate(label[1]):
                    if i == j:
                        indices.append(index)
                count = len(indices)
                if count > 1:
                    # Multi word cue
                    for j in indices:
                        label[0][j] = 2
            
            sent_scopes = []
            sent_cues = []
            or_sents.append(sentence)
            scope_sent = []
            for i in range(num_cues):
                sc = []

                if affix_num == -1:
                    # No affix cue in this sent
                    scope_sent.append(sentence)

                    for a, b in zip(label[0], label[1]):
                        if i == b:
                            sc.append(a)
                        else:
                            sc.append(3)
                else:
                    if affix_num == i and param.sherlock_seperate_affix:
                        # Detect affix cue
                        scope_sent.append(afsent)

                        for a, b in zip(aflabel[0], aflabel[1]):
                            if i == b:
                                sc.append(a)
                            else:
                                sc.append(3)
                    else:
                        scope_sent.append(sentence)

                        for a, b in zip(label[0], label[1]):
                            if i == b:
                                sc.append(a)
                            else:
                                sc.append(3)
                sent_scopes.append(scope[i])
                sent_cues.append(sc)
            data_scope.append(sent_scopes)
            scope_cues.append(sent_cues)
            scope_sents.append(scope_sent)
            labels.append(label[0])
            data.append(sentence)
            num_cue_list.append(num_cues)
            cue_sep.append([e+1 for e in label[1]])

    new_cue_sents = []
    new_cue_cues = []
    new_cue_sep = []
    for i, sent in enumerate(data):
        if 'n\'t' in sent:
            t_sent = []
            t_cue = []
            t_sep = []
            for ii, word in enumerate(sent):
                if word == 'n\'t':
                    t_sent[-1] += word
                    # change the root's (do, can, is, etc.) cue label
                    # it's possible that even it comes with a n't but is not a cue
                    t_cue[-1] = labels[i][ii]
                    t_sep[-1] = cue_sep[i][ii]
                else:
                    t_sent.append(word)
                    t_cue.append(labels[i][ii])
                    t_sep.append(cue_sep[i][ii])
            new_cue_sents.append(t_sent)
            new_cue_cues.append(t_cue)
            new_cue_sep.append(t_sep)
        else:
            new_cue_sents.append(sent)
            new_cue_cues.append(labels[i])
            new_cue_sep.append(cue_sep[i])

    new_or_sents = []
    for i, sent in enumerate(or_sents):
        if 'n\'t' in sent:
            t_sent = []
            for word in sent:
                if word == 'n\'t':
                    t_sent[-1] += word
                else:
                    t_sent.append(word)
            new_or_sents.append(t_sent)
        else:
            new_or_sents.append(sent)
    
    new_scope_sents = []
    new_scope_cues = []
    new_scopes = []
    for i, wraps in enumerate(scope_sents):
        scope_sent_unit = []
        scope_cue_unit = []
        scope_unit = []
        for si, sent in enumerate(wraps):
            if 'n\'t' in sent:
                t_sent = []
                t_scue = []
                t_scope = []
                for wi, word in enumerate(sent):
                    if word == 'n\'t':
                        t_sent[-1] += word
                        t_scue[-1] = scope_cues[i][si][wi]
                        t_scope[-1] = data_scope[i][si][wi]
                    else:
                        t_sent.append(word)
                        t_scue.append(scope_cues[i][si][wi])
                        t_scope.append(data_scope[i][si][wi])
                scope_sent_unit.append(t_sent)
                scope_cue_unit.append(t_scue)
                scope_unit.append(t_scope)
            else:
                scope_sent_unit.append(sent)
                scope_cue_unit.append(scope_cues[i][si])
                scope_unit.append(data_scope[i][si])
        new_scope_sents.append(scope_sent_unit)
        new_scope_cues.append(scope_cue_unit)
        new_scopes.append(scope_unit)

    new_noncue_sents = []
    for i, sent in enumerate(non_cue_data):
        if 'n\'t' in sent[0]:
            t_sent = []
            for ii, word in enumerate(sent[0]):
                if word == 'n\'t':
                    t_sent[-1] += word
                else:
                    t_sent.append(word)
            new_noncue_sents.append(t_sent)
        else:
            new_noncue_sents.append(sent[0])

    if param.sherlock_combine_nt:
        non_cue_sents = new_noncue_sents
        non_cue_cues = [[3 for i in sent] for sent in new_noncue_sents]
        non_cue_sep = [[0 for i in sent] for sent in new_noncue_sents]
        non_cue_num = [0 for sent in new_noncue_sents]

        sherlock_cues = (new_cue_sents + non_cue_sents, new_cue_cues + non_cue_cues, new_cue_sep+non_cue_sep, num_cue_list+non_cue_num)
        sherlock_scopes = (new_or_sents, new_scope_sents, new_scope_cues, new_scopes)
    else:
        non_cue_sents = [i[0] for i in non_cue_data]
        non_cue_cues = [i[1] for i in non_cue_data]
        non_cue_sep = [i[2] for i in non_cue_data]
        non_cue_num = [0 for i in non_cue_data]

        sherlock_cues = (data + non_cue_sents, labels + non_cue_cues, cue_sep+non_cue_sep, num_cue_list+non_cue_num)
        sherlock_scopes = (or_sents, scope_sents, scope_cues, data_scope)
    
    return [sherlock_cues, sherlock_scopes, non_cue_sents]

def bioscope(f_path) -> Tuple[List, List, List]:
    """
    return: raw data format. (cues, scopes, non_cue_sents)
        cues (list[list[T]]): (sentences, cue_labels, cue_sep, num_cues)
            cue_sep is the seperation label of cues. 
            num_cues notes the number of cues in this sent.
        scopes (list[list[T]]): (original_sent, sentences[n], cue_labels[n], scope_labels[n])
            where n = number of cues in this sentence.
        non_cue_sents: sentences that does not contain negation.
    """
    file = open(f_path, encoding='utf-8')
    sentences = []
    for s in file:
        sentences += re.split("(<.*?>)", html.unescape(s))
    cue_sentence = []
    cue_cues = []
    non_cue_data = []
    scope_cues = []
    scope_scopes = []
    scope_sentences = []
    scope_orsents = []
    sentence = []
    cue = {}
    scope = {}
    in_scope = []
    in_cue = []
    word_num = 0
    c_idx = []
    s_idx = []
    in_sentence = 0
    num_cue_list = []
    cue_sep = []
    for token in sentences:
        if token == '':
            continue
        elif '<sentence' in token:
            in_sentence = 1
        elif '<cue' in token:
            if 'negation' in token:
                in_cue.append(
                    str(re.split('(ref=".*?")', token)[1][4:]))
                c_idx.append(
                    str(re.split('(ref=".*?")', token)[1][4:]))
                if c_idx[-1] not in cue.keys():
                    cue[c_idx[-1]] = []
        elif '</cue' in token:
            in_cue = in_cue[:-1]
        elif '<xcope' in token:
            # print(re.split('(id=".*?")',token)[1][3:])
            in_scope.append(str(re.split('(id=".*?")', token)[1][3:]))
            s_idx.append(str(re.split('(id=".*?")', token)[1][3:]))
            scope[s_idx[-1]] = []
        elif '</xcope' in token:
            in_scope = in_scope[:-1]
        elif '</sentence' in token:
            #print(cue, scope)
            if len(cue.keys()) == 0:
                # no cue in this sent
                non_cue_data.append([sentence, [3]*len(sentence), [0]*len(sentence)])
            else:
                cue_sentence.append(sentence)
                cue_cues.append([3]*len(sentence))
                cue_sep.append([0]*len(sentence))
                scope_sentence = []
                scope_subscope = []
                scope_subcues = []
                for count, i in enumerate(cue.keys()):
                    scope_sentence.append(sentence)
                    scope_subcues.append([3]*len(sentence))
                    if len(cue[i]) == 1:
                        cue_cues[-1][cue[i][0]] = 1
                        scope_subcues[-1][cue[i][0]] = 1
                        cue_sep[-1][cue[i][0]] = count + 1
                    else:
                        for c in cue[i]:
                            cue_cues[-1][c] = 2
                            scope_subcues[-1][c] = 2
                            cue_sep[-1][c] = count + 1
                    scope_subscope.append([2]*len(sentence))

                    if i in scope.keys():
                        for s in scope[i]:
                            scope_subscope[-1][s] = 1
                scope_orsents.append(sentence)
                scope_sentences.append(scope_sentence)
                scope_cues.append(scope_subcues)
                scope_scopes.append(scope_subscope)
            num_cue_list.append(len(cue.keys()))

            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            in_sentence = 0
            c_idx = []
            s_idx = []
        elif '<' not in token:
            if in_sentence == 1:
                words = token.split()
                sentence += words
                if len(in_cue) != 0:
                    for i in in_cue:
                        cue[i].extend([word_num + i for i in range(len(words))])
                elif len(in_scope) != 0:
                    for i in in_scope:
                        scope[i] += [word_num + i for i in range(len(words))]
                word_num += len(words)
    non_cue_sents = [i[0] for i in non_cue_data]
    non_cue_cues = [i[1] for i in non_cue_data]
    non_cue_sep = [i[2] for i in non_cue_data]
    non_cue_num = [0 for i in non_cue_data]
    if param.mark_cue:
        for ci, c in enumerate(scope_cues):
            for i, e in enumerate(c):
                if e == 0 or e == 1 or e == 2:
                    scope_scopes[ci][i] = 3
    return [(cue_sentence+non_cue_sents, cue_cues+non_cue_cues, cue_sep+non_cue_sep, num_cue_list+non_cue_num), 
            (scope_orsents, scope_sentences, scope_cues, scope_scopes),
            non_cue_sents]

def sfu_review(f_path) -> Tuple[List, List, List]:
    """
    return: raw data format. (cues, scopes, non_cue_sents)
        cues (list[list[T]]): (sentences, cue_labels)
        scopes (list[list[T]]): (original_sent, sentences[n], cue_labels[n], scope_labels[n])
            where n = number of cues in this sentence.
        non_cue_sents: sentences that does not contain negation.
    """
    file = open(f_path, encoding='utf-8')
    sentences = []
    for s in file:
        sentences += re.split("(<.*?>)", html.unescape(s))
    cue_sentence = []
    cue_cues = []
    scope_cues = []
    scope_scopes = []
    scope_sentence = []
    scope_sentences = []
    scope_orsents = []
    sentence = []
    cue = {}
    scope = {}
    in_scope = []
    in_cue = []
    word_num = 0
    c_idx = []
    non_cue_data = []
    s_idx = []
    in_word = 0
    num_cue_list = []
    cue_sep = []
    for token in sentences:
        if token == '':
            continue
        elif token == '<W>':
            in_word = 1
        elif token == '</W>':
            in_word = 0
            word_num += 1
        elif '<cue' in token:
            if 'negation' in token:
                in_cue.append(
                    int(re.split('(ID=".*?")', token)[1][4:-1]))
                c_idx.append(
                    int(re.split('(ID=".*?")', token)[1][4:-1]))
                if c_idx[-1] not in cue.keys():
                    cue[c_idx[-1]] = []
        elif '</cue' in token:
            in_cue = in_cue[:-1]
        elif '<xcope' in token:
            continue
        elif '</xcope' in token:
            in_scope = in_scope[:-1]
        elif '<ref' in token:
            in_scope.append([int(i) for i in re.split(
                '(SRC=".*?")', token)[1][5:-1].split(' ')])
            s_idx.append([int(i) for i in re.split(
                '(SRC=".*?")', token)[1][5:-1].split(' ')])
            for i in s_idx[-1]:
                scope[i] = []
        elif '</SENTENCE' in token:
            if len(cue.keys()) == 0:
                non_cue_data.append([sentence, [3]*len(sentence), [0]*len(sentence)])
            else:
                cue_sentence.append(sentence)
                cue_cues.append([3]*len(sentence))
                cue_sep.append([0]*len(sentence))
                scope_sentence = []
                scope_subscope = []
                scope_subcues = []
                for count, i in enumerate(cue.keys()):
                    scope_sentence.append(sentence)
                    scope_subcues.append([3]*len(sentence))
                    if len(cue[i]) == 1:
                        cue_cues[-1][cue[i][0]] = 1
                        scope_subcues[-1][cue[i][0]] = 1
                        cue_sep[-1][cue[i][0]] = count + 1
                    else:
                        for c in cue[i]:
                            cue_cues[-1][c] = 2
                            scope_subcues[-1][c] = 2
                            cue_sep[-1][c] = count + 1
                    scope_subscope.append([2]*len(sentence))
                    if i in scope.keys():
                        for s in scope[i]:
                            scope_subscope[-1][s] = 1
                scope_orsents.append(sentence)
                scope_sentences.append(scope_sentence)
                scope_cues.append(scope_subcues)
                scope_scopes.append(scope_subscope)
                num_cue_list.append(len(cue.keys()))
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            in_word = 0
            c_idx = []
            s_idx = []
        elif '<' not in token:
            if in_word == 1:
                if len(in_cue) != 0:
                    for i in in_cue:
                        cue[i].append(word_num)
                if len(in_scope) != 0:
                    for i in in_scope:
                        for j in i:
                            scope[j].append(word_num)
                sentence.append(token)
    non_cue_sents = [i[0] for i in non_cue_data]
    non_cue_cues = [i[1] for i in non_cue_data]
    non_cue_sep = [i[2] for i in non_cue_data]
    non_cue_num = [0 for i in non_cue_data]
    if param.mark_cue:
        for ci, c in enumerate(scope_cues):
            for i, e in enumerate(c):
                if e == 0 or e == 1 or e == 2:
                    scope_scopes[ci][i] = 3

    return [(cue_sentence+non_cue_sents, cue_cues+non_cue_cues, cue_sep+non_cue_sep, num_cue_list+non_cue_num), 
            (scope_orsents, scope_sentences, scope_cues, scope_scopes), non_cue_sents]



class SplitData():
    def __init__(self, cue, scope, non_cue_sents):
        if isinstance(cue, list):
            self.cues = self.combine_lists(cue)
        if isinstance(scope, list):
            self.scopes = self.combine_lists(scope)
        if isinstance(non_cue_sents, list):
            self.non_cue_sents = self.combine_lists(non_cue_sents)

    def combine_lists(self, input_: List):
        t = []
        for e in input_:
            t.extend(e)
        return t

class SplitMoreData():
    def __init__(self, cue, scope, non_cue_sents):
        if isinstance(cue, list):
            self.cues = self.combine_lists(cue)
        if isinstance(scope, list):
            self.scopes = self.combine_lists(scope)
        if isinstance(non_cue_sents, list):
            self.non_cue_sents = self.pack_(non_cue_sents)

    def pack_(self, input_):
        t = []
        for e in input_:
            t.extend(e)
        return t

    def combine_lists(self, input_: List[List]):
        tmp = input_[0]
        for i, e in enumerate(input_):
            if i == 0:
                continue
            for ii, elem in enumerate(e):
                tmp[ii].extend(elem)
        return tmp
