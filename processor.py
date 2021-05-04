import random
import math
import copy
import gc
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer
from transformers.file_utils import cached_path
import numpy as np
from typing import List, Tuple, T, Iterable, Union, NewType
import gensim
import _pickle

from util import pad_sequences, del_list_idx
from data import RawData
from params import param


class InputExample():
    def __init__(self, guid, sent: str, subword_mask=None):
        """
            sent: whitespace separated input sequence string
        """
        self.guid = guid
        self.sent = sent
        self.subword_mask = subword_mask


class CueExample(InputExample):
    def __init__(self, cues, cue_sep, num_cues, **kwargs):
        super().__init__(**kwargs)
        self.cues = cues
        self.num_cues = num_cues
        self.cue_sep = cue_sep


class ScopeExample(InputExample):
    def __init__(self, cues: List[int], scopes: List[T], sc_sent: List[str], num_cues=None, **kwargs):
        super().__init__(**kwargs)
        if num_cues is None:
            self.num_cues = len(scopes)
        else:
            self.num_cues = num_cues
        self.cues = cues
        self.scopes = scopes
        self.sc_sent = sc_sent

class SherScopeExample(ScopeExample):
    """deprecated"""
    def __init__(self, example):
        super().__init__(guid=example.guid, sent=example.sent, subword_mask=example.subword_mask,
                         cues=example.cues, scopes=example.scopes, sc_sent=example.sc_sent)
        self.cues, self.scopes, self.sc_sent = self.combine_nt(self.cues, self.scopes, self.sc_sent)
    
    def combine_nt(self, cues, scopes, sc_sent):
        if 'n\'t' in sc_sent[0]:
            num_cues = len(cues)
            new_cues = [[] for i in range(num_cues)]
            new_scopes = [[] for i in range(num_cues)]
            new_sc_sent = [[] for i in range(num_cues)]
            for i, (cue, scope, sent) in enumerate(zip(cues, scopes, sc_sent)):
                for c, s, token in zip(cue, scope, sent):
                    if token != 'n\'t':
                        new_cues[i].append(c)
                        new_scopes[i].append(s)
                        new_sc_sent[i].append(token)
                    else:
                        new_cues[i][-1] = c
                        new_scopes[i][-1] = s
                        new_sc_sent[i][-1] += token
            return new_cues, new_scopes, new_sc_sent
        else:
            return cues, scopes, sc_sent
    
    def mark_affix_scope(self, cues, scopes):
        new_scopes = scopes.copy()
        for i, (cue, scope) in enumerate(zip(cues, scopes)):
            if 4 in cue:
                new_scope = scope.copy()
                for j, c in enumerate(cue):
                    if c == 4:
                        new_scope[j] = 1
                new_scopes[i] = new_scope
        return new_scopes
            


ExampleLike = Union[CueExample, ScopeExample, InputExample, SherScopeExample]


class CueFeature():
    def __init__(self, guid, sent, input_ids, padding_mask, subword_mask, input_len, cues, cue_sep, num_cues):
        self.guid = guid
        self.sent = sent
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.cue_sep = cue_sep
        self.num_cues = num_cues


class ScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len,
                 cues, scopes, num_cues):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.scopes = scopes
        self.num_cues = num_cues

class MultiScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len, 
                 cues, scopes, num_cues, master_mat=None):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.num_cues = num_cues
        (self.cues, self.scopes), (self.master_input_ids, self.master_subword_masks, self.master_padding_masks) = \
            mark_all_cues(cues, scopes, input_len, (input_ids, subword_mask, padding_mask))
        self.master_mat = master_mat

class PipelineScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len, cues, cue_match, gold_scopes, gold_num_cues):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.cue_match = cue_match
        self.gold_scopes = gold_scopes
        self.gold_num_cues = gold_num_cues

FeatureLike = Union[CueFeature, ScopeFeature, PipelineScopeFeature, MultiScopeFeature]

class MultiScopeDataset(Dataset):
    def __init__(self, input_ids, master_padding_masks, scopes, input_len, cues, master_subword_masks, master_input_ids, master_mat):
        self.input_ids = input_ids
        self.master_padding_masks = master_padding_masks
        self.scopes = scopes
        self.input_len = input_len
        self.cues = cues
        self.master_subword_masks = master_subword_masks
        self.master_input_ids = master_input_ids
        self.master_mat = master_mat

    def __getitem__(self, index):
        return self.input_ids[index], self.master_padding_masks[index], self.scopes[index], self.input_len[index], \
               self.cues[index], self.master_subword_masks[index], self.master_input_ids[index], self.master_mat[index]

    def __len__(self):
        return self.input_ids.size(0)

bioes_to_id = {'<PAD>': 0, 'I': 1, 'O': 2, 'B': 3, 'E': 4, 'S': 5, 'C': 6}

def scope_to_bioes(scope):
    tmp = [''] * len(scope)
    sflag = False
    for i, e in enumerate(scope):
        if i > 0 and i < len(scope)-1:
            if i == 1:
                if sflag is True and e in [1, 6]:
                    tmp[i-1] = 'B'
                    tmp[i] = 'I'
                    sflag = False
                elif sflag is True and e not in [1, 6]:
                    tmp[i-1] = 'S'
                    tmp[i] = 'O'
                    sflag = False
                else:
                    tmp[i-1] = 'O'
                    if e in [1, 6]:
                        sflag = True
                    else:
                        tmp[i] = 'O'
            else:
                # 1< i < len-1
                if e in [1, 6] and prev in [1, 6]:
                    tmp[i] = 'I'
                    if sflag:
                        tmp[i-1] = 'B'
                        sflag = False
                    sflag = False
                else:
                    if e not in [1, 6]:
                        if prev in [1, 6] and sflag is True:
                            # prev is last scope token in subscope, start flag False
                            tmp[i-1] = 'S'
                            sflag = False
                        elif prev in [1, 6] and sflag is False:
                            tmp[i-1] = 'E'
                        tmp[i] = 'O'
                    elif e in [1, 6] and prev not in [1, 6]:
                        # when e != prev there is a change in label sequence
                        # e is scope token, i is start pos of subscope
                        sflag = True
                    elif e == 2:
                        tmp[i] = 'O'
            prev = e
        elif i == 0:
            prev = e
            if e == 2:
                tmp[i] = 'O'
            else:
                sflag = True
        else:
            # at end of sentence
            if e in [1, 6]:
                # when last token is scope, if prev != e: single subscope. 
                # elif prev == e: extention of subscope to end of sent.
                # hence append the end pos anyway
                if prev in [1, 6]:
                    if sflag:
                        tmp[i-1] = 'B'
                    tmp[i] = 'E'
                else:
                    # single subscope
                    tmp[i] = 'S'
                    tmp[i-1] = 'O'
            else:
                tmp[i] = 'O'
                if prev in [1, 6]:
                    # prev is end of subscope
                    if sflag:
                        tmp[i-1] = 'S'
                    else:
                        tmp[i-1] = 'E'
                else:
                    tmp[i-1] = 'O'
    ids = [bioes_to_id[l] for l in tmp]
    return ids

def single_scope_to_link_matrix_pad(scope: List, cues: List, input_len: int,
                                    mode=param.m_dir, cue_mode=param.cue_mode) -> torch.LongTensor:
    """
    To convert the scope list (single cue) to a link matrix that represents
    the relation (undirected) link between eachother token.
    cue_mode diag:
        Cue <-> Scope: 1
        Noncue <-> Noncue: 2
        Cue (ROOT) <-> Cue: 3
        Pad: 0
    
    cue_mode root:
        Outward link: 1 (cue -> scope, ROOT -> cue)
        Not linked / inward link: 2 (scope -> cue, scope -> scope)
        Pad: 0

    params:
        mode: ['d1', 'd2', 'ud'] d for directed, ud for undirected
        cue_mode: ['root', 'diag'] 'root' for adding an additional dummy [ROOT] token
        to indicate the link between [ROOT] and cue, avoiding labelling diagonal element of matrix
            'diag' for labelling diagonal
    """
    temp_scope = []
    for i, e in enumerate(scope):
        if e == 2:
            if cues[i] != 3:
                temp_scope.append(3)
            else:
                temp_scope.append(e)
        else:
            temp_scope.append(e)
    mat_dim = param.max_len
    mat = np.zeros((mat_dim, mat_dim), dtype=np.int)
    if cue_mode == 'root':
        # for root mode, force the direction to be single directed (d2)
        for i in range(input_len):
            # scan through the matrix by row to fill
            if scope[i] == 3:
                # The row at cue
                mat[0][i] = 1
                mat[i][0] = 2
                for j in range(input_len):
                    if scope[j] != 3:
                        mat[i][j] = scope[j]
                    else:
                        mat[i][j] = 2
            else:
                mat[0][i] = 2
                for j in range(input_len):
                    if scope[j] == 3:
                        if scope[i] == 1:
                            mat[i][j] = 2
                        else:
                            mat[i][j] = scope[i]
                    else:
                        mat[i][j] = 2
            mat[i][i] = 2
    elif cue_mode == 'diag':
        # scan through the matrix by row to fill
        for i in range(input_len):
            if scope[i] == 3:
                # The row at cue
                for j in range(input_len):
                    mat[i][j] = scope[j]
            else:
                for j in range(input_len):
                    if scope[j] == 3:
                        if scope[i] == 1 and mode == 'd1':
                            mat[i][j] = 4
                        elif scope[i] == 1 and mode == 'd2':
                            mat[i][j] = 2
                        else:
                            mat[i][j] = scope[i]
                    else:
                        mat[i][j] = 2
    mat = torch.LongTensor(mat)
    return mat

def mark_all_cues(cues, scopes, input_lens, inputs=None, neg_token_id=1):
    """
    inputs (input_ids, subword_masks, padding_masks)
    """
    num_cues = len(cues)
    cue_pos = []
    for cue in cues:
        tmp = []
        for i, e in enumerate(cue):
            if e != 3:
                tmp.append(i)
        cue_pos.append(tmp)
    all_cue_pos = cue_pos.copy()
    # Modify the cues and scopes sequences to fit for augmentation of all cues
    all_cue_scopes = []
    all_cue_cues = []

    for i in range(num_cues):
        if i == 0:
            continue
        # shift following cues position by 2 due to the special tokens
        # (i > 1)
        tmp = [e + 2*i for e in cue_pos[i]]
        all_cue_pos[i] = tmp
    
    for i in range(num_cues):
        stmp = scopes[i].copy()
        ctmp = cues[i].copy()
        for j in range(num_cues):
            if j == i:
                continue
            if j > i:
                c = 1
            else:
                c = 0
            left = np.min(all_cue_pos[j])
            right = np.max(all_cue_pos[j])
            cue_ = np.min(cue_pos[j]) + 2 * c
            stmp.insert(left, scopes[i][cue_])
            stmp.insert(right, scopes[i][cue_])
            ctmp.insert(left, cues[i][cue_])
            ctmp.insert(right, cues[i][cue_])
        all_cue_scopes.append(stmp)
        all_cue_cues.append(ctmp)
    
    if inputs is not None:
        input_ids = inputs[0].copy()[0]
        subword_masks = inputs[1].copy()[0]
        padding_masks = inputs[2].copy()[0]
        for i in range(1, num_cues):
            left = np.min(all_cue_pos[i])
            right = np.max(all_cue_pos[i])
            input_ids.insert(left, neg_token_id)
            input_ids.insert(right, neg_token_id)
            subword_masks.insert(left, 1)
            subword_masks[left+1] = 0
            subword_masks.insert(right, 0)
            padding_masks.insert(left, 1)
            padding_masks.insert(right, 1)
        return (all_cue_cues, all_cue_scopes), (input_ids, subword_masks, padding_masks)
    else:
        return (all_cue_cues, all_cue_scopes), None

def multi_scope_to_link_matrix_pad(scopes: List[List], cues: List[List], input_lens: List[int]) -> np.ndarray:
    """
    To convert the scope list (single cue) to a link matrix that represents
    the directed relation link between eachother token.
    Outward link: 1 (cue -> scope)
        Not linked / inward link: 2 (scope -> cue, scope -> scope)
        3 (cue -> cue)
        Pad: 0
    params:
        similar to single case, except cue_mode is fixed to 'diag'
    """
    num_cues = len(cues)
    if num_cues == 1:
        return single_scope_to_link_matrix_pad(scopes[0], cues[0], input_lens[0], 'd2')
    else:
        (all_cue_cues, all_cue_scopes), _ = mark_all_cues(cues, scopes, input_lens, None)

        mat_dim = param.max_len
        all_sub_matrix = []
        for c in range(num_cues):
            mat = np.zeros((mat_dim, mat_dim), dtype=np.int)
            full_seq_len = len(all_cue_scopes[c])
            """
            # for root mode, force the direction to be single directed (d2)
            for i in range(full_seq_len):
                # scan through the matrix by row to fill
                if all_cue_scopes[c][i] == 3:
                    # The row at cue
                    mat[0][i] = 1
                    mat[i][0] = 2
                    for j in range(full_seq_len):
                        if all_cue_scopes[c][j] != 3:
                            mat[i][j] = all_cue_scopes[c][j]
                        else:
                            mat[i][j] = 2
                else:
                    mat[0][i] = 2
                    for j in range(full_seq_len):
                        if all_cue_scopes[c][j] == 3:
                            if all_cue_scopes[c][i] == 1:
                                mat[i][j] = 2
                            else:
                                mat[i][j] = all_cue_scopes[c][i]
                        else:
                            mat[i][j] = 2
                mat[i][i] = 0
            all_sub_matrix.append(mat)"""

            for i in range(full_seq_len):
                if all_cue_scopes[c][i] == 3:
                    # The row at cue
                    for j in range(full_seq_len):
                        mat[i][j] = all_cue_scopes[c][j]
                else:
                    for j in range(full_seq_len):
                        if all_cue_scopes[c][j] == 3:
                            if all_cue_scopes[c][i] == 1:
                                mat[i][j] = 2
                            else:
                                mat[i][j] = all_cue_scopes[c][i]
                        else:
                            mat[i][j] = 2
            all_sub_matrix.append(mat)
        master_mat = np.zeros((mat_dim, mat_dim), dtype=np.int)
        for i in range(len(all_cue_scopes[c])):
            for j in range(len(all_cue_scopes[c])):
                master_mat[i][j] = 2
        for m in all_sub_matrix:
            for i in range(len(all_cue_scopes[c])):
                for j in range(len(all_cue_scopes[c])):
                    if m[i][j] == 1:
                        master_mat[i][j] = 1
                    if m[i][j] == 3:
                        master_mat[i][j] = 3
        master_mat = torch.LongTensor(master_mat)
    return master_mat

class Processor(object):
    def __init__(self):
        self.tokenizer = None

    @classmethod
    def read_data(cls, input_file, dataset_name=None) -> RawData:
        return RawData(input_file, dataset_name=dataset_name)

    def create_examples(self, data: RawData, example_type: str, dataset_name: str, cue_or_scope: str, cached_file=None,
                        test_size=0.15, val_size=0.15) -> Union[List[ExampleLike], Tuple[List, List, List]]:
        """
        Create packed example format for input data. Do train-test split if specified.

        params:
            data (RawData): Though it's not "raw". Already contains tag information
            example_type (str): "train", "test", "dev", "split". If set as split, 
                will perform train-test split as well. 
            cue_or_scope (str): cue or scope.
            cached_file (NoneType | str): if specified, save the packed examples to cache file.
        returns:
            examples (list[ExampleLike]): example of specified type
            examples (cues, scopes)>>>(tuple[tuple[train, dev, test], tuple[train, dev, test]]): overload for split.
        """
        assert example_type.lower() in [
            'train', 'test', 'dev', 'split', 'joint_train', 'joint_dev', 'joint_test', 'joint'], 'Wrong example type.'
        assert cue_or_scope in [
            'cue', 'scope', 'raw'], 'cue_or_scope: Must specify cue of scope, or raw to perform split and get vocab'

        cue_examples = []
        non_cue_examples = []
        non_cue_scopes = []
        scope_examples = []
        for i, _ in enumerate(data.cues[0]):
            guid = '%s-%d' % (example_type, i)
            sentence = data.cues[0][i]
            cues = data.cues[1][i]
            cue_sep = data.cues[2][i]
            num_cues = data.cues[3][i]
            sent = ' '.join(sentence)
            if num_cues > 0:
                cue_examples.append(CueExample(guid=guid, sent=sent, cues=cues,
                                            cue_sep=cue_sep, num_cues=num_cues, subword_mask=None))
            else:
                non_cue_examples.append(CueExample(guid=guid, sent=sent, cues=cues,
                                            cue_sep=cue_sep, num_cues=num_cues, subword_mask=None))
                non_cue_scopes.append(ScopeExample(guid=guid, sent=sent, cues=[cues], num_cues=0,
                                            scopes=[[2 for c in cues]], sc_sent=[sentence], subword_mask=None))

        for i, _ in enumerate(data.scopes[0]):
            guid = '%s-%d' % (example_type, i)
            or_sent = data.scopes[0][i]
            sentence = data.scopes[1][i]
            cues = data.scopes[2][i]
            sent = ' '.join(or_sent)
            scopes = data.scopes[3][i]
            scope_examples.append(ScopeExample(guid=guid, sent=sent, cues=cues,
                                               scopes=scopes, sc_sent=sentence, subword_mask=None))

        if example_type.lower() in ('train', 'test', 'dev'):
            if cue_or_scope.lower() == 'cue':
                if cached_file is not None:
                    print('Saving examples into cached file %s', cached_file)
                    torch.save(cue_examples, cached_file)
                return cue_examples
            elif cue_or_scope.lower() == 'scope':
                if cached_file is not None:
                    print('Saving examples into cached file %s', cached_file)
                    torch.save(scope_examples, cached_file)
                return scope_examples
        elif example_type.lower() == 'split':
            # Note: if set example type to split, will return both cue and scope examples
            
            scope_len = len(scope_examples)
            train_len = math.floor((1 - test_size - val_size) * scope_len)
            test_len = math.floor(test_size * scope_len)
            val_len = scope_len - train_len - test_len
            scope_index = list(range(scope_len))
            train_index = random.sample(scope_index, k=train_len)
            scope_index = del_list_idx(scope_index, train_index)
            test_index = random.sample(scope_index, k=test_len)
            scope_index = del_list_idx(scope_index, test_index)
            dev_index = scope_index.copy()

            train_cue = [cue_examples[i] for i in train_index]
            test_cue = [cue_examples[i] for i in test_index]
            dev_cue = [cue_examples[i] for i in dev_index]
            train_scope = [scope_examples[i] for i in train_index]
            test_scope = [scope_examples[i] for i in test_index]
            dev_scope = [scope_examples[i] for i in dev_index]

            random_state = np.random.randint(1, 2020)
            tr_nocue_, te_nocue = train_test_split(
                non_cue_examples, test_size=test_size, random_state=random_state)
            _, te_non_cue_sents = train_test_split(
                data.non_cue_sents, test_size=test_size, random_state=random_state)
            random_state2 = np.random.randint(1, 2020)
            tr_nocue, de_nocue = train_test_split(tr_nocue_, test_size=(
                val_size / (1 - test_size)), random_state=random_state2)
            train_cue.extend(tr_nocue)
            dev_cue.extend(de_nocue)
            test_cue.extend(te_nocue)
            for c, _ in enumerate(train_cue):
                train_cue[c].guid = f'train-{c}'
            for c, _ in enumerate(test_cue):
                test_cue[c].guid = f'test-{c}'
            for c, _ in enumerate(dev_cue):
                dev_cue[c].guid = f'dev-{c}'
            for c, _ in enumerate(train_scope):
                train_scope[c].guid = f'train-{c}'
            for c, _ in enumerate(test_scope):
                test_scope[c].guid = f'test-{c}'
            for c, _ in enumerate(dev_scope):
                dev_scope[c].guid = f'dev-{c}'
            if cached_file is not None:
                print('Saving examples into cached file %s', cached_file)
                torch.save(train_cue, f'{param.base_path}/split/train_cue_{cached_file}')
                torch.save(test_cue, f'{param.base_path}/split/test_cue_{cached_file}')
                torch.save(dev_cue, f'{param.base_path}/split/dev_cue_{cached_file}')
                torch.save(train_scope, f'{param.base_path}/split/train_scope_{cached_file}')
                torch.save(test_scope, f'{param.base_path}/split/test_scope_{cached_file}')
                torch.save(dev_scope, f'{param.base_path}/split/dev_scope_{cached_file}')
                torch.save(te_non_cue_sents, f'{param.base_path}/split/ns_{cached_file}')
            return (train_cue, dev_cue, test_cue), (train_scope, dev_scope, test_scope)
        elif 'joint' in example_type.lower():
            if dataset_name.lower() == 'sherlock':
                if cue_or_scope.lower() == 'cue':
                    cue_examples.extend(non_cue_examples)
                    return cue_examples
                elif cue_or_scope.lower() == 'scope':
                    scope_examples.extend(non_cue_scopes)
                    return scope_examples
            else:
                scope_len = len(scope_examples)
                train_len = math.floor((1 - test_size - val_size) * scope_len)
                test_len = math.floor(test_size * scope_len)
                val_len = scope_len - train_len - test_len
                scope_index = list(range(scope_len))
                train_index = random.sample(scope_index, k=train_len)
                scope_index = del_list_idx(scope_index, train_index)
                test_index = random.sample(scope_index, k=test_len)
                scope_index = del_list_idx(scope_index, test_index)
                dev_index = scope_index.copy()

                train_cue = [cue_examples[i] for i in train_index]
                test_cue = [cue_examples[i] for i in test_index]
                dev_cue = [cue_examples[i] for i in dev_index]
                train_scope = [scope_examples[i] for i in train_index]
                test_scope = [scope_examples[i] for i in test_index]
                dev_scope = [scope_examples[i] for i in dev_index]

                random_state = np.random.randint(1, 2020)
                tr_nocue_, te_nocue = train_test_split(
                    non_cue_examples, test_size=test_size, random_state=random_state)
                random_state2 = np.random.randint(1, 2020)
                tr_nocue, de_nocue = train_test_split(tr_nocue_, test_size=(
                    val_size / (1 - test_size)), random_state=random_state2)

                tr_nocue_s, te_nocue_s = train_test_split(
                    non_cue_scopes, test_size=test_size, random_state=random_state)
                tr_nocue_s, de_nocue_s = train_test_split(tr_nocue_, test_size=(
                    val_size / (1 - test_size)), random_state=random_state2)
                train_cue.extend(tr_nocue)
                dev_cue.extend(de_nocue)
                test_cue.extend(te_nocue)
                train_scope.extend(tr_nocue_s)
                dev_scope.extend(de_nocue_s)
                test_scope.extend(te_nocue_s)
                for c, _ in enumerate(train_cue):
                    train_cue[c].guid = f'train-{c}'
                for c, _ in enumerate(test_cue):
                    test_cue[c].guid = f'test-{c}'
                for c, _ in enumerate(dev_cue):
                    dev_cue[c].guid = f'dev-{c}'
                for c, _ in enumerate(train_scope):
                    train_scope[c].guid = f'train-{c}'
                for c, _ in enumerate(test_scope):
                    test_scope[c].guid = f'test-{c}'
                for c, _ in enumerate(dev_scope):
                    dev_scope[c].guid = f'dev-{c}'
                if cached_file is not None:
                    print('Saving examples into cached file %s', cached_file)
                    torch.save(train_cue, f'{param.base_path}/split/joint_train_cue_{cached_file}')
                    torch.save(test_cue, f'{param.base_path}/split/joint_test_cue_{cached_file}')
                    torch.save(dev_cue, f'{param.base_path}/split/joint_dev_cue_{cached_file}')
                    torch.save(train_scope, f'{param.base_path}/split/joint_train_scope_{cached_file}')
                    torch.save(test_scope, f'{param.base_path}/split/joint_test_scope_{cached_file}')
                    torch.save(dev_scope, f'{param.base_path}/split/joint_dev_scope_{cached_file}')
                return (train_cue, dev_cue, test_cue), (train_scope, dev_scope, test_scope)

    def combine_sher_ex(self, data: List[ExampleLike]) -> List[ExampleLike]:
        tmp_data = []
        for item in data:
            tmp_data.append(SherScopeExample(item))
        return tmp_data

    def load_examples(self, file: str):
        """
        Load a pre-saved example binary file. Or anything else.

        Warning: torch.load() uses pickle module implicitly, which is known to be insecure. 
        It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.
        Never load data that could have come from an untrusted source, or that could have been tampered with.
        Only load data you trust.
        """
        return torch.load(file)
    
    def create_feature_from_example(self, example: ExampleLike, cue_or_scope: str,
                                    max_seq_len: int = 128, is_bert=False) -> List[FeatureLike]:
        """
        Tokenize, convert (sub)words to ids, and augmentation for cues
        """
        features = []
        if cue_or_scope == 'scope':
            guid = example.guid
            wrap_input_id = []
            wrap_subword_mask = []
            wrap_sents = []
            wrap_cues = []
            wrap_scopes = []
            wrap_padding_mask = []
            wrap_input_len = []
            for c in range(example.num_cues):
                sent = example.sc_sent[c]
                cues = example.cues[c]
                scopes = example.scopes[c]

                if is_bert:
                    # For BERT model
                    temp_sent = []
                    temp_scope = []
                    temp_cues = []
                    temp_mask = []
                    for word, cue, scope in zip(sent, cues, scopes):
                        subwords = self.tokenizer.tokenize(word)
                        for count, subword in enumerate(subwords):
                            mask = 1
                            if count > 0:
                                mask = 0
                            temp_mask.append(mask)
                            if param.ignore_multiword_cue:
                                if cue == 2:
                                    cue = 1
                            temp_cues.append(cue)
                            temp_scope.append(scope)
                            temp_sent.append(subword)
                    new_text = []
                    new_cues = []
                    new_scopes = []
                    new_masks = []
                    pos = 0
                    prev = 3
                    for token, cue, label, mask in zip(temp_sent, temp_cues,
                                                            temp_scope, temp_mask):
                        # Process the cue augmentation.
                        # Different from the original repo, the strategy is indicate the cue border
                        if param.augment_cue:
                            if cue != 3:
                                if pos != 0 and pos != len(temp_sent)-1:
                                    if cue == prev:
                                        # continued cue, don't care is subword or multi word cue
                                        new_masks.append(mask)
                                        new_text.append(token)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                    else:
                                        # left bound
                                        new_text.append('[unused1]')
                                        new_cues.append(cue)
                                        new_masks.append(1)
                                        new_scopes.append(label)
                                        new_text.append(token)
                                        new_masks.append(0)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                elif pos == 0:
                                    # at pos 0
                                    new_text.append('[unused1]')
                                    new_masks.append(1)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_text.append(token)
                                    new_masks.append(0)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                else:
                                    # at eos
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_text.append('[unused1]')
                                    new_masks.append(0)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                            else:
                                if cue == prev:
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                else:
                                    # current non cue, insert right bound before current pos
                                    new_text.append('[unused1]')
                                    new_masks.append(0)
                                    if param.mark_cue:
                                        new_scopes.append(3)
                                    else:
                                        new_scopes.append(2)
                                    new_cues.append(prev)
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                            prev = cue
                            pos += 1
                        else:
                            new_masks.append(mask)
                            new_text.append(token)
                            new_scopes.append(label)
                            new_cues.append(cue)

                    if len(new_text) >= max_seq_len - 1:
                        new_text = new_text[0:(max_seq_len - 2)]
                        new_cues = new_cues[0:(max_seq_len - 2)]
                        new_masks = new_masks[0:(max_seq_len - 2)]
                        new_scopes = new_scopes[0:(max_seq_len - 2)]
                    new_text.insert(0, '[CLS]')
                    new_text.append('[SEP]')
                    new_masks.insert(0, 1)
                    new_masks.append(1)
                    new_cues.insert(0, 3)
                    new_cues.append(3)
                    new_scopes.insert(0, 2)
                    new_scopes.append(2)
                    if param.cue_mode == 'root':
                        new_text.insert(0, '[ROOT]')
                        new_masks.insert(0, 1)
                        new_cues.insert(0, 3)
                        new_scopes.insert(0, 2)
                    input_ids = self.tokenizer.convert_tokens_to_ids(
                        new_text)
                    padding_mask = [1] * len(input_ids)
                    input_len = len(input_ids)
                    wrap_sents.append(new_text)
                    wrap_input_id.append(input_ids)
                    wrap_subword_mask.append(new_masks)
                    wrap_cues.append(new_cues)
                    wrap_scopes.append(new_scopes)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)
                else:
                    # For non-BERT (non-BPE tokenization)
                    sent = example.sc_sent[c].copy()
                    cues = example.cues[c].copy()
                    if param.ignore_multiword_cue:
                        for i, c in enumerate(cues):
                            if c == 2:
                                cues[i] = 1
                    scopes = example.scopes[c].copy()

                    words = self.tokenizer.tokenize(sent)
                    words.insert(0, '[CLS]')
                    words.append('[SEP]')
                    cues.insert(0, 3)
                    cues.append(3)
                    scopes.insert(0, 2)
                    scopes.append(2)
                    input_ids = self.tokenizer.convert_tokens_to_ids(words)
                    padding_mask = [1] * len(input_ids)
                    input_len = len(input_ids)
                    wrap_sents.append(words)
                    wrap_input_id.append(input_ids)
                    wrap_subword_mask = None
                    wrap_cues.append(cues)
                    wrap_scopes.append(scopes)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)
                    #assert all(each_len == len(words) for each_len in seq_len)

            feature = ScopeFeature(guid=guid, or_sent=example.sent, sents=wrap_sents, input_ids=wrap_input_id,
                                    padding_mask=wrap_padding_mask, subword_mask=wrap_subword_mask,
                                    input_len=wrap_input_len, cues=wrap_cues, scopes=wrap_scopes, num_cues=example.num_cues)
            features.append(feature)
        else:
            ## Cue
            sent = example.sent.split()
            guid = example.guid
            num_cues = example.num_cues
            if is_bert:
                # For BERT model
                new_text = []
                new_cues = []
                new_cuesep = []
                subword_mask = []
                for word, cue, sep in zip(sent, example.cues, example.cue_sep):
                    subwords = self.tokenizer.tokenize(word)
                    for i, subword in enumerate(subwords):
                        mask = 1
                        if i > 0:
                            mask = 0
                        subword_mask.append(mask)
                        if param.ignore_multiword_cue:
                            if cue == 2:
                                cue = 1
                        new_cues.append(cue)
                        new_cuesep.append(sep)
                        new_text.append(subword)
                if len(new_text) >= max_seq_len - 1:
                    new_text = new_text[0:(max_seq_len - 2)]
                    new_cues = new_cues[0:(max_seq_len - 2)]
                    new_cuesep = new_cuesep[0:(max_seq_len - 2)]
                    subword_mask = subword_mask[0:(max_seq_len - 2)]
                new_text.insert(0, '[CLS]')
                new_text.append('[SEP]')
                subword_mask.insert(0, 1)
                subword_mask.append(1)
                new_cues.insert(0, 3)
                new_cues.append(3)
                new_cuesep.insert(0, 0)
                new_cuesep.append(0)
                input_ids = self.tokenizer.convert_tokens_to_ids(new_text)
                padding_mask = [1] * len(input_ids)
                input_len = len(input_ids)
                feature = CueFeature(guid=guid, sent=sent, input_ids=input_ids,
                                        padding_mask=padding_mask, subword_mask=subword_mask,
                                        input_len=input_len, cues=new_cues, cue_sep=new_cuesep, num_cues=num_cues)
            else:
                # For non-BERT (non-BPE tokenization)
                cues = example.cues
                if param.ignore_multiword_cue:
                    for i, c in enumerate(cues):
                        if c == 2:
                            cues[i] = 1
                cues.insert(0, 3)
                cues.append(3)
                cue_sep = example.cue_sep
                cue_sep.insert(0, 0)
                cue_sep.append(0)
                words = self.tokenizer.tokenize(sent)
                words.insert(0, '[CLS]')
                words.append('[SEP]')
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                padding_mask = [1] * len(input_ids)
                input_len = len(input_ids)
                feature = CueFeature(guid=guid, sent=sent, input_ids=input_ids,
                                        padding_mask=padding_mask, subword_mask=None,
                                        input_len=input_len, cues=cues, cue_sep=cue_sep, num_cues=num_cues)
            features.append(feature)

        return features


    def create_features(self, data: List[ExampleLike], cue_or_scope: str,
                        max_seq_len: int = 128, is_bert=False) -> List[Union[CueFeature, ScopeFeature]]:
        """
        Create packed 
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for example in data:
            f = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)
            if param.ignore_multi_negation:
                # For joint prediction, forced ignore multiple negation sentences
                if f[0].num_cues > 1:
                    continue
            if param.task == 'pipeline':
                if f[0].num_cues == 0:
                    continue
            features.append(f[0])
        return features
    
    def create_features_multi(self, data: List[ExampleLike], cue_or_scope: str,
                              max_seq_len: int = 128, is_bert=False) -> List[Union[CueFeature, ScopeFeature]]:
        """
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for example in data:
            num_cues = example.num_cues
            if num_cues == 1:
                tmp = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)[0]
                mat = self.single_scope_to_matrix(tmp.scopes[0], tmp.cues[0], len(tmp.scopes[0]))
                feature = MultiScopeFeature(tmp.guid, tmp.or_sent, tmp.sents, tmp.input_ids, tmp.padding_mask,
                                            tmp.subword_mask, tmp.input_len, tmp.cues, tmp.scopes, num_cues, mat)
                features.append(feature)
            else:
                tmp = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)[0]
                input_lens = [len(e) for e in tmp.cues]
                combined_matrix = self.multi_scopes_to_matrix(tmp.scopes, tmp.cues, input_lens, num_cues)
                feature = MultiScopeFeature(tmp.guid, tmp.or_sent, tmp.sents, tmp.input_ids, tmp.padding_mask,
                                            tmp.subword_mask, input_lens, tmp.cues, tmp.scopes, num_cues, combined_matrix)
                features.append(feature)
        return features
        


    def create_features_pipeline(self, cue_input: List[CueFeature], scope_input: List[ScopeFeature], cue_model, 
                                 max_seq_len: int, is_bert=False, non_cue_examples=None):
        """
        Create scope feature for pipeline TESTING. The cue info was predicted with a trained cue 
        model, instead of the golden cues. Training of scope model will be normal scope model, with golden cue input.
        calling cue_model returns the logits of cue label and the separation.
        It shouldn't be batched input as the output batch size is not controlable, due to different number of cues
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for counter, cue_ex in enumerate(cue_input):
            wrap_sents = []
            wrap_input_id = []
            wrap_subword_mask = []
            wrap_cues = []
            wrap_padding_mask = []
            wrap_input_len = []
            sent = cue_ex.sent
            tmp_mask = []
            gold_nc = np.max(cue_ex.cue_sep)
            if is_bert:
                tmp_text = []
                tmp_cue = []
                tmp_sep = []
                sent_list = sent.split(' ')
                for word, cue, sep in zip(sent_list, cue_ex.cues, cue_ex.cue_sep):
                    subwords = self.tokenizer.tokenize(word)
                    for i, subword in enumerate(subwords):
                        mask = 1
                        if i > 0:
                            mask = 0
                        tmp_mask.append(mask)
                        tmp_cue.append(cue)
                        tmp_text.append(subword)
                        tmp_sep.append(sep)
                if len(tmp_text) >= max_seq_len - 1:
                    tmp_text = tmp_text[0:(max_seq_len - 2)]
                    tmp_mask = tmp_mask[0:(max_seq_len - 2)]
                    tmp_cue = tmp_cue[0:(max_seq_len - 2)]
                    tmp_sep = tmp_sep[0:(max_seq_len - 2)]
                tmp_text.insert(0, '[CLS]')
                tmp_text.append('[SEP]')
                tmp_mask.insert(0, 1)
                tmp_mask.append(1)
                tmp_cue.insert(0, 3)
                tmp_cue.append(3)
                tmp_sep.insert(0, 0)
                tmp_sep.append(0)
                tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_text)

                tmp_pad_mask = [1] * len(tmp_input_ids)
                tmp_input_lens = len(tmp_input_ids)
                tmp_pad_mask_in = tmp_pad_mask.copy()
                while len(tmp_input_ids) < max_seq_len:
                    tmp_input_ids.append(0)
                    tmp_pad_mask_in.append(0)
                    tmp_mask.append(0)
                    tmp_cue.append(0)
                    tmp_sep.append(0)
                tmp_input_ids = torch.LongTensor(tmp_input_ids).unsqueeze(0).cuda()
                tmp_pad_mask_in = torch.LongTensor(tmp_pad_mask_in).unsqueeze(0).cuda()
                cue_logits, cue_sep_logits = cue_model(
                    tmp_input_ids, attention_mask=tmp_pad_mask_in)
                pred_cues = torch.argmax(cue_logits, dim=-1).squeeze()
                pred_cue_sep = torch.argmax(cue_sep_logits, dim=-1).squeeze()
                # e.g.
                # pred_cue:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
                # pred_cue_sep: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
                nu = pred_cue_sep.clone()
                num_cues = nu.max().item()
                sep_cues = [[3] * tmp_input_lens for c in range(num_cues)]
                for c in range(num_cues):
                    for i, e in enumerate(tmp_pad_mask):
                        if e == 1:
                            if pred_cue_sep[i].item() == c+1:
                                sep_cues[c][i] = pred_cues[i].item()
                
                gold_sep_cues = [[3] * tmp_input_lens for c in range(gold_nc)]
                for gc in range(gold_nc):
                    for i, e in enumerate(tmp_pad_mask):
                        if e == 1:
                            if tmp_sep[i] == gc+1:
                                gold_sep_cues[gc][i] = tmp_cue[i]
                nc = max(num_cues, gold_nc)
                cue_match = [-1 for c in range(nc)]
                for pc in range(num_cues):
                    pred_cue_pos = [index for index, v in enumerate(sep_cues[pc]) if v == 1 or v == 0]
                    for gc in range(gold_nc):
                        gold_cue_pos = [index for index, v in enumerate(gold_sep_cues[gc]) if v == 1 or v == 0]
                        match = bool(set(pred_cue_pos) & set(gold_cue_pos))
                        print()
                        if match:
                            cue_match[pc] = gc


                for c in range(num_cues):
                    new_text = []
                    new_cues = []
                    new_masks = []
                    pos = 0
                    prev = 3
                    for token, cue, mask in zip(tmp_text, sep_cues[c], tmp_mask):
                        # Process the cue augmentation.
                        # Different from the original repo, the strategy is indicate the cue border
                        if cue != 3:
                            if pos != 0 and pos != len(sent)-1:
                                if cue == prev:
                                    # continued cue, don't care is subword or multi word cue
                                    new_masks.append(mask)
                                    new_text.append(token)
                                    new_cues.append(cue)
                                else:
                                    # left bound
                                    new_text.append('[unused1]')
                                    new_cues.append(cue)
                                    new_masks.append(1)
                                    new_text.append(token)
                                    new_masks.append(0)
                                    new_cues.append(cue)
                            elif pos == 0:
                                # at pos 0
                                new_text.append('[unused1]')
                                new_masks.append(1)
                                new_cues.append(cue)
                                new_text.append(token)
                                new_masks.append(0)
                                new_cues.append(cue)
                            else:
                                # at eos
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                                new_text.append('[unused1]')
                                new_masks.append(0)
                                new_cues.append(cue)
                        else:
                            if cue == prev:
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                            else:
                                # current non cue, insert right bound before current pos
                                new_text.append('[unused1]')
                                new_masks.append(0)
                                new_cues.append(prev)
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                        prev = cue
                        pos += 1
                    input_id = self.tokenizer.convert_tokens_to_ids(new_text)
                    padding_mask = [1] * len(input_id)
                    input_len = len(input_id)
                    while len(input_id) < max_seq_len:
                        input_id.append(0)
                        padding_mask.append(0)
                        new_masks.append(0)
                        new_cues.append(0)
                    assert len(input_id) == max_seq_len
                    assert len(padding_mask) == max_seq_len
                    assert len(new_masks) == max_seq_len
                    assert len(new_cues) == max_seq_len
                    wrap_sents.append(new_text)
                    wrap_input_id.append(input_id)
                    wrap_subword_mask.append(new_masks)
                    wrap_cues.append(new_cues)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)

            if counter < len(scope_input):
                gold_scopes = scope_input[counter].scopes
                gold_num_cue = gold_nc
            else:
                gold_num_cue = gold_nc
                gold_scopes = [[2 for i in sent_list]]

            feature = PipelineScopeFeature(guid=cue_ex.guid, or_sent=cue_ex.sent, sents=wrap_sents, input_ids=wrap_input_id,
                                           padding_mask=wrap_padding_mask, subword_mask=wrap_subword_mask,
                                           input_len=wrap_input_len, cues=wrap_cues, cue_match=cue_match, gold_scopes=gold_scopes, gold_num_cues=gold_num_cue)
            features.append(feature)
        return features

    def create_dataset(self, features: List[FeatureLike], cue_or_scope: str, example_type: str,
                       is_sorted=False, is_bert=False) -> Union[Dataset, TensorDataset]:
        """
        Pack the features to dataset. If cue_or_scope is cue or scope, 
        return a TensorDataset for faster processing.
        If cue_or_scope is pipeline and example_type is dev or test, return a Dataset,
        in which the features still keep to the packed scope format.

        params:
            features (list): collection of features
            cue_or_scope (str): in ['cue', 'scope', 'pipeline']
            example_type (str): in ['train', 'dev', 'test']
            is_sorted (bool): to specify whether to sort the dataset or not. Save time and space when dumping.
            is_bert (bool)

        return:
            cue_or_scope == cue:
                TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_mask)
            cue_or_scope == scope:
        """

        if is_sorted:
            print('sorted data by th length of input')
            features = sorted(
                features, key=lambda x: x.input_len, reverse=True)
        if cue_or_scope.lower() == 'cue':
            if is_bert:
                input_ids = []
                padding_mask = []
                subword_mask = []
                cues = []
                cue_sep = []
                input_len = []
                for feature in features:
                    input_ids.append(feature.input_ids)
                    padding_mask.append(feature.padding_mask)
                    subword_mask.append(feature.subword_mask)
                    cues.append(feature.cues)
                    cue_sep.append(feature.cue_sep)
                    input_len.append(feature.input_len)

                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=0, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=3, padding="post",
                                     dtype="long", truncating="post").tolist()
                cue_sep = pad_sequences(cue_sep,
                                        maxlen=param.max_len, value=0, padding="post",
                                        dtype="long", truncating="post").tolist()
                subword_mask = pad_sequences(subword_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()

                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                cues = torch.LongTensor(cues)
                cue_sep = torch.LongTensor(cue_sep)
                input_len = torch.LongTensor(input_len)
                subword_mask = torch.LongTensor(subword_mask)
                return TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_mask)
            else:
                input_ids = []
                padding_mask = []
                cues = []
                cue_sep = []
                input_len = []
                for feature in features:
                    input_ids.append(feature.input_ids)
                    padding_mask.append(feature.padding_mask)
                    cues.append(feature.cues)
                    cue_sep.append(feature.cue_sep)
                    input_len.append(feature.input_len)

                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=0, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=3, padding="post",
                                     dtype="long", truncating="post").tolist()
                cue_sep = pad_sequences(cue_sep,
                                        maxlen=param.max_len, value=0, padding="post",
                                        dtype="long", truncating="post").tolist()

                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                cues = torch.LongTensor(cues)
                cue_sep = torch.LongTensor(cue_sep)
                input_len = torch.LongTensor(input_len)
                subword_masks = torch.zeros_like(input_ids)
                return TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_mask)
        elif cue_or_scope.lower() == 'scope':
            if is_bert:
                input_ids = []
                padding_mask = []
                subword_mask = []
                scopes = []
                input_len = []
                cues = []
                for feature in features:
                    for cue_i in range(feature.num_cues):
                        input_ids.append(feature.input_ids[cue_i])
                        padding_mask.append(feature.padding_mask[cue_i])
                        scopes.append(feature.scopes[cue_i])
                        subword_mask.append(feature.subword_mask[cue_i])
                        input_len.append(feature.input_len[cue_i])
                        cues.append(feature.cues[cue_i])

                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=0, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                
                scopes = pad_sequences(scopes,
                                    maxlen=param.max_len, value=0, padding="post",
                                    dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
                subword_mask = pad_sequences(subword_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()

                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                input_len = torch.LongTensor(input_len)
                scopes = torch.LongTensor(scopes)
                cues = torch.LongTensor(cues)
                subword_masks = torch.LongTensor(subword_mask)
                scopes_matrix = self.scope_to_matrix(scopes, cues, input_len)
                if param.matrix:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks, scopes_matrix)
                else:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks)
            else:
                input_ids = []
                padding_mask = []
                scopes = []
                input_len = []
                cues = []
                for feature in features:
                    for cue_i in range(feature.num_cues):
                        input_ids.append(feature.input_ids[cue_i])
                        padding_mask.append(feature.padding_mask[cue_i])
                        scopes.append(feature.scopes[cue_i])
                        input_len.append(feature.input_len[cue_i])
                        cues.append(feature.cues[cue_i])
                        assert len(feature.scopes[cue_i]) == len(feature.cues[cue_i])
                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=0, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                scopes = pad_sequences(scopes,
                                       maxlen=param.max_len, value=0, padding="post",
                                       dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                scopes = torch.LongTensor(scopes)
                input_len = torch.LongTensor(input_len)
                cues = torch.LongTensor(cues)
                subword_masks = torch.zeros_like(input_ids)
                scopes_matrix = self.scope_to_matrix(scopes, cues, input_len)
                if param.matrix:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks, scopes_matrix)
                else:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks)
        elif cue_or_scope.lower() == 'pipeline' and example_type.lower() == 'test':
            return Dataset(features)
        elif cue_or_scope.lower() == 'multi': 
            # feature: MultiScopeFeature
            master_input_ids = []
            master_mat = []
            master_padding_masks = []
            master_subword_masks = []
            input_len = []
            input_ids = []
            scopes = []
            cues = []
            for feature in features:
                input_ids.append(feature.input_ids)
                master_padding_masks.append(feature.master_padding_masks)
                scopes.append(feature.scopes)
                master_subword_masks.append(feature.master_subword_masks)
                input_len.append(feature.input_len)
                cues.append(feature.cues)
                master_input_ids.append(feature.master_input_ids)
                master_mat.append(feature.master_mat)

            master_padding_masks = pad_sequences(master_padding_masks,
                                         maxlen=param.max_len, value=0, padding="post",
                                         dtype="long", truncating="post").tolist()
            master_subword_masks = pad_sequences(master_subword_masks,
                                         maxlen=param.max_len, value=0, padding="post",
                                         dtype="long", truncating="post").tolist()
            master_input_ids = pad_sequences(master_input_ids,
                                      maxlen=param.max_len, value=0, padding="post",
                                      dtype="long", truncating="post").tolist()

            master_input_ids = torch.LongTensor(master_input_ids)
            master_padding_masks = torch.LongTensor(master_padding_masks)
            master_subword_masks = torch.LongTensor(master_subword_masks)
            master_mat = torch.stack(master_mat, 0)
            return MultiScopeDataset(input_ids, master_padding_masks, scopes, input_len, cues, master_subword_masks, master_input_ids, master_mat)
        else:
            raise ValueError(cue_or_scope, example_type)
    
    def create_pipeline_ds(self, feats: List[PipelineScopeFeature]):
        weak = []
        strict = []


    def ex_to_bioes(self, data: List[ScopeExample]):
        new_features = data
        for f_count, feat in enumerate(new_features):
            for c_count in range(feat.num_cues):
                cues = feat.cues[c_count]
                scopes = feat.scopes[c_count]
                temp_scope = []
                for i, e in enumerate(scopes):
                    if e == 2:
                        if cues[i] != 3:
                            temp_scope.append(6)
                        else:
                            temp_scope.append(e)
                    else:
                        temp_scope.append(e)
                if len(scopes) != 1:
                    scope_bioes = scope_to_bioes(temp_scope)
                else:
                    scope_bioes = scopes
                new_features[f_count].scopes[c_count] = scope_bioes
        return new_features

    def single_scope_to_matrix(self, scope: List, cue: List, input_len: int):
        temp_scope = []
        for j, e in enumerate(scope):
            if e == 2:
                if cue[j] != 3:
                    temp_scope.append(3)
                else:
                    temp_scope.append(e)
            else:
                temp_scope.append(e)
        if len(scope) != 1:
            assert len(scope) == len(cue)
            scope_matrix = single_scope_to_link_matrix_pad(temp_scope, cue, input_len)
        else:
            scope_matrix = torch.LongTensor(scope)
        return scope_matrix

    def scope_to_matrix(self, scopes: Tensor, cues: Tensor, input_lens: Tensor):
        dataset_size = scopes.size(0)
        all_scope_matrix = []
        for i in range(dataset_size):
            scope = scopes[i].tolist()
            cue = cues[i].tolist()
            input_len = input_lens[i].tolist()
            scope_matrix = self.single_scope_to_matrix(scope, cue, input_len)
            all_scope_matrix.append(scope_matrix)
        all_scope_matrix = torch.stack(all_scope_matrix, 0)
        return all_scope_matrix
    
    def multi_scopes_to_matrix(self, scopes: List, cues: List, input_lens: List, num_cues: int) -> np.ndarray:
        marked_scopes = []
        for i in range(num_cues):
            temp_scope = []
            for j, e in enumerate(scopes[i]):
                if e == 2:
                    if cues[i][j] != 3:
                        temp_scope.append(3)
                    else:
                        temp_scope.append(e)
                else:
                    temp_scope.append(e)
            marked_scopes.append(temp_scope)
        mat_m = multi_scope_to_link_matrix_pad(marked_scopes, cues, input_lens)
        return mat_m

    def scope_add_cue(self, data: List[ScopeExample]):
        new_features = data.copy()
        for f_count, feat in enumerate(new_features):
            for c_count in range(feat.num_cues):
                cues = feat.cues[c_count]
                scopes = feat.scopes[c_count]
                temp_scope = []
                for i, e in enumerate(scopes):
                    if cues[i] != 3:
                        temp_scope.append(3)
                    else:
                        temp_scope.append(e)
                new_features[f_count].scopes[c_count] = temp_scope
        return new_features
                

    def get_tokenizer(self, data: Tuple[InputExample], is_bert=False, do_lower_case=False, bert_path=None, non_cue_sents: List[str] = None, noembed=False):
        if is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name_or_path=param.bert_path, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        else:
            if noembed:
                self.tokenizer = NaiveTokenizer(data)
            else:
                self.tokenizer = OtherTokenizer(
                    data, external_vocab=False, non_cue_sents=non_cue_sents)


class Dictionary(object):
    def __init__(self):
        self.token2id = {}
        self.id2token = []

    def add_word(self, word):
        if word not in self.token2id:
            self.id2token.append(word)
            self.token2id[word] = len(self.id2token) - 1
        return self.token2id[word]

    def __len__(self):
        return len(self.id2token)


class NaiveTokenizer(object):
    def __init__(self, data: Tuple[InputExample], non_cue_sents: List[str] = None):
        self.dictionary = Dictionary()
        self.data = data
        self.dictionary.add_word('<PAD>')
        for s in self.data:
            if isinstance(s.sent, str):
                split_sent = s.sent.split(' ')
                for word in split_sent:
                    self.dictionary.add_word(word)
            else:
                for word in s.sent:
                    self.dictionary.add_word(word)
        if non_cue_sents is not None:
            for sent in non_cue_sents:
                for word in sent:
                    self.dictionary.add_word(word)
        self.dictionary.add_word('<OOV>')
        self.dictionary.add_word('[CLS]')
        self.dictionary.add_word('[SEP]')

    def tokenize(self, text: Union[str, List]):
        if isinstance(text, list):
            return text
        elif isinstance(text, str):
            words = text.split()
            return words

    def convert_tokens_to_ids(self, tokens: Iterable):
        if isinstance(tokens, list):
            ids = []
            for token in tokens:
                try:
                    ids.append(self.dictionary.token2id[token])
                except KeyError:
                    ids.append(self.dictionary.token2id['<OOV>'])
            return ids
        elif isinstance(tokens, str):
            try:
                return self.dictionary.token2id[tokens]
            except KeyError:
                return self.dictionary.token2id['<OOV>']

    def decode(self, ids):
        token_list = [self.dictionary.id2token[tid] for tid in ids]
        return " ".join(token_list)


class OtherTokenizer(NaiveTokenizer):
    def __init__(self, data, emb=param.embedding, external_vocab=False, non_cue_sents=None):
        super(OtherTokenizer, self).__init__(data, non_cue_sents)
        if external_vocab is True:
            with open('reduced_fasttext_vocab.bin', 'rb') as f:
                vocab = _pickle.load(f)
            for i, (k, v) in enumerate(vocab.items()):
                self.dictionary.add_word(k)
        self.vector = gensim.models.KeyedVectors.load_word2vec_format(
            param.emb_cache, binary=True)
        self.embedding = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06),
                                           (len(self.dictionary.token2id), param.word_emb_dim))
        for w in self.dictionary.token2id:
            if w in self.vector:
                self.embedding[self.dictionary.token2id[w]] = self.vector[w]
            elif w.lower() in self.vector:
                self.embedding[self.dictionary.token2id[w]
                               ] = self.vector[w.lower()]
        del self.vector
        gc.collect()


if __name__ == "__main__":
    proc = Processor()
    sfu_data = proc.read_data(param.data_path['sfu'], 'sfu')
    proc.create_examples(sfu_data, 'joint', 'sfu', 'cue', 'sfu.pt')
    bio_a_data = proc.read_data(
        param.data_path['bioscope_abstracts'], 'bioscope')
    proc.create_examples(bio_a_data, 'joint', 'bioscope_a', 'cue', 'bioA.pt')
    bio_f_data = proc.read_data(param.data_path['bioscope_full'], 'bioscope')
    proc.create_examples(bio_f_data, 'joint', 'bioscope_f', 'cue', 'bioF.pt')
