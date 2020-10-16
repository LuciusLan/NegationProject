import random
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer
from transformers.file_utils import cached_path
import numpy as np
from typing import List, Tuple, T, Iterable, Union, NewType
import gensim
import _pickle

from util import pad_sequences
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
    def __init__(self, cues: List[int], scopes: List[T], sc_sent: List[str], **kwargs):
        super().__init__(**kwargs)
        self.num_cues = len(scopes)
        self.cues = cues
        self.scopes = scopes
        self.sc_sent = sc_sent

ExampleLike = Union[CueExample, ScopeExample, InputExample]

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
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len, cues, scopes):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.scopes = scopes

class PipelineScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len, cues, gold_cues, gold_scopes):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.gold_cues = gold_cues
        self.gold_scopes = gold_scopes

FeatureLike = Union[CueFeature, ScopeFeature, PipelineScopeFeature]

class Processor(object):
    def __init__(self):
        self.tokenizer = None

    @classmethod
    def read_data(cls, input_file, dataset_name=None) -> RawData:
        return RawData(input_file, dataset_name=dataset_name)

    def create_examples(self, data: RawData, example_type: str, cue_or_scope: str, cached_file=None,
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
            'train', 'test', 'dev', 'split'], 'Wrong example type.'
        assert cue_or_scope in ['cue', 'scope', 'raw'], 'cue_or_scope: Must specify cue of scope, or raw to perform split and get vocab'
        
        cue_examples = []
        scope_examples = []
        for i, _ in enumerate(data.cues[0]):
            guid = '%s-%d' % (example_type, i)
            sentence = data.cues[0][i]
            cues = data.cues[1][i]
            cue_sep = data.cues[2][i]
            num_cues = data.cues[3][i]
            sent = ' '.join(sentence)
            cue_examples.append(CueExample(guid=guid, sent=sent, cues=cues, cue_sep=cue_sep, num_cues=num_cues, subword_mask=None))

        for i, _ in enumerate(data.scopes[0]):
            guid = '%s-%d' % (example_type, i)
            or_sent = data.scopes[0][i]
            sentence = data.scopes[1][i]
            cues = data.scopes[2][i]
            sent = ' '.join(or_sent)
            scopes = data.scopes[3][i]
            scope_examples.append(ScopeExample(guid=guid, sent=sent, cues=cues, scopes=scopes, sc_sent=sentence, subword_mask=None))

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
            random_state = np.random.randint(1, 2020)
            tr_cue_, te_cue, tr_scope_, te_scope = train_test_split(cue_examples, scope_examples, test_size=test_size, random_state=random_state)
            random_state2 = np.random.randint(1, 2020)
            tr_cue, dev_cue, tr_scope, dev_scope = train_test_split(tr_cue_, tr_scope_, test_size=(
                val_size / (1 - test_size)), random_state=random_state2)
            for i, e in enumerate(tr_cue):
                e.guid = f'train-{i}'
            for i, e in enumerate(te_cue):
                e.guid = f'test-{i}'
            for i, e in enumerate(dev_cue):
                e.guid = f'dev-{i}'
            for i, e in enumerate(tr_scope):
                e.guid = f'train-{i}'
            for i, e in enumerate(te_scope):
                e.guid = f'test-{i}'
            for i, e in enumerate(dev_scope):
                e.guid = f'dev-{i}'
            if cached_file is not None:
                print('Saving examples into cached file %s', cached_file)
                torch.save(tr_cue, f'train_cue_{cached_file}')
                torch.save(te_cue, f'test_cue_{cached_file}')
                torch.save(dev_cue, f'dev_cue_{cached_file}')
                torch.save(tr_scope, f'train_scope_{cached_file}')
                torch.save(te_scope, f'test_scope_{cached_file}')
                torch.save(dev_scope, f'dev_scope_{cached_file}')
            return (tr_cue, dev_cue, te_cue), (tr_scope, dev_scope, te_scope)
    
    def load_examples(self, file: str):
        """
        Load a pre-saved example binary file. Or anything else.
        
        Warning: torch.load() uses pickle module implicitly, which is known to be insecure. 
        It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.
        Never load data that could have come from an untrusted source, or that could have been tampered with.
        Only load data you trust.
        """
        return torch.load(file)

    def create_features(self, data: List[ExampleLike], cue_or_scope: str,
                        max_seq_len: int, is_bert=False) -> List[Union[CueFeature, ScopeFeature]]:
        """
        Create packed 
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        if cue_or_scope == 'cue':
            for example in data:
                sent = example.sent
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
                    new_cuesep.insert(0, 3)
                    new_cuesep.append(3)
                    input_ids = self.tokenizer.convert_tokens_to_ids(new_text)
                    padding_mask = [1] * len(input_ids)
                    input_len = len(input_ids)
                    while len(input_ids) < max_seq_len:
                        input_ids.append(0)
                        padding_mask.append(0)
                        subword_mask.append(0)
                        new_cues.append(3)
                        new_cuesep.append(0)
                    feature = CueFeature(guid=guid, sent=sent, input_ids=input_ids, 
                                         padding_mask=padding_mask, subword_mask=subword_mask,
                                         input_len=input_len, cues=new_cues, cue_sep=new_cuesep, num_cues=num_cues)
                else:
                    # For non-BERT (non-BPE tokenization)
                    cues = example.cues
                    cues.insert(0, 3)
                    cues.append(3)
                    cue_sep = example.cue_sep
                    cue_sep.insert(0, 3)
                    cue_sep.append(3)
                    words = self.tokenizer.tokenize(sent)
                    words.insert(0, '[CLS]')
                    words.append('[SEP]')
                    input_ids = self.tokenizer.convert_tokens_to_ids(words)
                    padding_mask = [1] * len(input_ids)
                    input_len = len(input_ids)
                    while len(input_ids) < max_seq_len:
                        input_ids.append(0)
                        padding_mask.append(0)
                        cues.append(3)
                        cue_sep.append(0)
                    feature = CueFeature(guid=guid, sent=sent, input_ids=input_ids, 
                                         padding_mask=padding_mask, subword_mask=None,
                                         input_len=input_len, cues=cues, cue_sep=cue_sep, num_cues=num_cues)
                features.append(feature)
        else:
            # For scope
            for example in data:
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
                            if '<AFF>' not in word:
                                # Deal with affix cue. (manually)
                                subwords = self.tokenizer.tokenize(word)
                                for count, subword in enumerate(subwords):
                                    mask = 1
                                    if count > 0:
                                        mask = 0
                                    temp_mask.append(mask)
                                    temp_cues.append(cue)
                                    temp_scope.append(scope)
                                    temp_sent.append(subword)
                            else:
                                word = word[5:]
                                subwords = self.tokenizer.tokenize(word)
                                for count, subword in enumerate(subwords):
                                    mask = 0
                                    if count > 0:
                                        mask = 0
                                    temp_mask.append(mask)
                                    temp_cues.append(cue)
                                    temp_scope.append(scope)
                                    if count != 0:
                                        temp_sent.append(subword)
                                    else:
                                        temp_sent.append("##"+subword)
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
                            if cue != 3:
                                if pos != 0 and pos != len(sent)-1:
                                    if cue == prev:
                                        # continued cue, don't care is subword or multi word cue
                                        new_masks.append(mask)
                                        new_text.append(token)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                    else:
                                        # left bound
                                        new_text.append(f'[unused{cue+1}]')
                                        new_cues.append(cue)
                                        new_masks.append(1)
                                        new_scopes.append(label)
                                        new_text.append(token)
                                        new_masks.append(0)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                elif pos == 0:
                                    # at pos 0
                                    new_text.append(f'[unused{cue+1}]')
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
                                    new_text.append(f'[unused{cue+1}]')
                                    new_masks.append(0)
                                    new_scopes.append(label)                            
                                    new_cues.append(cue)
                                    """
                                    if cue != 3:
                                        if first_part == 0:
                                            first_part = 1
                                            new_text.append(f'[unused{cue+1}]')
                                            new_masks.append(1)
                                            new_scopes.append(label)
                                            new_text.append(token)
                                            new_masks.append(0)
                                            new_scopes.append(label)
                                            continue
                                        if cue != 0:
                                            # only set the unused token when subword is not part of affix cue
                                            # when affix, set all following subword as non cue
                                            new_text.append(f'[unused{cue+1}]')
                                            new_masks.append(0)
                                            new_scopes.append(label)
                                    """
                            else:
                                if cue == prev:
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                else:
                                    # current non cue, insert right bound before current pos
                                    new_text.append(f'[unused{prev+1}]')
                                    new_masks.append(0)
                                    new_scopes.append(0)
                                    new_cues.append(prev)
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                            prev = cue
                            pos += 1
                        
                        
                        if len(new_text) >= max_seq_len - 1:
                            new_text = new_text[0:(max_seq_len - 2)]
                            new_cues = new_cues[0:(max_seq_len - 2)]
                            new_masks = new_masks[0:(max_seq_len - 2)]
                        new_text.insert(0, '[CLS]')
                        new_text.append('[SEP]')
                        new_masks.insert(0, 1)
                        new_masks.append(1)
                        new_cues.insert(0, 3)
                        new_cues.append(3)
                        new_scopes.insert(0, 2)
                        new_scopes.append(2)
                        input_ids = self.tokenizer.convert_tokens_to_ids(new_text)
                        padding_mask = [1] * len(input_ids)
                        input_len = len(input_ids)
                        while len(input_ids) < max_seq_len:
                            input_ids.append(0)
                            padding_mask.append(0)
                            new_masks.append(0)
                            new_cues.append(3)
                            new_scopes.append(0)
                        wrap_sents.append(new_text)
                        wrap_input_id.append(input_ids)
                        wrap_subword_mask.append(new_masks)
                        wrap_cues.append(new_cues)
                        wrap_scopes.append(new_scopes)
                        wrap_padding_mask.append(padding_mask)
                        wrap_input_len.append(input_len)
                    else:
                        # For non-BERT (non-BPE tokenization)
                        sent = example.sc_sent[c]
                        cues = example.cues[c]
                        scopes = example.scopes[c]
                        
                        words = self.tokenizer.tokenize(sent)
                        for i, w in enumerate(words):
                            if '<AFF>' in w:
                                words[i-1] = words[i-1] + words[i][5:]
                                words.pop(i)
                                cues.pop(i)
                                scopes.pop(i)

                        words.insert(0, '[CLS]')
                        words.append('[SEP]')
                        cues.insert(0, 3)
                        cues.append(3)
                        scopes.insert(0, 2)
                        scopes.append(2)
                        input_ids = self.tokenizer.convert_tokens_to_ids(words)
                        padding_mask = [1] * len(input_ids)
                        input_len = len(input_ids)
                        while len(input_ids) < max_seq_len:
                            input_ids.append(0)
                            padding_mask.append(0)
                            cues.append(3)
                        wrap_sents.append(words)
                        wrap_input_id.append(input_ids)
                        wrap_subword_mask = None
                        wrap_cues.append(cues)
                        wrap_scopes.append(scopes)
                        wrap_padding_mask.append(padding_mask)
                        wrap_input_len.append(input_len)
                    
                    feature = ScopeFeature(guid=guid, or_sent=example.sent, sents=wrap_sents, input_ids=wrap_input_id, 
                                           padding_mask=wrap_padding_mask, subword_mask=wrap_subword_mask,
                                           input_len=wrap_input_len, cues=wrap_cues, scopes=wrap_scopes)
                features.append(feature)
        return features

    def create_features_pipeline(self, data: List[ScopeFeature], cue_model,
                        max_seq_len: int, is_bert=False):
        """
        Create scope feature for pipeline TESTING. The cue info was predicted with a trained cue 
        model, instead of the golden cues. Training of scope model will be normal scope model, with golden cue input.
        calling cue_model returns the logits of cue label and the separation.
        It shouldn't be batched input as the output batch size is not controlable, due to different number of cues
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for example in data:
            wrap_sents = []
            wrap_input_id = []
            wrap_subword_mask = []
            wrap_cues = []
            wrap_padding_mask = []
            wrap_input_len = []
            sent = example.sent
            tmp_mask = []
            if is_bert:
                tmp_text = []
                for word in sent.split(' '):
                    subwords = self.tokenizer.tokenize(word)
                    for i, subword in enumerate(subwords):
                        mask = 1
                        if i > 0:
                            mask = 0
                        tmp_mask.append(mask)
                        tmp_text.append(subword)
                    if len(tmp_text) >= max_seq_len - 1:
                        tmp_text = tmp_text[0:(max_seq_len - 2)]
                        tmp_mask = tmp_mask[0:(max_seq_len - 2)]
                tmp_text.insert(0, '[CLS]')
                tmp_text.append('[SEP]')
                tmp_mask.insert(0, 1)
                tmp_mask.append(1)
                tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_text)

                tmp_pad_mask = [1] * len(tmp_input_ids)
                tmp_input_lens = len(tmp_input_ids)
                while len(tmp_input_ids) < max_seq_len:
                    tmp_input_ids.append(0)
                    tmp_pad_mask.append(0)
                    tmp_mask.append(0)

                cue_logits, cue_sep_logits = cue_model(tmp_input_ids, attention_mask=tmp_pad_mask)
                pred_cues = torch.argmax(cue_logits, dim=-1)
                pred_cue_sep = torch.argmax(cue_sep_logits, dim=-1)
                # e.g. 
                # pred_cue:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
                # pred_cue_sep: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
                _ = pred_cue_sep.copy()
                num_cues = _.sort()[-1]
                sep_cues = [[3] * tmp_input_lens for c in range(num_cues)]
                for c in range(num_cues):
                    for i, e in enumerate(pred_cue_sep):
                        if c == e:
                            sep_cues[c][i] = pred_cues[i]

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
                                    new_text.append(f'[unused{cue+1}]')
                                    new_cues.append(cue)
                                    new_masks.append(1)
                                    new_text.append(token)
                                    new_masks.append(0)
                                    new_cues.append(cue)
                            elif pos == 0:
                                # at pos 0
                                new_text.append(f'[unused{cue+1}]')
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
                                new_text.append(f'[unused{cue+1}]')
                                new_masks.append(0)                     
                                new_cues.append(cue)
                                """
                                if cue != 3:
                                    if first_part == 0:
                                        first_part = 1
                                        new_text.append(f'[unused{cue+1}]')
                                        new_masks.append(1)
                                        new_scopes.append(label)
                                        new_text.append(token)
                                        new_masks.append(0)
                                        new_scopes.append(label)
                                        continue
                                    if cue != 0:
                                        # only set the unused token when subword is not part of affix cue
                                        # when affix, set all following subword as non cue
                                        new_text.append(f'[unused{cue+1}]')
                                        new_masks.append(0)
                                        new_scopes.append(label)
                                """
                        else:
                            if cue == prev:
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                            else:
                                # current non cue, insert right bound before current pos
                                new_text.append(f'[unused{prev+1}]')
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
                        new_cues.append(3)
                    wrap_sents.append(new_text)
                    wrap_input_id.append(input_id)
                    wrap_subword_mask.append(new_masks)
                    wrap_cues.append(new_cues)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)
            else:
                # Non-BERT
                tmp_text = self.tokenizer.tokenize(sent)
                tmp_text.insert(0, '[CLS]')
                tmp_text.append('[SEP]')
                tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_text)
                cue_logits, cue_sep_logits = cue_model(tmp_input_ids)
                pred_cues = torch.argmax(cue_logits, dim=-1)
                pred_cue_sep = torch.argmax(cue_sep_logits, dim=-1)
                # e.g. 
                # pred_cue:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
                # pred_cue_sep: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
                _ = pred_cue_sep.copy()
                num_cues = _.sort()[-1]
                sep_cues = [[3] * tmp_input_lens for c in range(num_cues)]
                for c in range(num_cues):
                    for i, e in enumerate(pred_cue_sep):
                        if c == e:
                            sep_cues[c][i] = pred_cues[i]
                
                for c in range(num_cues):
                    padding_mask = [1] * len(tmp_input_ids)
                    input_len = len(tmp_input_ids)
                    while len(tmp_input_ids) < max_seq_len:
                        tmp_input_ids.append(0)
                        padding_mask.append(0)
                        sep_cues[c].append(3)
                    wrap_sents.append(tmp_text)
                    wrap_input_id.append(tmp_input_ids)
                    wrap_subword_mask.append(None)
                    wrap_cues.append(sep_cues[c])
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)

            gold_cues = example.cues
            gold_scopes = example.scopes
                
            feature = PipelineScopeFeature(guid=example.guid, or_sent=example.sent, sents=wrap_sents, input_ids=wrap_input_id, 
                                    padding_mask=wrap_padding_mask, subword_mask=wrap_subword_mask,
                                    input_len=wrap_input_len, cues=wrap_cues, gold_cues=gold_cues, gold_scopes=gold_scopes)
            features.append(feature)
        return features
    
    def create_dataset(self, features, cue_or_scope: str, is_sorted=False, is_bert=False):
        # Convert to Tensors and build dataset
        if is_sorted:
            print("sorted data by th length of input")
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        dataset = Dataset(features)
        return dataset
    
    def get_tokenizer(self, data: Tuple[InputExample], is_bert=False, do_lower_case=False, bert_path=None):
        if is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(
                bert_path, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        else:
            self.tokenizer = OtherTokenizer(data, external_vocab=False)


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
    def __init__(self, data: Tuple[InputExample], non_cue_sents: List[str]=None):
        self.dictionary = Dictionary()
        self.data = data
        self.dictionary.add_word('<PAD>')
        for s in self.data:
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
    def __init__(self, data, emb=param.embedding, external_vocab=False):
        super(OtherTokenizer, self).__init__(data)
        if external_vocab is True:
            with open('reduced_fasttext_vocab.bin', 'rb') as f:
                vocab = _pickle.load(f)
            for i, (k, v) in enumerate(vocab.items()):
                self.dictionary.add_word(k)
        self.vector = gensim.models.KeyedVectors.load_word2vec_format(param.emb_cache, binary=True)
        self.embedding = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), 
                                           (len(self.dictionary.token2id), param.word_emb_dim))
        for w in self.dictionary.token2id:
            if w in self.vector:
                self.embedding[self.dictionary.token2id[w]] = self.vector[w]
            elif w.lower() in self.vector:
                self.embedding[self.dictionary.token2id[w]] = self.vector[w.lower()]


if __name__ == "__main__":
    proc = Processor()
    #proc.get_tokenizer(data=None, is_bert=True, bert_path='bert-base-cased')
    data = proc.read_data(param.data_path['sherlock']['train'], dataset_name='sherlock')
    examples = proc.create_examples(data=data,
                    example_type='train',
                    cached_file=None,
                    cue_or_scope='scope')
    proc.get_tokenizer(data=examples, is_bert=False)
    features = proc.create_features(data=examples, max_seq_len=128, is_bert=False,
                    cue_or_scope='scope')
    print()
