import torch
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from sklearn.metrics import classification_report

from data import SplitMoreData
from processor import Processor, NaiveTokenizer, CueExample, CueFeature, ScopeExample, ScopeFeature, PipelineScopeFeature
from trainer import CueTrainer, ScopeTrainer, target_weight_score
from model import ScopeRNN
from model_bert import CueBert, ScopeBert
from optimizer import BertAdam, BERTReduceLROnPlateau, ASLSingleLabel
from util import TrainingMonitor, ModelCheckpoint, EarlyStopping, global_logger, pack_subword
from params import param

proc = Processor()


sherlock_test1 = proc.read_data(
param.data_path['sherlock']['test1'], dataset_name='sherlock')
sherlock_test2 = proc.read_data(
param.data_path['sherlock']['test2'], dataset_name='sherlock')
sherlock_raw = SplitMoreData([sherlock_test1.cues, sherlock_test2.cues], [sherlock_test1.scopes, sherlock_test2.scopes], [
                    sherlock_test1.non_cue_sents, sherlock_test2.non_cue_sents])
cue_data = proc.create_examples(sherlock_raw, 'test', 'sherlock', 'cue')
scope_data = proc.create_examples(sherlock_raw, 'test', 'sherlock', 'scope')
"""
cue_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['test'])
scope_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['test'])
"""

if param.bioes:
    scope_data = proc.ex_to_bioes(scope_data)

proc.get_tokenizer(data=None, is_bert=True, bert_path=param.bert_path)

cue_model = CueBert.from_pretrained('D:\\Dev\\Bert\\cue_bert_base_sherlock')
#model = torch.load('model_chk/scope_bert_base_sherlock.pt')
cue_model.cuda()



p = []
r = []
def match(scope, cue):
    if scope.sent == cue.sent:
        return [cue, scope]
    else:
        return 0

for cue in cue_data:
    matched = 0
    for scope in scope_data:
        t = match(scope, cue)
        if t != 0:
            p.append(t)
            matched = 1
        
    if matched == 0:
        r.append(cue)

#new_scope = [e[1] for e in p]
#new_cue = [e[0] for e in p]
#new_cue.extend(r)

#new_scope = proc.create_features(new_scope, 'scope', max_seq_len=param.max_len, is_bert=param.is_bert)
#fp = proc.create_features_pipeline(new_cue, new_scope, cue_model, max_seq_len=param.max_len, is_bert=param.is_bert, non_cue_examples=None)
fp = proc.create_features_pipeline(cue_data, scope_data, cue_model, max_seq_len=param.max_len, is_bert=param.is_bert, non_cue_examples=None)

del cue_model

scope_model = ScopeBert.from_pretrained('D:\\Dev\\Bert\\scope_bert_base_sherlock')
num_labels = scope_model.num_labels
all_tar = []
all_pred = []
all_match = []
for pair in fp:
    pred_nc = len(pair.cue_match)
    if pred_nc == []:
        continue
    for input_, pad, subw, c_match, index in zip(pair.input_ids, pair.padding_mask, pair.subword_mask, pair.cue_match, list(range(pred_nc))):
        with torch.no_grad():
            input_ = torch.LongTensor(input_).unsqueeze(0)
            pad = torch.LongTensor(pad).unsqueeze(0)
            scope_logits = scope_model(input_, pad)[0].argmax(-1)
            scope_pred = pack_subword(scope_logits, subw)
        if c_match != -1:
            if pair.gold_num_cues != 0:
                target = pair.gold_scopes[c_match]
                pred = scope_pred[1:-1]
            else:
                target = pair.gold_scopes
                pred = scope_pred[1:-1]
        else:
            if index > pair.gold_num_cues - 1:
                target = [2 for v in scope_pred]
                pred = scope_pred
            else:
                target = pair.gold_scopes[index]
                pred = [2 for v in target]
    if len(target) != len(pred):
        pred = pred[:-1]
    assert len(target) == len(pred)
    all_tar.append(target)
    all_pred.append(pred)
    all_match.append(c_match)

result = classification_report(
            [i for j in all_tar for i in j], [i for j in all_pred for i in j], digits=4, output_dict=True)

print()
