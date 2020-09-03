import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Data, SplitData, NaiveTokenizer, OtherTokenizer, GetDataLoader
from model import Neg
from tqdm import tqdm
from params import param
from rouge import Rouge
from sklearn.metrics import classification_report
import numpy as np
import random

rouge = Rouge()
device = torch.device("cuda")

def r_scope(input_tensor, y_true, y_pred, tokenizer, method='rouge-1', print_=False):
    """
    args:
        input_tensor, y_true, y_pred: input sentence, actual label, predicted label tensor in shape of collection of batches.
        method: use which rouge metric. Available options: rouge-1, rouge-2, rouge-l
    """
    input_tensor = [np.array(e) for e in input_tensor]
    true_mask = [np.array(e==1, dtype=bool) for e in y_true]
    pred_mask = [np.array(e==1, dtype=bool) for e in y_pred]
    true_scopes = []
    pred_scopes = []
    for i in range(len(input_tensor)):
        ts = input_tensor[i][true_mask[i]]
        if len(ts) != 0:
            ts_string = tokenizer.decode(ts)
        else:
            ts_string = ' '
        true_scopes.append(ts_string)
        ps = input_tensor[i][pred_mask[i]]
        if len(ps) != 0:
            ps_string = tokenizer.decode(ps)
        else:
            ps_string = ' '
        if ts_string is None or ps_string is None:
            print()
        pred_scopes.append(ps_string)
    for (hyp, ref) in zip(true_scopes, pred_scopes):
        hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
        ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
        if len(hyp) <=0 or len(ref) <=0:
            print()
    score = rouge.get_scores(true_scopes, pred_scopes, avg=True)[method]
    if print_:
        print(f"{method} f1: {score}")
    return score['f']

def f1_scope(y_true, y_pred, print_=False):
    tp = 0
    fn = 0
    fp = 0
    for y_t, y_p in zip(y_true, y_pred):
        if torch.equal(y_t, y_p):
            tp += 1
        else:
            fn += 1
    prec = 1
    rec = tp / (tp + fn)
    if print_ is True:
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {2*prec*rec/(prec+rec)}")

if param.dataset_name == 'sfu':
    in_data = Data(param.data_path['sfu'], 'sfu')
    train_data = SplitData(in_data.cue_data[0], in_data.scope_data[0])
    test_data = SplitData(in_data.cue_data[1], in_data.scope_data[1])
    dev_data = SplitData(in_data.cue_data[2], in_data.scope_data[2])
elif param.dataset_name == 'bioscope_abstracts':
    in_data = Data(param.data_path['bioscope_abstracts'], 'bioscope')
    train_data = SplitData(in_data.cue_data[0], in_data.scope_data[0])
    test_data = SplitData(in_data.cue_data[1], in_data.scope_data[1])
    dev_data = SplitData(in_data.cue_data[2], in_data.scope_data[2])
elif param.dataset_name == 'bioscope_full':
    in_data = Data(param.data_path['bioscope_full'], 'bioscope')
    train_data = SplitData(in_data.cue_data[0], in_data.scope_data[0])
    test_data = SplitData(in_data.cue_data[1], in_data.scope_data[1])
    dev_data = SplitData(in_data.cue_data[2], in_data.scope_data[2])
elif param.dataset_name == 'starsem':
    train_data = Data(param.data_path['starsem']['train'], 'starsem')
    test1_data = Data(param.data_path['starsem']['test1'], 'starsem')
    test2_data = Data(param.data_path['starsem']['test2'], 'starsem')
    dev_data = Data(param.data_path['starsem']['dev'], 'starsem')
    test_data = test1_data#_data if random.randint(0, 1) == 0 else test2_data

tokenizer = OtherTokenizer(train_data, external_vocab=True)

if param.cross_test is True:
    sfu = Data(param.data_path['sfu'], 'sfu')
    bioscope_abstracts = Data(param.data_path['bioscope_abstracts'], 'bioscope')
    bioscope_full = Data(param.data_path['bioscope_full'], 'bioscope')
    starsem_1 = Data(param.data_path['starsem']['test1'], 'starsem')
    starsem_2 = Data(param.data_path['starsem']['test2'], 'starsem')
    test_datasets = {
        'sfu': SplitData(sfu.cue_data[1], sfu.scope_data[1]),
        'bioscope_abstracts': SplitData(bioscope_abstracts.cue_data[1], bioscope_abstracts.scope_data[1]),
        'bioscope_full': SplitData(bioscope_full.cue_data[1], bioscope_full.scope_data[1]),
        'starsem_1': starsem_1,
        'starsem_2': starsem_2
    }
    test_dls = {
        'sfu': GetDataLoader(data=test_datasets['sfu'], tokenizer=tokenizer).get_scope_dataloader(),
        'bioscope_abstracts': GetDataLoader(data=test_datasets['bioscope_abstracts'], tokenizer=tokenizer).get_scope_dataloader(),
        'bioscope_full': GetDataLoader(data=test_datasets['bioscope_full'], tokenizer=tokenizer).get_scope_dataloader(),
        'starsem_1': GetDataLoader(data=test_datasets['starsem_1'], tokenizer=tokenizer).get_scope_dataloader(),
        'starsem_2': GetDataLoader(data=test_datasets['starsem_2'], tokenizer=tokenizer).get_scope_dataloader()
    }

#traindl, testdl, devdl = GetDataLoader(data=in_data, tokenizer=tokenizer).get_scope_dataloader(split_train=True)
traindl = GetDataLoader(data=train_data, tokenizer=tokenizer).get_scope_dataloader()
devdl = GetDataLoader(data=dev_data, tokenizer=tokenizer).get_scope_dataloader()
testdl = GetDataLoader(data=test_data, tokenizer=tokenizer).get_scope_dataloader()

model = Neg(vocab_size=tokenizer.dictionary.__len__(), tokenizer=tokenizer)
model.to(device)
if isinstance(tokenizer, NaiveTokenizer):
    model.init_embedding(model.word_emb)
else:
    model.load_pretrained_word_embedding(tokenizer.embedding)
model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params=model_params, lr=param.lr)
decay = 0.7
losses = []

early_stop_count = 0
best_f = 0
best_r = 0

print(f'Training on {param.dataset_name}\n')
ep_bar = tqdm(total=param.num_ep, desc="Epoch")
for _ in range(param.num_ep):
    model.train()
    el = 0.0
    train_step = 0
    for step, batch in enumerate(traindl):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        targets = batch[2]
        loss = model(batch, targets)
        el += loss.item()
        loss.backward()
        optimizer.step()
        train_step += 1
        if step % 50 == 0:
            ep_bar.set_postfix({"Ep":_ ,"Batch": step, "loss": loss.item()})
    model.eval()
    eval_loss = 0.0
    ep_bar.update()
    allinput, alltarget, allpred = [], [], []
    for step, batch in enumerate(devdl):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            targets = batch[2]
            logits = model.lstm_output(batch)
            pred = torch.argmax(logits, 2)
            for sent in batch[0]:
                allinput.append(sent.detach().cpu())
            for tar in batch[2]:
                alltarget.append(tar.detach().cpu())
            for pre in pred:
                allpred.append(pre.detach().cpu())
    r_f1 = r_scope(allinput, alltarget, allpred, devdl.tokenizer, print_=True)
    t_flat = [i for t in alltarget for i in t]
    p_flat = [i for p in allpred for i in p]
    f1_scope(alltarget, allpred, True)
    conf_matrix = classification_report([i for i in t_flat], [i for i in p_flat], output_dict=True)
    f1 = conf_matrix["1"]["f1-score"]
    if best_r > r_f1:
        early_stop_count += 1
    else:
        early_stop_count = 0
        best_r = r_f1
    if early_stop_count > 8:
        for paramg in optimizer.param_groups:
            paramg['lr'] *= decay
    if early_stop_count > param.early_stop_thres:
        print("Early stopping")
        break

def testing(testdl):
    testinput, testtarget, testpred = [], [], []
    for step, batch in enumerate(testdl):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            targets = batch[2]
            logits = model.lstm_output(batch)
            pred = torch.argmax(logits, 2)
            for sent in batch[0]:
                testinput.append(sent.detach().cpu())
            for tar in batch[2]:
                testtarget.append(tar.detach().cpu())
            for pre in pred:
                testpred.append(pre.detach().cpu())
    r_scope(testinput, testtarget, testpred, testdl.tokenizer, print_=True)
    tt_flat = [i for t in testtarget for i in t]
    pt_flat = [i for p in testpred for i in p]
    f1_scope(testtarget, testpred, True)
    print(classification_report([i for i in tt_flat], [i for i in pt_flat]))

print("\nEvaluating")
if param.cross_test is False:
    testing(testdl)
else:
    test_list = ['sfu', 'bioscope_abstracts', 'bioscope_full', 'starsem_1', 'starsem_2']
    #test_list.pop(test_list.index(param.dataset_name))
    for tn in test_list:
        testing(test_dls[tn])
        print(f"Evaluation on {tn} is done!\n\n")
