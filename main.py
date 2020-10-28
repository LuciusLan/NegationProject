import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from rouge import Rouge
from sklearn.metrics import classification_report
import numpy as np
import random
import logging

from data import SplitData, cue_label2id
from processor import Processor, NaiveTokenizer, CueExample, CueFeature, ScopeExample, ScopeFeature, PipelineScopeFeature
from trainer import CueTrainer, ScopeTrainer
from model import Neg
from model_bert import CueBert, ScopeBert
from optimizer import BertAdam, BERTReduceLROnPlateau, ASLSingleLabel
from util import TrainingMonitor, ModelCheckpoint, EarlyStopping, global_logger
from params import param

rouge = Rouge()
device = torch.device("cuda")


def r_scope(input_tensor, y_true, y_pred, tokenizer, method='rouge-1', print_=False):
    """
    args:
        input_tensor, y_true, y_pred: input sentence, actual label, predicted label tensor in shape of collection of batches.
        method: use which rouge metric. Available options: rouge-1, rouge-2, rouge-l
    """
    input_tensor = [np.array(e) for e in input_tensor]
    true_mask = [np.array(e == 1, dtype=bool) for e in y_true]
    pred_mask = [np.array(e == 1, dtype=bool) for e in y_pred]
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
        if len(hyp) <= 0 or len(ref) <= 0:
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


proc = Processor()
if param.split_and_save:
    sfu_data = proc.read_data(param.data_path['sfu'], 'sfu')
    proc.create_examples(sfu_data, 'split', 'cue', 'sfu.pt')
    bio_a_data = proc.read_data(
        param.data_path['bioscope_abstracts'], 'bioscope')
    proc.create_examples(bio_a_data, 'split', 'cue', 'bioA.pt')
    bio_f_data = proc.read_data(param.data_path['bioscope_full'], 'bioscope')
    proc.create_examples(bio_f_data, 'split', 'cue', 'bioF.pt')


# Please perform split and save first and load the splitted version of datasets,
# for the sake of consistency of dataset when testing for different model structures.
if param.task == 'cue':
    if param.dataset_name == 'sfu':
        train_data = proc.load_examples(
            param.split_path['sfu']['cue']['train'])
        dev_data = proc.load_examples(param.split_path['sfu']['cue']['dev'])
        test_data = proc.load_examples(param.split_path['sfu']['cue']['test'])
    elif param.dataset_name == 'bioscope_abstracts':
        train_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['train'])
        dev_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['dev'])
        test_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['test'])
    elif param.dataset_name == 'bioscope_full':
        train_data = proc.load_examples(
            param.split_path['bioscope_full']['cue']['train'])
        dev_data = proc.load_examples(
            param.split_path['bioscope_full']['cue']['dev'])
        test_data = proc.load_examples(
            param.split_path['bioscope_full']['cue']['test'])
    elif param.dataset_name == 'sherlock':
        train_raw = proc.read_data(
            param.data_path['sherlock']['train'], dataset_name='sherlock')
        sherlock_test1 = proc.read_data(
            param.data_path['sherlock']['test1'], dataset_name='sherlock')
        sherlock_test2 = proc.read_data(
            param.data_path['sherlock']['test2'], dataset_name='sherlock')
        test_raw = SplitData([sherlock_test1.cues, sherlock_test2.cues], [sherlock_test1.scopes, sherlock_test2.scopes], [
                             sherlock_test1.non_cue_sents, sherlock_test2.non_cue_sents])
        dev_raw = proc.read_data(
            param.data_path['sherlock']['dev'], dataset_name='sherlock')
        train_data = proc.create_examples(
            train_raw, cue_or_scope='cue', example_type='train')
        dev_data = proc.create_examples(
            dev_raw, cue_or_scope='cue', example_type='dev')
        test_data = proc.create_examples(
            test_raw, cue_or_scope='cue', example_type='test')

    if param.embedding == 'BERT':
        proc.get_tokenizer(data=None, is_bert=True, bert_path=param.bert_path)
    else:
        proc.get_tokenizer(data=train_data, is_bert=False)

    train_feature = proc.create_features(
        train_data, cue_or_scope='cue', max_seq_len=param.max_len, is_bert=param.is_bert)
    dev_feature = proc.create_features(
        dev_data, cue_or_scope='cue', max_seq_len=param.max_len, is_bert=param.is_bert)
    test_feature = proc.create_features(
        test_data, cue_or_scope='cue', max_seq_len=param.max_len, is_bert=param.is_bert)

    train_ds = proc.create_dataset(
        train_feature, cue_or_scope='cue', example_type='train')
    dev_ds = proc.create_dataset(
        dev_feature, cue_or_scope='cue', example_type='dev')
    test_ds = proc.create_dataset(
        test_feature, cue_or_scope='cue', example_type='test')
elif param.task == 'scope':
    if param.dataset_name == 'sfu':
        train_vocab = proc.load_examples(
            param.split_path['sfu']['cue']['train'])
        train_data = proc.load_examples(
            param.split_path['sfu']['scope']['train'])
        dev_data = proc.load_examples(param.split_path['sfu']['scope']['dev'])
        test_data = proc.load_examples(
            param.split_path['sfu']['scope']['test'])
    elif param.dataset_name == 'bioscope_abstracts':
        train_vocab = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['train'])
        train_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['train'])
        dev_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['dev'])
        test_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['test'])
    elif param.dataset_name == 'bioscope_full':
        train_vocab = proc.load_examples(
            param.split_path['bioscope_full']['cue']['train'])
        train_data = proc.load_examples(
            param.split_path['bioscope_full']['scope']['train'])
        dev_data = proc.load_examples(
            param.split_path['bioscope_full']['scope']['dev'])
        test_data = proc.load_examples(
            param.split_path['bioscope_full']['scope']['test'])
    elif param.dataset_name == 'sherlock':
        train_raw = proc.read_data(
            param.data_path['sherlock']['train'], dataset_name='sherlock')
        sherlock_test1 = proc.read_data(
            param.data_path['sherlock']['test1'], dataset_name='sherlock')
        sherlock_test2 = proc.read_data(
            param.data_path['sherlock']['test2'], dataset_name='sherlock')
        test_raw = SplitData([sherlock_test1.cues, sherlock_test2.cues], [sherlock_test1.scopes, sherlock_test2.scopes], [
                             sherlock_test1.non_cue_sents, sherlock_test2.non_cue_sents])
        dev_raw = proc.read_data(
            param.data_path['sherlock']['dev'], dataset_name='sherlock')
        train_vocab = proc.create_examples(
            train_raw, cue_or_scope='cue', example_type='train')
        train_data = proc.create_examples(
            train_raw, cue_or_scope='scope', example_type='train')
        dev_data = proc.create_examples(
            dev_raw, cue_or_scope='scope', example_type='dev')
        test_data = proc.create_examples(
            test_raw, cue_or_scope='scope', example_type='test')

    if param.embedding == 'BERT':
        proc.get_tokenizer(data=None, is_bert=True, bert_path=param.bert_path)
    else:
        # For standalone scope model, the training vocab should be all training sentences,
        # but the scope train data only contains negation sents. Need to use cue data for a bigger dictionary
        proc.get_tokenizer(data=train_vocab, is_bert=False)

    train_feature = proc.create_features(
        train_data, cue_or_scope='scope', max_seq_len=param.max_len, is_bert=param.is_bert)
    dev_feature = proc.create_features(
        dev_data, cue_or_scope='scope', max_seq_len=param.max_len, is_bert=param.is_bert)
    test_feature = proc.create_features(
        test_data, cue_or_scope='scope', max_seq_len=param.max_len, is_bert=param.is_bert)

    train_ds = proc.create_dataset(
        train_feature, cue_or_scope='scope', example_type='train')
    dev_ds = proc.create_dataset(
        dev_feature, cue_or_scope='scope', example_type='dev')
    test_ds = proc.create_dataset(
        test_feature, cue_or_scope='scope', example_type='test')
elif param.task == 'pipeline':
    if param.dataset_name == 'sfu':
        cue_train_data = proc.load_examples(
            param.split_path['sfu']['cue']['train'])
        cue_dev_data = proc.load_examples(
            param.split_path['sfu']['cue']['dev'])
        cue_test_data = proc.load_examples(
            param.split_path['sfu']['cue']['test'])
        scope_train_data = proc.load_examples(
            param.split_path['sfu']['scope']['train'])
        scope_dev_data = proc.load_examples(
            param.split_path['sfu']['scope']['dev'])
        scope_test_data = proc.load_examples(
            param.split_path['sfu']['scope']['test'])
    elif param.dataset_name == 'bioscope_abstracts':
        cue_train_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['train'])
        cue_dev_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['dev'])
        cue_test_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['cue']['test'])
        scope_train_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['train'])
        scope_dev_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['dev'])
        scope_test_data = proc.load_examples(
            param.split_path['bioscope_abstracts']['scope']['test'])
    elif param.dataset_name == 'bioscope_full':
        cue_train_data = proc.load_examples(
            param.split_path['bioscope_full']['cue']['train'])
        cue_dev_data = proc.load_examples(
            param.split_path['bioscope_full']['cue']['dev'])
        cue_test_data = proc.load_examples(
            param.split_path['bioscope_full']['cue']['test'])
        scope_train_data = proc.load_examples(
            param.split_path['bioscope_full']['scope']['train'])
        scope_dev_data = proc.load_examples(
            param.split_path['bioscope_full']['scope']['dev'])
        scope_test_data = proc.load_examples(
            param.split_path['bioscope_full']['scope']['test'])
    elif param.dataset_name == 'sherlock':
        train_raw = proc.read_data(
            param.data_path['sherlock']['train'], dataset_name='sherlock')
        sherlock_test1 = proc.read_data(
            param.data_path['sherlock']['test1'], dataset_name='sherlock')
        sherlock_test2 = proc.read_data(
            param.data_path['sherlock']['test2'], dataset_name='sherlock')
        test_raw = SplitData([sherlock_test1.cues, sherlock_test2.cues], [sherlock_test1.scopes, sherlock_test2.scopes], [
                             sherlock_test1.non_cue_sents, sherlock_test2.non_cue_sents])
        dev_raw = proc.read_data(
            param.data_path['sherlock']['dev'], dataset_name='sherlock')
        cue_train_data = proc.create_examples(
            train_raw, cue_or_scope='cue', example_type='train')
        cue_dev_data = proc.create_examples(
            dev_raw, cue_or_scope='cue', example_type='dev')
        cue_test_data = proc.create_examples(
            test_raw, cue_or_scope='cue', example_type='test')
        scope_train_data = proc.create_examples(
            train_raw, cue_or_scope='scope', example_type='train')
        scope_dev_data = proc.create_examples(
            dev_raw, cue_or_scope='scope', example_type='dev')
        scope_test_data = proc.create_examples(
            test_raw, cue_or_scope='scope', example_type='test')

    if param.embedding == 'BERT':
        proc.get_tokenizer(data=None, is_bert=True, bert_path=param.bert_path)
    else:
        proc.get_tokenizer(data=cue_train_data, is_bert=False)

    cue_train_feature = proc.create_features(
        train_data, cue_or_scope='cue', max_seq_len=param.max_len)
    cue_dev_feature = proc.create_features(
        dev_data, cue_or_scope='cue', max_seq_len=param.max_len)
    cue_test_feature = proc.create_features(
        test_data, cue_or_scope='cue', max_seq_len=param.max_len)

    cue_train_ds = proc.create_dataset(
        train_feature, cue_or_scope='cue', example_type='train')
    cue_dev_ds = proc.create_dataset(
        dev_feature, cue_or_scope='cue', example_type='dev')
    cue_test_ds = proc.create_dataset(
        test_feature, cue_or_scope='cue', example_type='test')

    cue_train_sp = RandomSampler(train_ds)
    cue_train_dl = DataLoader(
        dataset=train_ds, batch_size=param.batch_size, sampler=cue_train_sp)

    scope_train_feature = proc.create_features(
        train_data, cue_or_scope='scope', max_seq_len=param.max_len)
    scope_dev_feature = proc.create_features(
        dev_data, cue_or_scope='scope', max_seq_len=param.max_len)
    scope_test_feature = proc.create_features(
        test_data, cue_or_scope='scope', max_seq_len=param.max_len)

    pipeline_test_feature = proc.create_features_pipeline(
        scope_test_feature, cue_model=None, non_cue_examples=cue_test_data, is_bert=param.embedding == 'BERT', max_seq_len=param.max_len)
    scope_train_ds = proc.create_dataset(
        train_feature, cue_or_scope='scope', example_type='train')
    scope_dev_ds = proc.create_dataset(
        dev_feature, cue_or_scope='scope', example_type='dev')
    scope_test_ds = proc.create_dataset(
        test_feature, cue_or_scope='scope', example_type='test')

    scope_train_sp = RandomSampler(train_ds)
    scope_train_dl = DataLoader(
        dataset=train_ds, batch_size=param.batch_size, sampler=scope_train_sp)

train_sp = RandomSampler(train_ds)
train_dl = DataLoader(train_ds, batch_size=param.batch_size, sampler=train_sp)
dev_dl = DataLoader(dev_ds, batch_size=param.batch_size)
tokenizer = proc.tokenizer


for run in range(param.num_runs):
    if param.task == 'cue':
        model = CueBert.from_pretrained(
            'bert-base-cased', cache_dir='bert_base_cased_model', num_labels=4)
        model = model.to(device)
        bert_param_optimizer = list(model.bert.named_parameters())
        cue_fc_param_optimizer = list(model.cue.named_parameters())
        sep_fc_param_optimizer = list(model.cue_sep.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            'lr': param.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': param.lr},
            {'params': [p for n, p in cue_fc_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': 0.0005},
            {'params': [p for n, p in cue_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': 0.0005},
            {'params': [p for n, p in sep_fc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            'lr': 0.0005},
            {'params': [p for n, p in sep_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': 0.0005}
        ]
    elif param.task == 'scope':
        model = ScopeBert.from_pretrained(
            'bert-base-cased', cache_dir='bert_base_cased_model', num_labels=param.label_dim)
        model = model.to(device)
        bert_param_optimizer = list(model.bert.named_parameters())
        scope_fc_param_optimizer = list(model.scope.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            'lr': param.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': param.lr},
            {'params': [p for n, p in scope_fc_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': 0.0005},
            {'params': [p for n, p in scope_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': 0.0005},
        ]
    t_total = int(len(train_dl) / 1 * 200)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=param.lr,
                         warmup=0.05, t_total=t_total)
    lr_scheduler = BERTReduceLROnPlateau(optimizer, lr=param.lr, mode='max', factor=0.5, patience=5,
                                         verbose=1, epsilon=1e-8, cooldown=0, min_lr=0, eps=1e-8)
    train_monitor = TrainingMonitor(file_dir='pics', arch='testmodel')
    early_stopping = EarlyStopping(patience=10, monitor='valid_f1')
    if param.task == 'cue':
        trainer = CueTrainer(n_gpu=1,
                            model=model,
                            logger=global_logger,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            label2id=cue_label2id,
                            criterion=ASLSingleLabel(),
                            training_monitor=train_monitor,
                            resume_path=None,
                            grad_clip=5.0,
                            gradient_accumulation_steps=1
                            )
    elif param.task == 'scope':
        trainer = ScopeTrainer(n_gpu=1,
                               model=model,
                               logger=global_logger,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               label2id=None,
                               criterion=nn.CrossEntropyLoss(),
                               training_monitor=train_monitor,
                               resume_path=None,
                               grad_clip=5.0,
                               gradient_accumulation_steps=1
                               )

    trainer.train(train_data=train_dl, valid_data=dev_dl,
                  epochs=200, is_bert=param.is_bert)
    print()
    '''model = Neg(vocab_size=tokenizer.dictionary.__len__(), tokenizer=tokenizer)
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
                ep_bar.set_postfix(
                    {"Ep": _, "Batch": step, "loss": loss.item()})
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
        r_f1 = r_scope(allinput, alltarget, allpred,
                       devdl.tokenizer, print_=False)
        t_flat = [i for t in alltarget for i in t]
        p_flat = [i for p in allpred for i in p]
        f1_scope(alltarget, allpred, False)
        conf_matrix = classification_report(
            [i for i in t_flat], [i for i in p_flat], output_dict=True)
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
        print(classification_report(
            [i for i in tt_flat], [i for i in pt_flat]))

    print(f"\n Run {run} Evaluating")
    if param.cross_test is False:
        testing(testdl)
    else:
        test_list = ['sfu', 'bioscope_abstracts', 'bioscope_full', 'sherlock']
        # test_list.pop(test_list.index(param.dataset_name))
        for tn in test_list:
            testing(test_dls[tn])
            print(f"Evaluation on {tn} is done!\n\n")

    if param.cross_test is False:
        testing(testdl)
    else:
        test_list = ['sfu', 'bioscope_abstracts', 'bioscope_full', 'sherlock']
        # test_list.pop(test_list.index(param.dataset_name))
        for tn in test_list:
            testing(test_dls[tn])
            print(f"Evaluation on {tn} is done!\n\n")
    '''
