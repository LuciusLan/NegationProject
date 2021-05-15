import time
import os
import random
from typing import List
from collections import Counter
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

import util
from util import pack_subword_pred
from params import param

DEVICE = torch.device('cuda')


class AverageMeter(object):
    '''
    computes and stores the average and current value
    Example:
        >>> loss = AverageMeter()
        >>> for step,batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metrics(pred,target)
        >>>     loss.update(raw_loss.item(),n = 1)
        >>> cur_loss = loss.avg
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed=1029):
    '''
    set the seed for the whole enviornment
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def target_weight_score(score_dict, target_class):
    f1 = []
    prec = []
    rec = []
    num_classes = len(target_class)
    total_support = 0
    for class_ in target_class:
        try:
            f1.append(score_dict[class_]['f1-score'] * score_dict[class_]['support'])
            prec.append(score_dict[class_]['precision'] * score_dict[class_]['support'])
            rec.append(score_dict[class_]['recall'] * score_dict[class_]['support'])
            total_support += score_dict[class_]['support']
        except KeyError:
            num_classes -= 1
            continue
    return np.sum(f1)/total_support, np.sum(prec)/total_support, np.sum(rec)/total_support

class CueTrainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.criterion = criterion
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device = DEVICE
        self.id2label = {y: x for x, y in label2id.items()}
        self.start_epoch = 1
        self.global_step = 0
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path + '\\checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_features, is_bert=True):
        pbar = tqdm(total=len(data_features), desc='Evaluating')
        valid_loss = AverageMeter()
        wrap_cue_pred = []
        wrap_cue_sep_pred = []
        wrap_cue_tar = []
        wrap_cue_sep_tar = []
        for step, f in enumerate(data_features):
            input_lens = f[4]
            num_labels = len(self.id2label)
            input_ids = f[0].to(self.device)
            padding_mask = f[1].to(self.device)
            subword_mask = f[-1].to(self.device)
            cues = f[2].to(self.device)
            cue_sep = f[3].to(self.device)
            active_padding_mask = padding_mask.view(-1) == 1

            self.model.eval()
            with torch.no_grad():
                if param.predict_cuesep:
                    cue_logits, cue_sep_logits = self.model(input_ids, padding_mask)
                    cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
                    cue_sep_loss = self.criterion(cue_sep_logits.view(-1, 5)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
                    loss = cue_loss + cue_sep_loss
                else:
                    cue_logits = self.model(input_ids, padding_mask)
                    loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
            valid_loss.update(val=loss.item(), n=input_ids.size(0))
            if is_bert:
                cue_pred, cue_tar = pack_subword_pred(cue_logits.detach().cpu(), cues.detach().cpu(), subword_mask.detach().cpu(), padding_mask.detach().cpu())
                if param.predict_cuesep:
                    cue_sep_pred, cue_sep_tar = pack_subword_pred(cue_sep_logits.detach().cpu(), cue_sep.detach().cpu(), subword_mask.detach().cpu(), padding_mask.detach().cpu())
            else:
                cue_pred = cue_logits.argmax()
                cue_tar = cues
                if param.predict_cuesep:
                    cue_sep_pred = cue_sep_logits.argmax()
                    cue_sep_tar = cue_sep

            if param.predict_cuesep:
                for i1, sent in enumerate(cue_pred):
                    for i2, _ in enumerate(sent):
                        wrap_cue_pred.append(cue_pred[i1][i2])
                        wrap_cue_tar.append(cue_tar[i1][i2])
                        wrap_cue_sep_pred.append(cue_sep_pred[i1][i2])
                        wrap_cue_sep_tar.append(cue_sep_tar[i1][i2])
            else:
                for i1, sent in enumerate(cue_pred):
                    for i2, _ in enumerate(sent):
                        wrap_cue_pred.append(cue_pred[i1][i2])
                        wrap_cue_tar.append(cue_tar[i1][i2])

            #wrap_cue_pred.append([cp for sent_ in cue_pred for cp in sent_])
            #wrap_cue_sep_pred.append([csp for sent_ in cue_sep_pred for csp in sent_])
            #wrap_cue_tar.append([ct for sent_ in cue_tar for ct in sent_])
            #wrap_cue_sep_tar.append([cst for sent_ in cue_sep_tar for cst in sent_])
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        #self.logger.info('\n Token Level F1:\n')
        cue_val_info = classification_report(wrap_cue_tar, wrap_cue_pred, output_dict=True, digits=5)
        if param.predict_cuesep:
            cue_sep_val_info = classification_report(wrap_cue_sep_tar, wrap_cue_sep_pred, output_dict=True, digits=5)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        if param.predict_cuesep:
            return cue_val_info, cue_sep_val_info
        else:
            return cue_val_info

    def train_epoch(self, data_loader, valid_data, epoch, is_bert=True, max_num_cue=4, eval_every=800):
        pbar = tqdm(total=len(data_loader), desc='Training')
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            bs = batch[0].size(0)
            num_labels = len(self.id2label)
            if param.matrix:
                input_ids, padding_mask, cues, cue_sep, input_len, subword_mask, cue_matrix = batch
            else:
                input_ids, padding_mask, cues, cue_sep, input_len, subword_mask = batch
            active_padding_mask = padding_mask.view(-1) == 1

            if param.cue_matrix:
                pad_matrix = []
                for i in range(bs):
                    tmp = padding_mask[i].clone()
                    tmp = tmp.view(param.max_len, 1)
                    tmp_t = tmp.transpose(0, 1)
                    mat = tmp * tmp_t
                    pad_matrix.append(mat)
                pad_matrix = torch.stack(pad_matrix, 0)
                active_padding_mask = pad_matrix.view(-1) == 1
                cue_logits = self.model(input_ids, padding_mask)[0]
            else:
                if param.predict_cuesep:
                    cue_logits, cue_sep_logits = self.model(input_ids, padding_mask, cue_teacher=cues)
                    cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
                    cue_sep_loss = self.criterion(cue_sep_logits.view(-1, max_num_cue+1)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
                    loss = cue_loss + cue_sep_loss
                else:
                    cue_logits = self.model(input_ids, padding_mask)
                    loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])

            #if len(self.n_gpu.split(",")) >= 2:
            #    loss = loss.mean()
            
            # scale the loss
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            # gradient clip
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=1)
            if step % eval_every == 0 and step > eval_every:
                if param.predict_cuesep:
                    cue_log, cue_sep_log = self.valid_epoch(valid_data, is_bert)
                    cue_f1 = target_weight_score(cue_log, ['0', '1', '2'])
                    cue_sep_f1 = target_weight_score(cue_sep_log, ['1', '2', '3', '4'])
                    metric = cue_f1[0] + cue_sep_f1[0]
                else:
                    cue_log = self.valid_epoch(valid_data, is_bert)
                    cue_f1 = target_weight_score(cue_log, ['0', '1', '2'])
                    metric = cue_f1[0]
                if hasattr(self.lr_scheduler,'epoch_step'):
                    self.lr_scheduler.epoch_step(metrics=metric, epoch=epoch)
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        info = {'loss': tr_loss.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def train(self, train_data, valid_data, epochs, is_bert=True):
        #seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data, valid_data, epoch, is_bert)

            if param.predict_cuesep:
                cue_log, cue_sep_log = self.valid_epoch(valid_data, is_bert)
                cue_f1 = target_weight_score(cue_log, ['1'])
                cue_sep_f1 = target_weight_score(cue_sep_log, ['1', '2', '3', '4'])
                logs = {'loss': train_log['loss'], 'val_cue_f1': cue_f1[0], 'val_cuesep_f1': cue_sep_f1[0]}
                score = cue_f1[0]+cue_sep_f1[0]
            else:
                cue_log = self.valid_epoch(valid_data, is_bert)
                cue_f1 = target_weight_score(cue_log, ['0', '1', '2'])
                logs = {'loss': train_log['loss'], 'val_cue_f1': cue_f1[0]}
                score = cue_f1[0]
                
            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(logs)
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=score, epoch=epoch)
            # save log
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=score)
                self.model_checkpoint.bert_epoch_step(current=score, state=state)
                
            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=score)
                if self.early_stopping.stop_training:
                    break

class ScopeTrainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.criterion = criterion
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device = DEVICE
        #self.id2label = {y: x for x, y in label2id.items()}
        self.start_epoch = 1
        self.global_step = 0
        self.arc_loss = torch.nn.BCELoss()
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path + '/checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_features, is_bert):
        """
        batch shape:
            [bs*][input_ids, padding_mask, scopes, input_len, cues, subword_mask]
        """
        pbar = tqdm(total=len(data_features), desc='Evaluating')
        valid_loss = AverageMeter()
        wrap_scope_pred = []
        wrap_scope_tar = []
        sm_pred = []
        sm_tar = []
        cm_pred = []
        cm_tar = []
        for step, f in enumerate(data_features):
            num_labels = param.label_dim
            input_ids = f[0].to(self.device)
            padding_mask = f[1].to(self.device)
            input_lens = f[3]
            if not param.multi:
                scopes = f[2].to(self.device)
                cues = f[4].to(self.device)
            else:
                scopes = f[2]
                cues = f[4]
            subword_mask = f[5].to(self.device)
            bs = f[0].size(0)
            if param.matrix:
                scopes_matrix = f[-1].to(self.device)
            self.model.eval()
            with torch.no_grad():
                active_padding_mask = padding_mask.view(-1) == 1
                if param.matrix:
                    pad_matrix = []
                    for i in range(bs):
                        tmp = padding_mask[i].clone()
                        tmp = tmp.view(param.max_len, 1)
                        tmp_t = tmp.transpose(0, 1)
                        mat = tmp * tmp_t
                        pad_matrix.append(mat)
                    pad_matrix = torch.stack(pad_matrix, 0)
                    active_padding_mask = pad_matrix.view(-1) == 1
                    if not param.augment_cue and param.task != 'joint':
                        scope_logits = self.model([input_ids, cues], padding_mask)[0]
                    else:
                        scope_logits = self.model(input_ids, padding_mask)[0]
                    
                    if param.fact:
                        # Factorized (arc and label classifier)
                        arc_targets = util.label_to_arc_matrix(scopes_matrix)
                        arc_logits, label_logits = scope_logits
                        arc_logit_masked = arc_logits.view(-1)[active_padding_mask]
                        arc_target_masked = arc_targets.view(-1)[active_padding_mask]
                        arc_mask = arc_logits.view(-1) > 0
                        label_logit_masked = label_logits.view(-1, num_labels)[arc_mask]
                        label_target_masked = scopes_matrix.view(-1)[arc_mask]
                        arc_loss = self.arc_loss(arc_logit_masked, arc_target_masked.float())
                        label_loss = self.criterion(label_logit_masked, label_target_masked)
                        loss = arc_loss + label_loss
                    else:
                        logits_masked = scope_logits.view(-1, num_labels)[active_padding_mask]
                        target_masked = scopes_matrix.view(-1)[active_padding_mask]
                        loss = self.criterion(logits_masked, target_masked)
                else:
                    if not param.augment_cue and param.task != 'joint':
                        scope_logits = self.model([input_ids, cues], padding_mask)[0]
                    else:
                        scope_logits = self.model(input_ids, padding_mask)[0]
                    active_padding_mask = padding_mask.view(-1) == 1
                    loss = self.criterion(scope_logits.view(-1, num_labels)[active_padding_mask.view(-1)], scopes.view(-1)[active_padding_mask.view(-1)])
            valid_loss.update(val=loss.item(), n=input_ids.size(0))

            if is_bert:
                if param.matrix:
                    if param.multi:
                        scope_pred = []
                        scope_tar = []
                        temp_scope_pred, temp_cue_pred = util.multi_matrix_decode_toseq(scope_logits, pad_matrix)
                        if param.dataset_name != 'sherlock' and param.dataset_name != 'sfu':
                            for i in range(bs):
                                sp, st = util.handle_eval_multi(temp_scope_pred[i], scopes[i], temp_cue_pred[i], cues[i])
                                for j, _ in enumerate(sp):
                                    pred, tar = pack_subword_pred(sp[j].detach().cpu().unsqueeze(0), st[j].detach().cpu().unsqueeze(0),
                                                            subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                                    scope_pred.append(pred)
                                    scope_tar.append(tar)
                        else:
                            for i in range(bs):
                                sp, st = util.handle_eval_multi(temp_scope_pred[i], scopes[i], temp_cue_pred[i], cues[i])
                                scope_pred.append(sp)
                                scope_tar.append(st)
                    else:
                        label_logits = scope_logits[1] if param.fact else scope_logits
                        tmp_scope_pred = util.matrix_decode_toseq(label_logits, pad_matrix)                    
                        scope_pred = []
                        scope_tar = []
                        for i in range(bs):
                            pred, tar = pack_subword_pred(tmp_scope_pred[i].detach().cpu().unsqueeze(0), scopes[i].detach().cpu().unsqueeze(0),
                                                        subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                            scope_pred.append(pred[0])
                            scope_tar.append(tar[0])
                else:
                    scope_pred, scope_tar = pack_subword_pred(scope_logits.detach().cpu(), scopes.detach().cpu(), subword_mask.detach().cpu(), padding_mask.cpu())

            else:
                scope_pred = scope_logits.argmax()
                scope_tar = scopes
                scope_pred = [e.tolist() for e in scope_pred]
                scope_tar = [e.tolist() for e in scope_tar]
            if param.task == 'joint' and not param.multi:
                new_pred = []
                for i, seq in enumerate(input_ids):
                    new_pred.append(util.handle_eval_joint(scope_pred[i], scope_tar[i]))
                scope_pred = new_pred
            if param.dataset_name == 'sherlock' or param.dataset_name == 'sfu':
                # Post process for Sherlock, separate "n't" words and mark affixal cues
                new_pred = []
                new_tar = []
                if param.multi:
                    for i, seq in enumerate(input_ids):
                        text_seq = data_features.tokenizer.convert_ids_to_tokens(seq)
                        text_string = data_features.tokenizer.decode(seq)
                        for j, _ in enumerate(scope_pred[i]):
                            p, t = pack_subword_pred(scope_pred[i][j].detach().cpu().unsqueeze(0), scope_tar[i][j].detach().cpu().unsqueeze(0),
                                              subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                            new_pred.append(util.postprocess_sher(p[0], cues[i], subword_mask[i], input_lens[i], text_seq, text_string, scope_tar=scope_tar[i][j], sp=sp, st=st))
                            new_tar.append(util.postprocess_sher(t[0], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))
                else:
                    for i, seq in enumerate(input_ids):
                        text_seq = data_features.tokenizer.convert_ids_to_tokens(seq)
                        text_string = data_features.tokenizer.decode(seq)
                        new_pred.append(util.postprocess_sher(scope_pred[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))
                        new_tar.append(util.postprocess_sher(scope_tar[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))

                for i1, sent in enumerate(new_pred):
                    sp, st = util.full_scope_match(new_pred[i1], new_tar[i1])
                    cp, ct = util.cue_match(new_pred[i1], new_tar[i1])
                    sm_pred.append(sp)
                    sm_tar.append(st)
                    cm_pred.append(cp)
                    cm_tar.append(ct)
                    for i2, _ in enumerate(sent):
                        wrap_scope_pred.append(new_pred[i1][i2])
                        wrap_scope_tar.append(new_tar[i1][i2])
            else:
                for i1, sent in enumerate(scope_pred):
                    sp, st = util.full_scope_match(scope_pred[i1], scope_tar[i1])
                    cp, ct = util.cue_match(scope_pred[i1], scope_tar[i1])
                    sm_pred.append(sp)
                    sm_tar.append(st)
                    cm_pred.append(cp)
                    cm_tar.append(ct)
                    for i2, _ in enumerate(sent):
                        wrap_scope_pred.append(scope_pred[i1][i2])
                        wrap_scope_tar.append(scope_tar[i1][i2])

            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        if param.label_dim > 3:
            cue_f1 = f1_score(cm_tar, cm_pred)
        else:
            cue_f1 = 1
        scope_match = f1_score(sm_tar, sm_pred)
        if (param.mark_cue or param.matrix) and ('bioscope' in param.dataset_name):
            # For bioscope and sfu, include "cue" into scope if predicting cue
            wrap_scope_tar = [e if e != 3 else 1 for e in wrap_scope_tar]
            wrap_scope_pred = [e if e != 3 else 1 for e in wrap_scope_pred]
        scope_val_info = classification_report(wrap_scope_tar, wrap_scope_pred, output_dict=True, digits=5)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return scope_val_info, cue_f1, scope_match

    def train_epoch(self, data_loader, valid_data):
        pbar = tqdm(total=len(data_loader), desc='Training')
        num_labels = param.label_dim
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
            if param.matrix:
                input_ids, padding_mask, scopes, input_len, cues, subword_mask, scopes_matrix = batch
            else:
                input_ids, padding_mask, scopes, input_len, cues, subword_mask = batch
            bs = batch[0].size(0)
            active_padding_mask = padding_mask.view(-1) == 1
            if param.matrix:
                pad_matrix = []
                for i in range(bs):
                    tmp = padding_mask[i].clone()
                    tmp = tmp.view(param.max_len, 1)
                    tmp_t = tmp.transpose(0, 1)
                    mat = tmp * tmp_t
                    pad_matrix.append(mat)
                pad_matrix = torch.stack(pad_matrix, 0)
                active_padding_mask = pad_matrix.view(-1) == 1
                if not param.augment_cue and param.task != 'joint':
                    scope_logits = self.model([input_ids, cues], padding_mask)[0]
                else:
                    scope_logits = self.model(input_ids, padding_mask)[0]
                
                if param.fact:
                    # Factorized (arc and label classifier)
                    arc_targets = util.label_to_arc_matrix(scopes_matrix)
                    arc_logits, label_logits = scope_logits
                    arc_logit_masked = arc_logits.view(-1)[active_padding_mask]
                    arc_target_masked = arc_targets.view(-1)[active_padding_mask]
                    arc_mask = arc_logits.view(-1) > 0
                    label_logit_masked = label_logits.view(-1, num_labels)[arc_mask]
                    label_target_masked = scopes_matrix.view(-1)[arc_mask]
                    arc_loss = self.arc_loss(arc_logit_masked, arc_target_masked.float())
                    label_loss = self.criterion(label_logit_masked, label_target_masked)
                    loss = arc_loss + label_loss
                else:
                    # Unfactorized (Single classifier for both arc and label)
                    logits_masked = scope_logits.view(-1, num_labels)#[active_padding_mask]
                    target_masked = scopes_matrix.view(-1)#[active_padding_mask]
                    loss = self.criterion(logits_masked, target_masked)
            else:
                if not param.augment_cue and param.task != 'joint':
                    scope_logits = self.model([input_ids, cues], padding_mask)[0]
                else:
                    scope_logits = self.model(input_ids, padding_mask)[0]
                loss = self.criterion(scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask])

            #if len(self.n_gpu.split(",")) >= 2:
            #    loss = loss.mean()
            
            # scale the loss
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            # gradient clip
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=1)
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        info = {'loss': tr_loss.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def train(self, train_data, valid_data, epochs, is_bert=False):
        #seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data, valid_data)
            scope_log, cue_f1, scope_match = self.valid_epoch(valid_data, is_bert)
            scope_f1 = target_weight_score(scope_log, ['1'])
            logs = {'loss': train_log['loss'], 'val_scope_token_f1': scope_f1[0]}
            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            if param.task == 'joint':
                self.logger.info('cue_f1 %f', cue_f1)
            self.logger.info(logs)
            self.logger.info('scope match %f', scope_match)
            #self.logger.info("The entity scores of valid data : ")
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=scope_f1[0], epoch=epoch)
            # save log
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=logs['val_scope_token_f1'])
                if self.early_stopping.stop_training:
                    break
