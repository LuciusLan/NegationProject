import time
import os
import random
from collections import Counter
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from util import pack_subword_pred

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

    def test_epoch(self, data_features, is_bert):
        pbar = tqdm(total=len(data_features), desc='Testing')
        test_loss = AverageMeter()
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

            self.model.eval()
            with torch.no_grad():
                cue_logits, cue_sep_logits = self.model(input_ids, padding_mask)
                active_padding_mask = padding_mask.view(-1) == 1
                cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
                cue_sep_loss = self.criterion(cue_sep_logits.view(-1, 5)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
                loss = cue_loss + cue_sep_loss
            test_loss.update(val=loss.item(), n=input_ids.size(0))
            if is_bert:
                cue_pred, cue_tar = pack_subword_pred(cue_logits.detach().cpu(), cues.detach().cpu(), subword_mask.detach().cpu())
                cue_sep_pred, cue_sep_tar = pack_subword_pred(cue_sep_logits.detach().cpu(), cue_sep.detach().cpu(), subword_mask.detach().cpu())
            else:
                cue_pred = cue_logits.argmax()
                cue_sep_pred = cue_sep_logits.argmax()
                cue_tar = cues
                cue_sep_tar = cue_sep

            for i1, sent in enumerate(cue_pred):
                for i2, _ in enumerate(sent):
                    wrap_cue_pred.append(cue_pred[i1][i2])
                    wrap_cue_sep_pred.append(cue_sep_pred[i1][i2])
                    wrap_cue_tar.append(cue_tar[i1][i2])
                    wrap_cue_sep_tar.append(cue_sep_tar[i1][i2])

            #wrap_cue_pred.append([cp for sent_ in cue_pred for cp in sent_])
            #wrap_cue_sep_pred.append([csp for sent_ in cue_sep_pred for csp in sent_])
            #wrap_cue_tar.append([ct for sent_ in cue_tar for ct in sent_])
            #wrap_cue_sep_tar.append([cst for sent_ in cue_sep_tar for cst in sent_])
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        self.logger.info('\n Test Token Level F1:\n')
        cue_test_info = classification_report(wrap_cue_tar, wrap_cue_pred, output_dict=True, digits=5)
        cue_sep_test_info = classification_report(wrap_cue_sep_tar, wrap_cue_sep_pred, output_dict=True, digits=5)
        self.logger.info('Cue:')
        self.logger.info(cue_test_info)
        self.logger.info('Cue :')
        self.logger.info(cue_sep_test_info)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return cue_test_info, cue_sep_test_info

    def valid_epoch(self, data_features, is_bert):
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

            self.model.eval()
            with torch.no_grad():
                cue_logits, cue_sep_logits = self.model(input_ids, padding_mask)
                active_padding_mask = padding_mask.view(-1) == 1
                cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
                cue_sep_loss = self.criterion(cue_sep_logits.view(-1, 5)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
                loss = cue_loss + cue_sep_loss
            valid_loss.update(val=loss.item(), n=input_ids.size(0))
            if is_bert:
                cue_pred, cue_tar = pack_subword_pred(cue_logits.detach().cpu(), cues.detach().cpu(), subword_mask.detach().cpu())
                cue_sep_pred, cue_sep_tar = pack_subword_pred(cue_sep_logits.detach().cpu(), cue_sep.detach().cpu(), subword_mask.detach().cpu())
            else:
                cue_pred = cue_logits.argmax()
                cue_sep_pred = cue_sep_logits.argmax()
                cue_tar = cues
                cue_sep_tar = cue_sep

            for i1, sent in enumerate(cue_pred):
                for i2, _ in enumerate(sent):
                    wrap_cue_pred.append(cue_pred[i1][i2])
                    wrap_cue_sep_pred.append(cue_sep_pred[i1][i2])
                    wrap_cue_tar.append(cue_tar[i1][i2])
                    wrap_cue_sep_tar.append(cue_sep_tar[i1][i2])

            #wrap_cue_pred.append([cp for sent_ in cue_pred for cp in sent_])
            #wrap_cue_sep_pred.append([csp for sent_ in cue_sep_pred for csp in sent_])
            #wrap_cue_tar.append([ct for sent_ in cue_tar for ct in sent_])
            #wrap_cue_sep_tar.append([cst for sent_ in cue_sep_tar for cst in sent_])
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        self.logger.info('\n Token Level F1:\n')
        cue_val_info = classification_report(wrap_cue_tar, wrap_cue_pred, output_dict=True, digits=5)
        cue_sep_val_info = classification_report(wrap_cue_sep_tar, wrap_cue_sep_pred, output_dict=True, digits=5)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return cue_val_info, cue_sep_val_info

    def train_epoch(self, data_loader, valid_data, max_num_cue=4):
        pbar = tqdm(total=len(data_loader), desc='Training')
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            num_labels = len(self.id2label)
            input_ids, padding_mask, cues, cue_sep, input_len, subword_mask = batch
            cue_logits, cue_sep_logits = self.model(input_ids, padding_mask)
            active_padding_mask = padding_mask.view(-1) == 1
            cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
            cue_sep_loss = self.criterion(cue_sep_logits.view(-1, max_num_cue+1)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
            loss = cue_loss + cue_sep_loss

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
            cue_log, cue_sep_log = self.valid_epoch(valid_data, is_bert)

            cue_f1 = target_weight_score(cue_log, ['0', '1', '2'])
            cue_sep_f1 = target_weight_score(cue_sep_log, ['1', '2', '3', '4'])
            logs = {'loss': train_log['loss'], 'val_cue_f1': cue_f1[0], 'val_cuesep_f1': cue_sep_f1[0]}
            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            #self.logger.info(show_info)
            #self.logger.info("The entity scores of valid data : ")
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=cue_f1[0]+cue_sep_f1[0], epoch=epoch)
            # save log
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=logs['val_cue_f1']+logs['val_cuesep_f1'])
                if self.early_stopping.stop_training:
                    break
