from typing import List, Tuple, T
import logging
import math
import json
import re
import six
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from params import param

def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    # log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger
    
global_logger = init_logger(log_file=f'{param.model_name}.log')

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def label_to_arc_matrix(mat: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(mat)
    one = torch.ones_like(mat)
    tmp = torch.where(mat != 3, mat, one)
    tmp = torch.where(tmp == 1, tmp, zero)
    return tmp

def matrix_decode_toseq(logits: torch.Tensor, pad: List[torch.Tensor], islogit=True, mode='row'):
    """
    Decode the link-matrix to a prediction sequence (single cue case)
    Assumed single cue (multiword cue included), and high quality prediction. 
    That is, only first row containing cue indicator will be extracted

    Params:
        logits(Tensor): [batch_size, seq_length, seq_length, num_classes]
        pad(List[Tensor]): padding matrix
        isLogit(boolean): drop a dimension when input is target (isLogit=False)
        mode: ('row', 'col', 'max', 'mean') rule to select target sequence. 
            row: return the first row containing a cue
            col: return the first column containing a cue
            max: return the max among all candidates.
            mean: return the mean of all candidates
    Retruns:
        target_seq(Tensor): [batch_size, seq_length, num_classes]
    """
    bs = logits.size(0)
    num_labels = logits.size(-1)
    padding_mask = pad == 1
    if islogit:
        batch = []
        for i in range(bs):
            tmp = logits.view(bs, param.max_len, param.max_len, param.label_dim)[i][padding_mask[i]]
            dim = int(math.sqrt(tmp.size(0)))
            tmp = tmp.view(dim, dim, -1)
            batch.append(tmp)
        mat_tmp = [e.argmax(-1).tolist() for e in batch]
    else:
        batch = []
        for i in range(bs):
            tmp = logits.view(bs, param.max_len, param.max_len)[i][padding_mask[i]]
            dim = int(math.sqrt(tmp.size(0)))
            tmp = tmp.view(dim, dim)
            batch.append(tmp)
        mat_tmp = [e.tolist() for e in batch]
    def pos_first_cue_row(mat):
        rp = 0
        cp = 0
        for row_i, row_e in enumerate(mat):
            for col_i, col_e in enumerate(row_e):
                if col_e == 3:
                    rp = row_i + 1
                    cp = col_i + 1
                    return rp, cp
        return 0, 0

    def all_cue_pos(mat):
        rp = []
        cp = []
        if param.cue_mode == 'diag':
            for row_i, row_e in enumerate(mat):
                for col_i, col_e in enumerate(row_e):
                    if col_e == 3:
                        rp.append(row_i)
                        cp.append(col_i)
        elif param.cue_mode == 'root':
            for i, e in enumerate(mat[0]):
                if e == 1:
                    rp.append(i)
        rp = list(set(rp))
        cp = list(set(cp))
        return rp, cp
                
    pred_scopes = []
    for i, m in enumerate(mat_tmp):
        if mode == 'row':
            tmp = []
            rows, cols = all_cue_pos(mat_tmp[i])
            if len(rows) == 0 and len(cols) == 0:
                tmp.append(batch[i][0])
                tmp.append(batch[i][1])
            else:
                for ri in rows:
                    tmp.append(batch[i][ri])
            tmp = torch.stack(tmp)
            if islogit:
                curr_pred = tmp.mean(0)
                if param.cue_mode == 'root' and len(rows) != 0:
                    # Force the predicted scope tensor's cue position to be decoding to cue
                    mock_cue = [0 for i in range(num_labels+1)]
                    mock_cue[3] = 1
                    mock_cue = torch.Tensor(mock_cue).to(curr_pred.device)
                    z = torch.zeros([curr_pred.size(0),1]).to(curr_pred.device)
                    curr_pred = torch.cat([curr_pred, z], 1)
                    curr_pred[rows] = mock_cue
                pred_scopes.append(curr_pred)
            else:
                pred_scopes.append(tmp.float().mean(0).long())
        elif mode == 'col':
            rp, cp = pos_first_cue_row(m)
            pred_scopes.append(batch[i][:][cp])
        elif mode == 'max':
            tmp = []
            rows, cols = all_cue_pos(mat_tmp[i])
            if len(rows) == 0 and len(cols) == 0:
                tmp.append(batch[i][0])
                tmp.append(batch[i][1])
            else:
                for ri in rows:
                    tmp.append(batch[i][ri])
                for ci in cols:
                    tmp.append(batch[i][:][ci])
            tmp = torch.stack(tmp)
            if islogit:
                pred_scopes.append(tmp.max(0).values)
            else:
                pred_scopes.append(tmp.float().max(0).values.long())
        elif mode == 'mean':
            tmp = []
            rows, cols = all_cue_pos(mat_tmp[i])
            if len(rows) == 0 and len(cols) == 0:
                tmp.append(batch[i][0])
                tmp.append(batch[i][1])
            else:
                for ri in rows:
                    tmp.append(batch[i][ri])
                for ci in cols:
                    tmp.append(batch[i][:][ci])
            tmp = torch.stack(tmp)
            if islogit:
                pred_scopes.append(tmp.mean(0))
            else:
                pred_scopes.append(tmp.float().mean(0).long())              
    return pred_scopes

def multi_matrix_decode_toseq(logits: torch.Tensor, pad: List[torch.Tensor], cues: torch.Tensor=None):
    bs = logits.size(0)
    num_labels = logits.size(-1)
    padding_mask = pad == 1
    batch = []
    for i in range(bs):
        tmp = logits.view(bs, param.max_len, param.max_len, param.label_dim)[i][padding_mask[i]]
        dim = int(math.sqrt(tmp.size(0)))
        tmp = tmp.view(dim, dim, -1)
        batch.append(tmp)
    mat_tmp = [e.argmax(-1).tolist() for e in batch]

    pred_scopes = []
    if cues is None:
        pred_cues = []
        for i, mat in enumerate(mat_tmp):
            stmp = []
            ctmp = []
            pos = []
            for row_i, row_e in enumerate(mat):
                for col_i, col_e in enumerate(row_e):
                    if col_e == 3:
                        pos.append(row_i)
            if len(pos) == 0:
                stmp.append(batch[i][0])
                ctmp.append([0])
            candidates = [mat[i]==1 for i in pos]
            candidates = [e.argmax(-1) for e in candidates]
            picked = filter_same_cue(candidates)
            for e in picked:
                if isinstance(e, list):
                    stmp.append(batch[i][e[0]])
                    ctmp.append(e)
                else:
                    stmp.append(batch[i][e])
                    ctmp.append([e])
            pred_scopes.append(stmp)
            pred_cues.append(ctmp)
    else:
        pred_cues = None
        
    return pred_scopes, pred_cues
        
def filter_same_cue(candidates: List, thres=0.95):
    """
    Using Non-Maximum Suppression (NMS) to filter out highly similar scopes (that possibly belong to same cue)
    """
    pick = []
    next_ = list(range(len(candidates)))
    while len(next_) > 0:
        last = next_[-1]
        pick.append(last)
        next_.pop(-1)
        if len(next_) == 0:
            break
        for i in range(last):
            f = f1_score(candidates[last], candidates[i])
            if f > thres:
                pick[-1] = [pick[-1], i]
                next_.pop(i)
    return pick

def handle_eval_multi(scope_pred, scope_tar, cue_pred, cue_tar):
    if len(scope_pred) == 1:
        return [scope_pred], [scope_tar]
    else:
        cue_matches = []
        for i, cp in enumerate(cue_pred):
            match = -1
            if isinstance(cp, list):
                for j, cg in enumerate(cue_tar):
                    for c in cp:
                        if cg[c] == 1:
                            match = j
                cue_matches.append(match)
        preds = []
        tars = []
        for i, cm in enumerate(cue_matches):
            if cm == -1:
                # Predicting non-existing cue, mark all predicted scope token as false positive
                preds.append(scope_pred[i])
                tars.append([2 for e in scope_pred[i]])
            else:
                preds.append(scope_pred[i])
                tars.append(scope_tar[cm])
        return preds, tars
    
def handle_eval_joint(pred, tar):
    tar_cue_pos = []
    pred_cue_pos = []
    for i, e in enumerate(tar):
        if e == 3:
            tar_cue_pos.append(i)
    for i, e in enumerate(pred):
        if e == 3:
            pred_cue_pos.append(i)
    tar_cue_pos = set(tar_cue_pos)
    pred_cue_pos = set(pred_cue_pos)
    match = tar_cue_pos.intersection(pred_cue_pos)
    if len(match) == 0 and (len(tar_cue_pos) != 0 or len(pred_cue_pos) != 0):
        if len(pred_cue_pos) > 0 and len(tar_cue_pos) == 0:
            # if predicting a cue but no negation in ground truth, mark all predicted
            # scope as false positive (set all tar negative) 
            # (Actually no need to do this, as the target scope will already be all 2)
            pass
        elif len(pred_cue_pos) == 0 and len(tar_cue_pos) > 0:
            # if failed to predict a cue / predicting wrong cue
            # mark all predicted scope as false negative
            pred = [2 for e in tar]
        else:
            pred = [2 for e in tar]
    return pred

def cue_match(pred, tar):
    tar_cue_pos = []
    pred_cue_pos = []
    for i, e in enumerate(tar):
        if e == 3:
            tar_cue_pos.append(i)
    for i, e in enumerate(pred):
        if e == 3:
            pred_cue_pos.append(i)
    tar_cue_pos = set(tar_cue_pos)
    pred_cue_pos = set(pred_cue_pos)
    match = tar_cue_pos.intersection(pred_cue_pos)
    if len(tar_cue_pos) != 0:
        if tar_cue_pos == pred_cue_pos:
            p = 1
            t = 1
        else:
            p = 0
            t = 1
    else:
        if tar_cue_pos == pred_cue_pos:
            p = 0
            t = 0
        else:
            p = 1
            t = 0
    return p, t

def full_scope_match(pred, tar):
    tar_scope_pos = []
    pred_scope_pos = []
    for i, e in enumerate(tar):
        if e == 1:
            tar_scope_pos.append(i)
    for i, e in enumerate(pred):
        if e == 1:
            pred_scope_pos.append(i)
    if len(tar_scope_pos) != 0:
        if pred_scope_pos == tar_scope_pos:
            p = 1
            t = 1
        else:
            p = 0
            t = 1
    else:
        if pred_scope_pos == tar_scope_pos:
            p = 0
            t = 0
        else:
            p = 1
            t = 0
    return p, t

def pack_subword_pred(logits, targets, subword_mask, padding_mask) -> Tuple[T, T]:
    """
    Apply subword mask to restore the original sentence and labels

    Params:
        logits(Tensor): [batch_size, seq_length, num_classes]
        targets(Tensor): [batch_size, seq_length]
        subword_mask(Tensor): [bs, seq_length]
        padding_mask(Tensor): [bs, seq_length]
    Returns:
        (prediction, actual_labels)
    """
    bs = logits.size(0)
    logits = logits.numpy()
    targets = targets.numpy()
    subword_mask = subword_mask.numpy()
    logits = [list(p) for p in logits]

    actual_logits = []
    actual_label_ids = []
    for i in range(bs):
        pad = padding_mask[i] == 1
        logit = logits[i]
        label = targets[i]
        mask = subword_mask[i][pad]

        actual_label_ids.append(
            [i for i, j in zip(label, mask) if j == 1])
        word_pieces = []
        new_logits = []
        in_split = 0
        head = 0
        for i, j in zip(logit, mask):
            if j == 1:
                if in_split == 1:
                    word_pieces.insert(0, head)
                    mode_pred = np.argmax(np.average(word_pieces, axis=0), axis=0)
                    if len(new_logits) > 0:
                        new_logits[-1] = mode_pred
                    else:
                        new_logits.append(mode_pred)
                    word_pieces = []
                    in_split = 0
                new_logits.append(np.argmax(i))
                head = i
            if j == 0:
                word_pieces.append(i)
                in_split = 1
        if in_split == 1:
            word_pieces.insert(0, head)
            mode_pred = np.argmax(np.average(
                np.array(word_pieces), axis=0), axis=0)
            if len(new_logits) > 0:
                new_logits[-1] = mode_pred
            else:
                new_logits.append(mode_pred)
        actual_logits.append(new_logits)

    return actual_logits, actual_label_ids

def postprocess_sher(input_id, scope_pred, cue, subword_mask, input_len, text_seq, text_string):
    new_pred = scope_pred
    packed_text = pack_subword_text(text_seq, subword_mask, input_len)
    if 'n\'t' in text_string:
        for i, word in enumerate(packed_text):
            if 'n\'t' in word:
                if scope_pred[i] == 3:
                    # if n't word appears and classified as cue, mark the root as scope
                    new_pred.insert(i, 1)
                else:
                    # if not cue, simply split it
                    new_pred.insert(i, scope_pred[i])
    if 4 in cue:
        # for affixal cue, mark it as part of scope, to simulate separation of affixes 
        # (root being scope, affix being cue)
        packed_cue = pack_subword_text(cue, subword_mask, input_len)
        if param.task == 'joint':
            # For joint model, don't leak gold cue information, use simple rules instead.
            is_aff = r'(^(un)|(non)|(dis)|(in)|(ir)|(im)|(il))|(less$)'
            for i, word in enumerate(packed_text):
                if re.search(is_aff, word, re.I) is not None and new_pred[i] == 3:
                    new_pred[i] = 1
        else:
            # For pipeline model, use input cue information (either predicted or gold) to get better result
            for i, c in enumerate(packed_cue):
                if c == 4 and new_pred[i] == 3:
                    new_pred[i] = 1
    return new_pred


def pack_subword_text(seq, subword_mask, input_len) -> Tuple[T, T]:
    """
    Apply subword mask to restore the original sentence and labels

    Params:
        seq (List[str|int]): text list 
        subword_mask (Tensor | List): subword mask array
        input_len: length of the input sequence (indicating the padding margin)
    
    Retruns:
        
    """
    subword_mask = subword_mask.tolist()
    input_len = input_len.tolist()
    new_seq = []
    for i , (e, sub) in enumerate(zip(seq, subword_mask)):
        if i >= input_len:
            break
        if sub == 1:
            new_seq.append(e)
        elif sub == 0:
            if type(e) == str:
                new_seq[-1] += e
            elif type(e) == int:
                continue

    return new_seq

def pack_subword(seq, subword_mask) -> Tuple[T, T]:
    """
    Apply subword mask to restore the original sentence and labels

    Params:
        logits(Tensor): [batch_size, seq_length, num_classes]
        targets(Tensor): [batch_size, seq_length]
    
    Retruns:
        (prediction, actual_labels)
    """
    seq = np.array(seq.squeeze())
    subword_mask = np.array(subword_mask)

    new_seq = []
    for i, j in zip(seq, subword_mask):
        if j == 1:
            new_seq.append(i)

    return new_seq


class TrainingMonitor():
    def __init__(self, file_dir, arch, add_test=False):
        '''
        :param startAt: 重新开始训练的epoch点
        '''
        if isinstance(file_dir, Path):
            pass
        else:
            file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.H = {}
        self.add_test = add_test
        self.json_path = file_dir / (arch + "_training_monitor.json")

    def reset(self,start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():
                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v, np.float):
                v = round(float(v), 4)
            l.append(v)
            self.H[k] = l

        # 写入文件
        if self.json_path is not None:
            save_json(data = self.H,file_path=self.json_path)

        # 保存train图像
        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key.upper()}') for key in self.H.keys()}

        if len(self.H["loss"]) > 1:
            # 指标变化
            # 曲线
            # 需要成对出现
            keys = [key for key, _ in self.H.items() if '_' not in key]
            for key in keys:
                N = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, self.H[key], label=f"train_{key}")
                # plt.plot(N, self.H[f"valid_{key}"], label=f"valid_{key}")
                if self.add_test:
                    plt.plot(N, self.H[f"test_{key}"], label=f"test_{key}")
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[key])}]")
                plt.savefig(str(self.paths[key]))
                plt.close()

class ModelCheckpoint(object):
    '''
    模型保存，两种模式：
    1. 直接保存最好模型
    2. 按照epoch频率保存模型
    '''
    def __init__(self, checkpoint_dir,
                 monitor,
                 arch,mode='min',
                 epoch_freq=1,
                 best = None,
                 save_best_only = True):
        if isinstance(checkpoint_dir,Path):
            checkpoint_dir = checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        assert checkpoint_dir.is_dir()
        self.base_path = checkpoint_dir
        self.arch = arch
        self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only

        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best

        if save_best_only:
            self.model_name = f"BEST_{arch}_MODEL.pth"

    def epoch_step(self, state,current):
        '''
        正常模型
        :param state: 需要保存的信息
        :param current: 当前判断指标
        :return:
        '''
        # 是否保存最好模型
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                print(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                best_path = self.base_path/ self.model_name
                torch.save(state, str(best_path))
        # 每隔几个epoch保存下模型
        else:
            filename = self.base_path / f"EPOCH_{state['epoch']}_{state[self.monitor]}_{self.arch}_MODEL.pth"
            if state['epoch'] % self.epoch_freq == 0:
                print(f"\nEpoch {state['epoch']}: save model to disk.")
                torch.save(state, str(filename))

    def bert_epoch_step(self, state,current):
        '''
        适合bert类型模型，适合pytorch_transformer模块
        :param state:
        :param current:
        :return:
        '''
        model_to_save = state['model']
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                global_logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                model_to_save.save_pretrained(str(self.base_path))
                #output_config_file = self.base_path / 'configs.json'
                #with open(str(output_config_file), 'w') as f:
                #    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state,self.base_path / 'checkpoint_info.bin')
        else:
            if state['epoch'] % self.epoch_freq == 0:
                save_path = self.base_path / f"checkpoint-epoch-{state['epoch']}"
                save_path.mkdir(exist_ok=True)
                global_logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
                model_to_save.save_pretrained(save_path)
                output_config_file = save_path / 'configs.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state, save_path / 'checkpoint_info.bin')


class EarlyStopping(object):
    '''
    early stopping 功能
    # Arguments
        min_delta: 最小变化
        patience: 多少个epoch未提高，就停止训练
        verbose: 信息大于，默认打印信息
        mode: 计算模式
        monitor: 计算指标
        baseline: 基线
    '''
    def __init__(self,
                 min_delta = 0,
                 patience  = 10,
                 verbose   = 1,
                 mode      = 'min',
                 monitor   = 'loss',
                 baseline  = None):

        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.monitor = monitor

        assert mode in ['min','max']

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self.reset()

    def reset(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stop_training = False
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_step(self,current):
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose >0:
                    global_logger.info(f"{self.patience} epochs with no improvement after which training will be stopped")
                self.stop_training = True

def del_list_idx(li, ids):
    temp = [i for i in li if i not in frozenset(ids)]
    return temp