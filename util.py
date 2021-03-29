from typing import List, Tuple, T
import logging
import math
import json
import six
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
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

def matrix_decode_toseq(logits: torch.Tensor, pad: List[torch.Tensor], islogit=True, mode='col'):
    """
    Decode the link-matrix to a prediction sequence (single cue case)
    Assumed single cue (multiword cue included), and high quality prediction. 
    That is, only first row containing cue indicator will be extracted

    Params:
        logits(Tensor): [batch_size, seq_length, seq_length, num_classes]
        pad(List[Tensor]): padding matrix
        isLogit(boolean): drop a dimension when input is target (isLogit=False)
        mode: ('row', 'col', 'mean') rule to select target sequence. 
            row: return the first row containing a cue
            col: return the first column containing a cue
            mean: returan the mean of the all candidates.
    Retruns:
        target_seq(Tensor): [batch_size, seq_length, num_classes]
    """
    bs = logits.size(0)
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
        for row_i, row_e in enumerate(mat):
            for col_i, col_e in enumerate(row_e):
                if col_e == 3:
                    rp.append(row_i)
                    cp.append(col_i)
        rp = list(set(rp))
        cp = list(set(cp))
        return rp, cp
                
    pred_scopes = []
    for i, m in enumerate(mat_tmp):
        rp, cp = pos_first_cue_row(m)
        if mode == 'row':
            pred_scopes.append(batch[i][rp])
        elif mode == 'col':
            pred_scopes.append(batch[i][:][cp])
        elif mode == 'mean':
            tmp = []
            rows, cols = all_cue_pos(mat_tmp[i])
            for ri in rows:
                tmp.append(batch[i][ri])
            for ci in cols:
                tmp.append(batch[i][:][ci])
            tmp = torch.stack(tmp)
            pred_scopes.append(tmp.mean(0))
    return pred_scopes

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

def drop_padding(sequences):
    pass

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
