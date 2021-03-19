import os

class Param(object):
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = {
            'sfu': 'SFU_Review_Corpus_Negation_Speculation',
            'bioscope_abstracts': 'bioscope/abstracts.xml',
            'bioscope_full': 'bioscope/full_papers.xml',
            'starsem': {
                'train': 'starsem-st-2012-data/cd-sco/corpus/training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt',
                'dev': 'starsem-st-2012-data/cd-sco/corpus/dev/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt',
                'test1': 'starsem-st-2012-data/cd-sco/corpus/test-gold/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt',
                'test2': 'starsem-st-2012-data/cd-sco/corpus/test-gold/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'
            }
        }
        self.split_path = {
            'sfu': {
                'cue': {
                    'train': 'split/train_cue_sfu.pt',
                    'dev': 'split/dev_cue_sfu.pt',
                    'test': 'split/test_cue_sfu.pt',
                },
                'scope': {
                    'train': 'split/train_scope_sfu.pt',
                    'dev': 'split/dev_scope_sfu.pt',
                    'test': 'split/test_scope_sfu.pt',
                    'ns': 'split/ns_sfu.pt',
                }
            },
            'bioscope_abstracts': {
                'cue': {
                    'train': 'split/train_cue_bioA.pt',
                    'dev': 'split/dev_cue_bioA.pt',
                    'test': 'split/test_cue_bioA.pt',
                },
                'scope': {
                    'train': 'split/train_scope_bioA.pt',
                    'dev': 'split/dev_scope_bioA.pt',
                    'test': 'split/test_scope_bioA.pt',
                    'ns': 'split/ns_bioA.pt',
                }
            },
            'bioscope_full': {
                'cue': {
                    'train': 'split/train_cue_bioF.pt',
                    'dev': 'split/dev_cue_bioF.pt',
                    'test': 'split/test_cue_bioF.pt',
                },
                'scope': {
                    'train': 'split/train_scope_bioF.pt',
                    'dev': 'split/dev_scope_bioF.pt',
                    'test': 'split/test_scope_bioF.pt',
                    'ns': 'split/ns_bioF.pt',
                }
            }
        }
        self.seg_path = {
            'sfu': 'seg/sfu_seg.bin',
            'bioscope_a': 'seg/bioscope_a_seg.bin',
            'bioscope_f': 'seg/bioscope_f_seg.bin',
            'sherlock_sep': {
                'train': 'seg/train_seg_s.bin',
                'dev': 'seg/dev_seg_s.bin',
                'test': 'seg/test_seg_s.bin',
            },
            'sherlock_com': {
                'train': 'seg/train_seg_c.bin',
                'dev': 'seg/dev_seg_c.bin',
                'test': 'seg/test_seg_c.bin',
            }
        }
        self.split_and_save = False
        self.num_runs = 1
        self.dataset_name = 'sherlock' #Available options: 'bioscope_full', 'bioscope_abstracts', 'sherlock', 'sfu'
        self.task = 'scope' # Available options: 'cue', 'scope', 'pipeline'
        self.predict_cuesep = True # Specify whether to predict the cue seperation
        self.model_name = f'{self.task}_bert_biaf_{self.dataset_name}'

        self.embedding = 'BERT' # Available options: Word2Vec, FastText, GloVe, BERT\
        if self.embedding != 'BERT':
            if self.embedding == 'FastText':
                self.emb_cache = 'Dev/Vector/generated.bin'
        self.bert_path = 'bert-base-cased'
        self.is_bert = self.embedding == 'BERT'

        self.sherlock_seperate_affix = False
        self.sherlock_combine_nt = False

        self.use_ASL = False
        self.ignore_multiword_cue = False
        self.cased = True
        self.max_len = 190
        self.batch_size = 4
        self.num_ep = 60
        self.lr = 5e-5
        self.early_stop_thres = 15
        
        self.gru_or_lstm = 'LSTM'
        self.scope_method = 'augment' # Available options: augment, replace
    
        self.word_emb_dim = 300
        self.lstm_emb_type = 'pre_emb'
        self.cue_emb_dim = 10
        self.position_emb_dim = 0
        self.segment_emb_dim = 0
        self.hidden_dim = 200
        self.dropout = 0.5
        self.label_dim = 3
        self.mark_cue = False
        if self.mark_cue:
            self.label_dim += 1
        self.bioes = False
        if self.bioes:
            self.label_dim += 3
        self.matrix = True
        if self.matrix:
            self.label_dim += 1
        self.encoder_attention = None # 'meta'
        self.decoder_attention = ['multihead'] # 'multihead', 'label'
        self.num_attention_head = 5
        self.external_vocab = False
        self.use_crf = False
        if self.use_crf is True:
            self.label_dim += 2

        self.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
            'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json"
        }

        self.BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
            'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin"
        }
        self.config = 'config.json'


param = Param()
