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
                },
                'joint_cue': {
                    'train': 'split/joint_train_cue_sfu.pt',
                    'dev': 'split/joint_dev_cue_sfu.pt',
                    'test': 'split/joint_test_cue_sfu.pt',
                },
                'joint_scope': {
                    'train': 'split/joint_train_scope_sfu.pt',
                    'dev': 'split/joint_dev_scope_sfu.pt',
                    'test': 'split/joint_test_scope_sfu.pt',
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
                },
                'joint_cue': {
                    'train': 'split/joint_train_cue_bioA.pt',
                    'dev': 'split/joint_dev_cue_bioA.pt',
                    'test': 'split/joint_test_cue_bioA.pt',
                },
                'joint_scope': {
                    'train': 'split/joint_train_scope_bioA.pt',
                    'dev': 'split/joint_dev_scope_bioA.pt',
                    'test': 'split/joint_test_scope_bioA.pt',
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
                },
                'joint_cue': {
                    'train': 'split/joint_train_cue_bioF.pt',
                    'dev': 'split/joint_dev_cue_bioF.pt',
                    'test': 'split/joint_test_cue_bioF.pt',
                },
                'joint_scope': {
                    'train': 'split/joint_train_scope_bioF.pt',
                    'dev': 'split/joint_dev_scope_bioF.pt',
                    'test': 'split/joint_test_scope_bioF.pt',
                }
            }
        }
        self.split_and_save = False
        self.num_runs = 1
        self.dataset_name = 'sherlock' #Available options: 'bioscope_full', 'bioscope_abstracts', 'sherlock', 'sfu'
        self.task = 'scope' # Available options: 'cue', 'scope', 'pipeline', 'joint'
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
        self.ignore_multiword_cue = True
        self.cased = True
        self.max_len = 260
        self.batch_size = 2
        self.num_ep = 60
        self.lr = 5e-5
        self.early_stop_thres = 15
        
        self.gru_or_lstm = 'LSTM'
        self.scope_method = 'augment' # Available options: augment, replace
    
        self.word_emb_dim = 300
        self.lstm_emb_type = 'pre_emb'
        self.cue_emb_dim = 10
        self.position_emb_dim = 0
        self.hidden_dim = 200
        self.dropout = 0.5
        self.biaffine_hidden_dropout = 0.33
        self.label_dim = 3
        self.mark_cue = True
        if self.mark_cue:
            self.label_dim += 1
        self.bioes = False
        if self.bioes:
            self.label_dim += 3
        self.matrix = False
        self.fact = False
        self.m_dir = 'd2'
        self.cue_mode = 'diag'
        if self.m_dir == 'd1':
            self.label_dim += 1
        if self.cue_mode == 'root':
            self.label_dim -= 1 
        self.augment_cue = True
        self.encoder_attention = None # 'meta'
        self.decoder_attention = [] # 'multihead', 'label'
        self.num_attention_head = 5
        self.external_vocab = False
        self.use_crf = False
        if self.use_crf is True:
            self.label_dim += 2

        self.multi = False
        self.ignore_multi_negation = False

        self.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
            'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json"
        }

        self.BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
            'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin"
        }
        self.config = 'config.json'


param = Param()
