import os

class Param(object):
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.cased = True
        self.max_len = 190
        self.batch_size = 8
        self.num_ep = 200
        self.lr = 5e-5
        self.early_stop_thres = 15
        self.cross_test = False
        self.excess_train = True
        self.gru_or_lstm = 'LSTM'
        self.dataset_name = 'starsem' #Available options: 'bioscope_full', 'bioscope_abstracts', 'starsem', 'sfu'
        self.data_path = {
            'sfu': 'SFU_Review_Corpus_Negation_Speculation',
            'bioscope_abstracts': 'bioscope\\abstracts.xml',
            'bioscope_full': 'bioscope\\full_papers.xml',
            'starsem': {
                'train': 'starsem-st-2012-data\\cd-sco\\corpus\\training\\SEM-2012-SharedTask-CD-SCO-training-09032012.txt',
                'dev': 'starsem-st-2012-data\\cd-sco\\corpus\\dev\\SEM-2012-SharedTask-CD-SCO-dev-09032012.txt',
                'test1': 'starsem-st-2012-data\\cd-sco\\corpus\\test-gold\\SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt',
                'test2': 'starsem-st-2012-data\\cd-sco\\corpus\\test-gold\\SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'
            }
        }
        self.task = 'scope'
        self.scope_method = 'augment' # Available options: augment, replace
        self.embedding = 'FastText' # Available options: Word2Vec, FastText, GloVe, BERT\
        if self.embedding != 'BERT':
            if self.embedding == 'FastText':
                self.emb_cache = 'cc.en.300.bin'
        self.word_emb_dim = 300
        self.lstm_emb_type = 'pre_emb'
        self.cue_emb_dim = 10
        self.position_emb_dim = 0
        self.hidden_dim = 200
        self.dropout = 0.5
        self.label_dim = 3
        self.encoder_attention = None # 'softmax' or 'meta'
        self.decoder_attention = None # 'simple', 'multihead', 'label'
        self.num_attention_head = 8
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
