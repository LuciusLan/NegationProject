# Negation Detection with deep learning methods.

------

Code for my master project at Nanyang Technological University Singapore.

Requirements:

python: 3.x

pytorch: 1.5+

gensim: 3.x

keras (only for the padding of input sequence, can be removed)


Performance(for now):

Previous performance was tested under wrong setting. Pending new round of training.

The transformer part (transformer.py) was adapted based on Aditya and Suraj's Transformers-For-Negation-and-Speculation: 

https://github.com/adityak6798/Transformers-For-Negation-and-Speculation


The data pre-processing part (data.py) was modified from the same repo, but added my personal label augmentation options and changed the dataloader structure.

The GRU model's attention part (work in progress) was modified from several sources:

Changhan Wang 's meta embedding:
https://github.com/facebookresearch/DME

Leyang Cui and Yue Zhang's Label attention network:
https://github.com/Nealcly/BiLSTM-LAN

Yihang Wu's BiGRU-CRF-with-Attention-for-NER
https://github.com/ROBINADC/BiGRU-CRF-with-Attention-for-NER