import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from params import param

class CueSepPooler(nn.Module):
    def __init__(self, hidden_size, max_num_cue):
        super().__init__()
        self.pack = nn.LSTM(hidden_size + 1, hidden_size//2, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, max_num_cue + 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, hidden_states, cues):
        pack, _ = self.pack(torch.cat([hidden_states, cues], dim=-1))
        pack = self.tanh(pack)
        pack = self.ln(pack)

        fc = self.fc(pack)
        return fc

class CueBert(BertPreTrainedModel):
    def __init__(self, config, max_num_cue=4):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_num_cue = max_num_cue + 1
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cue = nn.Linear(config.hidden_size, config.num_labels)
        self.cue_sep = CueSepPooler(config.hidden_size, max_num_cue)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        cue_labels=None,
        cue_sep_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cue_teacher=None
    ):
        r"""
        cue_labels & cue_sep_labels:
            e.g.:
            cue_labels:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
            cue_sep_labels: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
        To define how to seperate the cues
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        exists_label = cue_labels is not None and cue_sep_labels is not None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        cue_logits = self.cue(sequence_output)
        if cue_teacher is None:
            cue_temp = F.softmax(cue_logits, -1)
            cue_temp = torch.argmax(cue_temp, -1).unsqueeze(2).float()
            cue_sep_logits = self.cue_sep(sequence_output, cue_temp)
        else:
            cue_sep_logits = self.cue_sep(sequence_output, cue_teacher.unsqueeze(2).float())

        if param.predict_cuesep:
            return cue_logits, cue_sep_logits
        else:
            return cue_logits

class ScopeBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.5)
        if param.matrix:
            if param.augment_cue:
                if param.fact:
                    self.scope = nn.ModuleList([BiaffineClassifier(config.hidden_size, 1024, output_dim=1), 
                                                BiaffineClassifier(config.hidden_size, 1024, output_dim=config.num_labels)])
                else:
                    self.scope = BiaffineClassifier(config.hidden_size, 1024, output_dim=config.num_labels)
            else:
                self.lstm = nn.LSTM(config.hidden_size+1, 300, batch_first=True, bidirectional=True)
                self.scope = BiaffineClassifier(300*2, 1024, output_dim=config.num_labels)
        else:
            if param.augment_cue:
                self.scope = nn.Linear(config.hidden_size, config.num_labels)
            else:
                self.lstm = nn.LSTM(config.hidden_size+1, 300, batch_first=True, bidirectional=True)
                self.scope = nn.Linear(300*2, config.num_labels)
        self.sigm = nn.Sigmoid()
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False
        if not param.augment_cue:
            cues = input_ids[1]
            input_ids = input_ids[0]
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if not param.augment_cue:
            # Append the cue embedding to the BERT output
            ### Alternative way: concat at input to form doubled length input
            sequence_output = torch.cat([sequence_output, cues.unsqueeze(-1)], 2)
        sequence_output = self.dropout(sequence_output)
        if param.augment_cue:
            if param.fact:
                arc_logits = self.sigm(self.scope[0](sequence_output))
                label_logits = self.scope[1](sequence_output)
                logits = [arc_logits, label_logits]
            else:
                logits = self.scope(sequence_output)
        else:
            logits = self.lstm(sequence_output)
            logits = self.scope(logits[0])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

class BiaffineClassifier(nn.Module):
    def __init__(self, emb_dim, hid_dim, output_dim=param.label_dim, dropout=0.33):
        super().__init__()
        self.dep = nn.Linear(emb_dim, hid_dim)
        self.head = nn.Linear(emb_dim, hid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.biaffine = nn.Bilinear(hid_dim, hid_dim, output_dim)
        self.biaffine = PairwiseBiaffine(hid_dim, hid_dim, output_dim)
        self.output_dim = output_dim
    
    def forward(self, embedding):
        bs = embedding.size(0)
        dep = self.dropout(self.relu(self.dep(embedding)))
        head = self.dropout(self.relu(self.head(embedding)))
        out = self.biaffine(dep, head).view(bs, -1, self.output_dim)
        #out = out.transpose(1, 2)
        #out = self.decoder(self.dropout(out))
        #out = out.transpose(1, 2)
        return out
        

class PairwiseBilinear(nn.Module):
    """ A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""
    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.zeros(input1_size, input2_size, output_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3).contiguous()
        # (N x L1 x L2 x O) + (O) -> (N x L1 x L2 x O)
        output = output + self.bias

        return output

class PairwiseBiaffine(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1)
        return self.W_bilin(input1, input2)

class SpanScopeBert(BertPreTrainedModel):
    def __init__(self, config, num_layers=2, lstm_dropout=0.35, soft_label=False, num_labels=1, *args, **kwargs):
        super().__init__(config, **kwargs)
        self.soft_label = soft_label
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        self.init_weights()
        if param.use_lstm:
            self.bilstm = nn.LSTM(input_size=config.hidden_size,
                                hidden_size=config.hidden_size // 2,
                                batch_first=True,
                                num_layers=num_layers,
                                dropout=lstm_dropout,
                                bidirectional=True)
            self.init_lstm(self.bilstm)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        self.init_linear(self.start_fc.dense)
        if soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
            self.init_linear(self.end_fc.dense_0)
            self.init_linear(self.end_fc.dense_1)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
            self.init_linear(self.end_fc.dense_0)
            self.init_linear(self.end_fc.dense_1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_point=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if param.use_lstm:
            sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.layer_norm(sequence_output)
        ps1 = self.start_fc(sequence_output)
        if start_point is not None:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                start_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                start_logits.zero_()
                start_logits = start_logits.to(self.device)
                start_logits.scatter_(2, start_point.unsqueeze(2), 1)
            else:
                start_logits = start_point.unsqueeze(2).float()

        else:
            start_logits = F.softmax(ps1, -1)
            if not self.soft_label:
                start_logits = torch.argmax(start_logits, -1).unsqueeze(2).float()
        ps2 = self.end_fc(sequence_output, start_logits)
        return ps1, ps2
    
    def init_linear(self, input_):
        #bias = np.sqrt(6.0 / (input_.weight.size(0) + input_.weight.size(1)))
        nn.init.xavier_uniform_(input_.weight)
        if input_.bias is not None:
            input_.bias.data.zero_()

    def init_lstm(self, input_):
        for ind in range(0, input_.num_layers):
            weight = eval('input_.weight_ih_l' + str(ind))
            #bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.orthogonal_(weight)
            weight = eval('input_.weight_hh_l' + str(ind))
            #bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.orthogonal_(weight)
        if input_.bias:
            for ind in range(0, input_.num_layers):
                weight = eval('input_.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[input_.hidden_size: 2 * input_.hidden_size] = 1
                weight = eval('input_.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[input_.hidden_size: 2 * input_.hidden_size] = 1


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)
        

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x