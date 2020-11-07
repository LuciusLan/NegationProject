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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scope = nn.Linear(config.hidden_size, config.num_labels)
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        logits = self.scope(sequence_output)

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