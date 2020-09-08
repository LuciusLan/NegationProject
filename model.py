import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import numpy as np
from params import param
device = torch.device('cuda')

class Neg(nn.Module):
    """
    Wrapper for the whole model.
    args:
        vocab_size: size of the vocab learned from training for the word embedding
        tokenizer:
    """
    def __init__(self, vocab_size, tokenizer):
        super(Neg, self).__init__()
        self.word_emb_dim = param.word_emb_dim
        self.cue_emb_dim = param.cue_emb_dim
        self.position_emb_dim = param.position_emb_dim
        self.hidden_dim = param.hidden_dim
        self.label_dim = param.label_dim
        self.tokenizer = tokenizer

        self.word_emb = nn.Embedding(vocab_size, self.word_emb_dim)
        self.cue_emb = nn.Embedding(4, self.cue_emb_dim)
        self.init_embedding(self.cue_emb)
        if self.position_emb_dim != 0:
            self.position_emb = nn.Embedding(1, 10)
        self.word_lstm = nn.GRU(self.cue_emb_dim + self.word_emb_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.init_lstm(self.word_lstm)
        self.dropout = nn.Dropout(p=param.dropout)
        self.linear = nn.Linear(self.hidden_dim*2, self.label_dim)
        self.init_linear(self.linear)
        self.label_emb = nn.Embedding(self.label_dim, self.hidden_dim*2)
        self.init_embedding(self.label_emb)
        self.cel = nn.CrossEntropyLoss(ignore_index=0)

        if param.encoder_attention is None:
            pass
        else:
            self.projectors = nn.ModuleDict({
                'word': nn.Linear(self.word_emb_dim, self.word_emb_dim),
                'cue': nn.Linear(self.cue_emb_dim, self.cue_emb_dim)
            })
            self.attention_in_shape = self.word_emb_dim+self.cue_emb_dim
            if param.encoder_attention == 'meta':
                self.attn_0 = nn.LSTM(self.attention_in_shape, 2, bidirectional=True, batch_first=True)
                self.init_lstm(self.attn_0)
                self.attn_1 = nn.Linear(4, 1)
                self.init_linear(self.attn_1)
                self.attn_2 = nn.Linear(1, 1)
                self.init_linear(self.attn_2)
            elif param.encoder_attention == 'softmax':
                self.attn_0 = nn.Linear(self.attention_in_shape, 2)
                self.init_linear(self.attn_0)
                self.attn_1 = nn.Linear(2, 1)
                self.init_linear(self.attn_1)
                self.attn_2 = nn.Linear(1, 1)
                self.init_linear(self.attn_2)
        
        if param.decoder_attention is None:
            pass
        else:
            if param.decoder_attention == 'simple':
                self.mlp1_units = 1024
                self.mlp2_units = 512
                self.fc = nn.Sequential(
                    nn.Linear(self.hidden_dim*2, self.mlp1_units),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(self.mlp1_units, self.mlp2_units),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(self.mlp2_units, self.hidden_dim*2),
                    nn.ReLU(),
                )
                self.init_linear(self.fc[0])
                self.init_linear(self.fc[3])
                self.init_linear(self.fc[6])
            elif param.decoder_attention == 'multihead':
                self.attn = MultiHeadAttention(self.hidden_dim*2, num_heads=param.num_attention_head, dropout_rate=param.dropout)
                self.init_linear(self.attn.linear_q)
                self.init_linear(self.attn.linear_k)
                self.init_linear(self.attn.linear_v)
                """
                self.init_linear(self.attn.Q_proj[0])
                self.init_linear(self.attn.K_proj[0])
                self.init_linear(self.attn.V_proj[0]) """
                self.fc = nn.Linear(self.hidden_dim*4, self.hidden_dim*2, bias=False)
                self.init_linear(self.fc)
                self.ff = FeedForward(self.hidden_dim*2, 1024)

    
    def load_pretrained_word_embedding(self, pre_word_embeddings):
        assert pre_word_embeddings.shape[1] == self.word_emb_dim
        pre_word_embeddings = torch.Tensor(pre_word_embeddings).to(device)
        self.word_emb.weight = nn.Parameter(pre_word_embeddings)
        self.word_emb.weight.requires_grad = False

    def lstm_output(self, input_):
        """
        Input: batch input of shape: 
        batch_size*[input_ids: Tensor, cue_pad: Tensor, scope_pad: Tensor, sent_length: int]
        """
        batch_size = len(input_[0])
        word_emb = self.word_emb(input_[0])
        cue_emb = self.cue_emb(input_[1])
        input_label_seq_tensor = torch.zeros(batch_size, param.label_dim).long().to(device)
        for i in range(batch_size):
            input_label_seq_tensor[i, :param.label_dim] = torch.LongTensor([i for i in range(param.label_dim)])
        label_emb = self.label_emb(input_label_seq_tensor)
        if param.encoder_attention is None:
            embeds = torch.cat((word_emb, cue_emb), 2)
        else:
            projected = []
            projected.append(self.projectors['word'](word_emb))
            projected.append(self.projectors['cue'](cue_emb))
            projected_cat = torch.cat([p for p in projected], 2)
            if param.encoder_attention.startswith('meta'):
                m_attn = self.attn_1(self.attn_0(projected_cat)[0])
            elif param.encoder_attention.startswith('softmax'):
                m_attn = self.attn_1(self.attn_0(projected_cat))
            m_attn = F.relu(m_attn)
            m_attn = self.attn_2(m_attn)
            m_attn = F.softmax(m_attn, dim=2)
            embeds = projected_cat * m_attn

        embeds = self.dropout(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_[3], batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.word_lstm(packed)
        unpad_outputs, unpad_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=embeds.size()[1])
        
        if param.decoder_attention is None:
            fc_output = self.dropout(unpad_outputs)
        else:
            if param.decoder_attention == 'simple':
                attn = self.fc(unpad_outputs) # attention over the sequence length
                alpha = F.softmax(attn, dim=2) # gives the probability values for the time steps in the sequence (weights to each time step)

                fc_output = unpad_outputs * alpha # gives attention weighted lstm embedding
                #fc_output = torch.sum(fc_output, dim=1, keepdim=True) # computes the weighted sum
                fc_output = self.dropout(fc_output)
            elif param.decoder_attention == 'multihead':
                #unpad_outputs = self.dropout(unpad_outputs.transpose(1, 0))
                #attn_mask = self.attention_padding_mask(input_[0], input_[2], 0)
                attn_out, _ = self.attn(label_emb, unpad_outputs, unpad_outputs, None)
                #fc_output = torch.cat([unpad_outputs, attn_out], -1)
                fc_output = self.ff(attn_out)
                fc_output = self.dropout(fc_output)
                """
                #unpad_outputs = self.dropout(unpad_outputs.transpose(1, 0))
                attn_out = self.attn(unpad_outputs, label_emb, label_emb, False)
                fc_output = torch.cat([unpad_outputs, attn_out], -1)
                fc_output = self.fc(fc_output)
                fc_output = self.dropout(fc_output)"""

        lin_out = self.linear(fc_output)
        out = F.log_softmax(lin_out, dim=1)
        return out
    
    def forward(self, input_batch, targets):
        logits = self.lstm_output(input_batch)
        loss = 0.0
        targets = torch.autograd.Variable(targets)
        for i in range(len(targets)):
            loss += F.cross_entropy(logits[i], targets[i])
        return loss
    
    def init_embedding(self, input_):
        if isinstance(self.tokenizer, data.NaiveTokenizer):
            bias = np.sqrt(3.0 / input_.weight.size(1))
            nn.init.uniform_(input_.weight, -bias, bias)
    
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
    
    def init_linear(self, input_):
        #bias = np.sqrt(6.0 / (input_.weight.size(0) + input_.weight.size(1)))
        nn.init.xavier_uniform_(input_.weight)
        if input_.bias is not None:
            input_.bias.data.zero_()

    def attention_padding_mask(self, q, k, padding_index=0):
        """Generate mask tensor for padding value
        Args:
            q (Tensor): (B, T_q)
            k (Tensor): (B, T_k)
            padding_index (int): padding index. Default: 0
        Returns:
            (torch.BoolTensor): Mask with shape (B, T_q, T_k). True element stands for requiring making.
        Notes:
            Assume padding_index is 0:
            k.eq(0) -> BoolTensor (B, T_k)
            k.eq(0).unsqueeze(1)  -> (B, 1, T_k)
            k.eq(0).unsqueeze(1).expand(-1, q.size(-1), -1) -> (B, T_q, T_k)
        """

        mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
        return mask

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention calculation"""

    def __init__(self, dropout_rate=0.0, **kwargs):
        """Initialize ScaledDotProductAttention
        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_mask=None):
        """Forward
        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.
        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)
        """
        attention = torch.bmm(q, k.permute(0, 2, 1))  # (B, T_q, T_k)

        # Scale
        attention *= k.size(-1) ** -0.5

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention

"""
class MultiHeadAttention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(MultiHeadAttention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        if self.gpu:
            self.Q_proj = self.Q_proj.cuda()
            self.K_proj = self.K_proj.cuda()
            self.V_proj = self.V_proj.cuda()


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs
"""

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout_rate=0.0, attention_type='scaled_dot'):
        super().__init__()

        assert model_dim % num_heads == 0, 'model_dim should be devided by num_heads'

        self.h_size = model_dim
        self.num_heads = num_heads
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)

        self.attention = ScaledDotProductAttention(q_dim=self.head_h_size, k_dim=self.head_h_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lnorm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        # Residual
        residual = q

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_q, D / h)
        k = k.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_k, D / h)
        v = v.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)  # (h * B, T_q, T_k)

        context, attention = self.attention(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)

        # Dropout
        output = self.dropout(context)  # (B, T_q, D)

        # Residual connection and Layer Normalization
        output = self.lnorm(residual + output)  # (B, T_q, D)

        return output, attention


class FeedForward(nn.Module):

    def __init__(self, model_dim=512, hidden_dim=2048, dropout_rate=0.0):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.linear1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))

        output = self.dropout(output)

        output = self.norm(output + x)
        return output