from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math

from functools import partial
from core.model.tv2d_layer_2 import TV2DFunction
from entmax import sparsemax


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

        if str(gen_func)=='tvmax':
            self.gen_func='tvmax'
            self.sparsemax = partial(sparsemax, k=512)
            self.tvmax = TV2DFunction.apply
        else:
            self.gen_func = gen_func



    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches,-1,self.__C.MULTI_HEAD,self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)

        k = self.linear_k(k).view(n_batches,-1,self.__C.MULTI_HEAD,self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)

        q = self.linear_q(q).view(n_batches,-1,self.__C.MULTI_HEAD,self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches,-1,self.__C.HIDDEN_SIZE)

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        aux=0

        if str(self.gen_func)=='tvmax':
            att_map = self.sparsemax(scores, dim=-1)
        else:
            att_map = self.gen_func(scores, dim=-1)
            
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C,gen_func=torch.softmax):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C,gen_func=gen_func)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C,gen_func=torch.softmax):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C,gen_func=gen_func)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        y = self.norm1(y + self.dropout1(self.mhatt1(y, y, y, y_mask)))
        y = self.norm2(y + self.dropout2(self.mhatt2(x, x, y, x_mask)))
        y = self.norm3(y + self.dropout3(self.ffn(y)))

        return y


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C, gen_func=torch.softmax) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C, gen_func=gen_func) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        
        for enc in self.enc_list:
            x = enc(x, x_mask)
            

        for dec in self.dec_list:
            y = dec(x, y, x_mask, y_mask)

        return x, y
