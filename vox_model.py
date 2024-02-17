import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torch.nn.modules.utils import _pair, _triple
import copy
import math
from torch.nn.parameter import Parameter

import ml_collections
 

import torch.nn.functional as func
import math
from itertools import combinations
from einops.layers.torch import Rearrange, Reduce

 

def get_3d_trm_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8, 8)})
    config.patches.grid = (8, 8, 8)
    config.hidden_size = 128
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 256
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    return config


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, input_3d=True):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if input_3d:
            self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, bias=True , padding=padding,   dilation=d)
            self.bn = nn.BatchNorm3d(nOut)
        else:
            self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, bias=False, padding=padding,   dilation=d)
            self.bn = nn.BatchNorm1d(nOut)
        self.activation = nn.GELU()
    def forward(self, input):
        output =  self.activation( self.bn( self.conv(input)  )  ) 
        return output


class Embeddings(nn.Module):
    def __init__(self, config, img_size ): 
        super(Embeddings, self).__init__()
        
        self.config = config
         
        patch_size = _triple(config.patches["size"])
 
        self.hybrid_model = nn.Sequential(
            CDilated(8, 8, 3),  
            CDilated(8, 16, 3 ),
            # CDilated(16, 16, 3 ),
            CDilated(16, config.hidden_size, 3 ),
        ) # 空洞卷积

        n_patches = 27
        self.patch_embeddings = nn.Conv3d(in_channels= config.hidden_size,
                                       out_channels= config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # self.position_embeddings = nn.Parameter(self.sinusoidal_embedding(n_patches, config.hidden_size),
        #                                            requires_grad=False)
        nn.init.trunc_normal_(self.position_embeddings, std=0.2)
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print(x.size(), self.position_embeddings.size())
        embeddings =  x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu }

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        hidden_size = config.hidden_size
        self.fc1 = nn.Linear( hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size  / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output,  weights


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size =  config.hidden_size
        self.attention_norm = nn.LayerNorm( config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm( config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        # self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0 else nn.Identity()
        self.dropout1 = nn.Dropout(config.transformer.dropout_rate)
        self.dropout2 = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        h = x

        
        x, weights = self.attn(x)
        x = x +  h 
        x = self.attention_norm(x)

        h = x
        x = self.ffn(x)
        x = x + h
        x = self.ffn_norm(x)
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        # hidden_states = self.encoder_norm(hidden_states)
        return hidden_states, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis=False):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size) # 若干卷积
        self.encoder = Encoder(config, vis) # MLP+att

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights


  
 

class DDS3DSMILESDescriptorNetV3(nn.Module):
    def __init__(self, ):
        super(DDS3DSMILESDescriptorNetV3, self).__init__()
         
         
        config = get_3d_trm_config()
        hidden_size = 384
        self.input_mode = 1 # ppi_for_2,vox_for_1,fusion_for_12
        self.vox_encoder = Transformer(config,  img_size=(8, 24, 24))

        self.vox_proj = nn.Sequential(
            nn.Linear(config.hidden_size,hidden_size), 
            # nn.GELU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
        ) 

 
        self.cell_nn = nn.Sequential(
            nn.Linear(256,hidden_size),
            # nn.GELU(), 
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
        ) 

        self.cell_g_nn = nn.Sequential(
            # nn.BatchNorm1d(18046),
            nn.Linear(18046,hidden_size), 
            # nn.GELU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Dropout(0.5),
        ) 
        
        
        self.final_pred_ln_vox = nn.Sequential(
            # Highway(hidden_size*3),
            nn.BatchNorm1d(hidden_size*3),
            nn.Linear(hidden_size*3, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.5),
            nn.Linear(1024, 1),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # # # nn.Dropout(0.5),
            # nn.Linear(1024, 1),
            # nn.BatchNorm1d(512),
            # nn.GELU(),
            # nn.Linear(512, 1)
        )
        self.final_pred_ln_ppi = nn.Sequential(
            # Highway(hidden_size*3),
            nn.BatchNorm1d(hidden_size*3),
            nn.Linear(hidden_size*3, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.5),
            nn.Linear(1024, 1),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # # # nn.Dropout(0.5),
            # nn.Linear(1024, 1),
            # nn.BatchNorm1d(512),
            # nn.GELU(),
            # nn.Linear(512, 1)
        )
        self.final_pred_ln = nn.Sequential(
            # Highway(hidden_size*3),
            nn.BatchNorm1d(hidden_size*6),
            nn.Linear(hidden_size*6, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.5),
            nn.Linear(1024, 1),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # # # nn.Dropout(0.5),
            # nn.Linear(1024, 1),
            # nn.BatchNorm1d(512),
            # nn.GELU(),
            # nn.Linear(512, 1)
        )
         

        self.proj_dp  = nn.Sequential( 
            nn.Linear(256, hidden_size),
            # nn.GELU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
        )

        # self.proj_feature  = nn.Sequential( 
        #     nn.Linear(hidden_size*3, hidden_size),
        #     # nn.BatchNorm1d(hidden_size),
        # )

        # self.bifusion = nn.Sequential( 
        #     nn.Bilinear(hidden_size, hidden_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        # )
        # self.bifusion = nn.Bilinear(hidden_size*3, hidden_size*3, hidden_size*3)
        # self.bn = nn.BatchNorm1d(hidden_size)

    '''!!!TODO 搞三个函数，一个vox，一个ppi，一个融合 '''

    def forward(self,    d1, d2,  x_vox1, x_vox2,  celllines, cell_g):
        # ppi1，ppi2，vox1，vox2，celllines，cg
        # vox
        drug_feature1, _ = self.vox_encoder(x_vox1.squeeze(1))
        drug_feature2, _ = self.vox_encoder(x_vox2.squeeze(1))
        drug_feature1 = reduce(drug_feature1, 'b p c-> b c', 'max' ) # 池化
        drug_feature2 = reduce(drug_feature2, 'b p c-> b c', 'max' ) # 池化
        drug_feature1 = self.vox_proj( drug_feature1 )
        drug_feature2 = self.vox_proj( drug_feature2 )
        # cellines and cg
        cell_features = self.cell_nn(celllines)
        cell_g = self.cell_g_nn(cell_g)


        # ppi
        drug1_dp = self.proj_dp(d1)
        drug2_dp = self.proj_dp(d2)

        feature_view1 = torch.cat( ( drug_feature1, drug_feature2, cell_g), 1 )
        feature_view2 = torch.cat( ( drug1_dp, drug2_dp, cell_features ), 1 )


        # feature_view1 = self.proj_feature(feature_view1)
        # feature_view2 = self.proj_feature(feature_view2)

        # 1-order fusion
        # fusion_1o = feature_view1 + feature_view2
        # # 2-order fusion
        # fusion_2o =  self.bifusion(feature_view1, feature_view2) 
        # pred = self.final_pred_ln(   feature_view1 )

        # pred = self.final_pred_ln( feature_view1  + feature_view2 )
        # pred2 = self.final_pred_ln( feature_view2 )
        # input_mode:   ppi_for_2, vox_for_1, fusion_for_12
        if self.input_mode == 1:
            pred = self.final_pred_ln_vox(feature_view1)
        elif self.input_mode == 2:
            pred = self.final_pred_ln_ppi( feature_view2 )
        elif self.input_mode == 12:
            pred = self.final_pred_ln(  torch.cat( (feature_view1, feature_view2  ), 1) )
        return pred, 0


 

class DDS3DNetV4(nn.Module):
    def __init__(self, ):
        super(DDS3DNetV4, self).__init__()
         
         
        hidden_size = 512
        
        self.agfp_proj1 = nn.Sequential(
            nn.Linear(900,hidden_size), 
            # nn.ReLU(),
            # nn.GELU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
        ) 
        self.agfp_proj2 = nn.Sequential(
            nn.Linear(900,hidden_size), 
            # nn.ReLU(),
            # nn.GELU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
        ) 
 
        self.cell_nn = nn.Sequential(
            nn.Linear(256,hidden_size),
            # nn.ReLU(),
            # nn.GELU(), 
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
        ) 

        self.cell_g_nn = nn.Sequential(
            nn.Linear(18046,hidden_size), 
            # nn.ReLU(),
            # nn.GELU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Dropout(0.5),
        ) 
        
        
        # self.final_pred_ln = nn.Sequential( 
        #     # Highway(hidden_size*3),
        #     nn.BatchNorm1d(hidden_size*6),
        #     nn.Linear(hidden_size*6, 1024),
        #     nn.GELU(),
        #     nn.BatchNorm1d(1024),
        #     nn.Linear(1024, 1024),
        #     nn.GELU(),
        #     nn.BatchNorm1d(1024),
        #     # # nn.Dropout(0.5),
        #     nn.Linear(1024, 1),
        #     # nn.BatchNorm1d(512),
        #     # nn.GELU(),
        #     # nn.Linear(512, 1)
        # )
         

        self.proj_dp1  = nn.Sequential( 
            nn.Linear(256, hidden_size),
            
        )
        self.proj_dp2  = nn.Sequential( 
            nn.Linear(256, hidden_size),
            
        )
        
        self.final_pred_ln = nn.Sequential( 
            # Highway(hidden_size*3),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(hidden_size*4),
            nn.Linear(hidden_size*4, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            # # nn.Dropout(0.5),
            nn.Linear(1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.GELU(),
            # nn.Linear(1024, 1)
        )
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

          

 
    def forward(self,    d1, d2,  agfp1, agfp2,  celllines, cell_g):
         
         
        cell_features = self.cell_nn(celllines)
        cell_g = self.cell_g_nn(cell_g)
        drug1_dp = self.proj_dp1(d1)
        drug2_dp = self.proj_dp1(d2)
        # drug_feature1 = self.agfp_proj1(agfp1)
        # drug_feature2 = self.agfp_proj1(agfp2)

        feature  = torch.stack( ( drug1_dp, drug2_dp, cell_features, cell_g ), 1 )
        # feature, _ = self.multihead_attn(feature, feature, feature)

        feature = Rearrange('b n d -> b (n d)')(feature)
        pred = self.final_pred_ln(  feature )

         
        # feature_view1 = torch.cat( ( drug_feature1, drug_feature2, cell_g), 1 )
        # feature_view2 = torch.cat( ( drug1_dp, drug2_dp, cell_features ), 1 )

        
        # feature_view1 = self.proj_feature(feature_view1)
        # feature_view2 = self.proj_feature(feature_view2)

        # 1-order fusion
        # fusion_1o = feature_view1 + feature_view2 
        # # 2-order fusion
        # fusion_2o =  self.bifusion(feature_view1, feature_view2) 
        # pred = self.final_pred_ln(   feature_view1 )

        # pred = self.final_pred_ln(  torch.cat( (feature_view1, feature_view2  ), 1) )

        # fusion_2o =  self.bifusion(feature_view1, feature_view2) 
        # pred = self.final_pred_ln(   feature_view2 )


        return pred, 0
