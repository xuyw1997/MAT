import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL_REGISTRY
from .my_mem_transformer import MemTransformerLM, DecoderLayer
from movinets.config import _C
from movinets import MoViNet

class TextEncoder(nn.Module):
    def __init__(self, n_layer, in_dim, out_dim, n_head, d_inner, max_len, mask_token=False):
        super(TextEncoder, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.token_type_embedding = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + max_len, out_dim))
        self.norme = nn.LayerNorm(out_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.token_type_embedding, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)
        if mask_token:
            self.mask_embedding = nn.Embedding(1, out_dim) 
            self.mask_embedding.weight.data.normal_(mean=0.0, std=0.02)
        else:
            self.mask_embedding = None



    def forward(self, txt, txt_mask, word_mask):
        """

        :param txt:
        :param txt_mask: (B, T)
        :return:
        """
        bsz, n = txt.size(0), txt.size(1)

        txt = F.relu(self.fc(txt))
        if self.training and self.mask_embedding is not None:
            _zero_id = torch.zeros((bsz, n), dtype=torch.long, device=txt.device)
            txt = torch.where(word_mask[..., None] == 1, self.mask_embedding(_zero_id), txt)

        cls = self.cls_token.expand(bsz, -1, -1)
        txt = torch.cat([cls, txt], dim=1)
        embeddings = txt + self.token_type_embedding.expand(bsz, n + 1, -1) + self.position_embeddings[:, :n + 1]
        embeddings = self.norme(embeddings)
        embeddings = self.dropout(embeddings)
        
        txt_mask = torch.cat([txt_mask.new_ones(bsz, 1), txt_mask], dim=1)
        return embeddings, txt_mask






class SpatialEncoder(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.trm = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = True )
        self.token_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        self.fc = nn.Sequential(
            nn.Linear(640, d_model),
            nn.ReLU(),
        )
        # self.freeze_backbone()

    
    def freeze_backbone(self):
        for name, m in self.trm.named_modules():
            for param in m.parameters(recurse=False):
                param.requires_grad = False
                               
    def forward(self, images, txt, txt_mask):
        """
        : param images: (B, C, H, W)
        : param txt: (B, L, d)
        : param txt_mask: (B, L)

        """

        x = self.trm.forward_image_text(images, txt, txt_mask)
        return x[:, 0]
    
    def forward_image(self, images):
        """
        : param images: (B, C, T, H, W)
        : param txt: (B, L, d)
        : param txt_mask: (B, L)

        """
        
        bsz, c, n, h, w = images.shape
        x = self.trm(images)
        x = x.transpose(1, 2).contiguous()
        x = self.fc(x)
        x = x + self.token_type_embedding.expand(bsz, n, -1)
        return x

    def _pos_embed(self, x):
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.spa_pos_embed
        return x

    def forward_image_txt(self, images, txt, txt_mask):
        """
        : param images: (B, C, H, W)
        : param txt: (B, L, d)
        : param txt_mask: (B, L)

        """
        # MoViNet backbone
        bsz, n = images.size(0), images.size(2)
        x = self.trm(images)
        c, h, w = x.shape[1], x.shape[3], x.shape[4]
        x = x.transpose(1, 2) # b, t, c, h, w
        x = x.reshape(bsz * n, c, h, w)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.fc(x)
        x = self._pos_embed(x)
        txt = torch.repeat_interleave(txt, n, dim=0)
        txt_mask = torch.repeat_interleave(txt_mask, n, dim=0)
        co_embeds = torch.cat([x, txt], dim=1)
        co_mask = torch.cat([x.new_ones(bsz * n, x.shape[1]), txt_mask], dim=1)
        for i, blk in enumerate(self.blocks):
            co_embeds = blk(co_embeds, co_mask)
        co_embeds = self.norm(co_embeds)
        x = co_embeds[:, 0]
        x = x.view(bsz, n, -1)
        x = x + self.token_type_embedding.expand(bsz, n, -1)

        return x  


def init_weight(weight):
    nn.init.normal_(weight, 0.0, 0.02)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def my_weights_init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.GroupNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv1d):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)
        torch.nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    
   
class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return 1.5 ** (input * self.scale)


class AnchorFreeDecoder(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        loc_tower = [
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding='same'),
                nn.GroupNorm(16, d_model),
                nn.ReLU(inplace=True)
            )
        ]
        self.loc_tower = nn.Sequential(*loc_tower)

        conf_tower = [
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding='same'),
                nn.GroupNorm(16, d_model),
                nn.ReLU(inplace=True)
            )
        ]
        self.conf_tower = nn.Sequential(*conf_tower)

        self.loc_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=d_model, out_channels=2, kernel_size=3, stride=1, padding='same'),
            ScaleExp()
        )

        self.conf_head = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=3, stride=1, padding='same')

        self.center_head = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=3, stride=1, padding='same')

        center_tower = [
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding='same'),
                nn.GroupNorm(16, d_model),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding='same'),
                nn.GroupNorm(16, d_model),
                nn.ReLU(inplace=True)
            )
           
        ]
        self.center_tower =  nn.Sequential(*center_tower)

    
    def forward(self, x):
        x = x.transpose(1, 2)
        loc_feat = self.loc_tower(x)
        conf_feat = self.conf_tower(x)
        loc = self.loc_head(loc_feat)
        conf = self.conf_head(conf_feat)
        loc = loc.transpose(1, 2)
        conf = conf.squeeze(1)
        center_feat = self.center_tower(loc_feat)
        centerness = self.center_head(center_feat).squeeze(1)

        return loc, conf, centerness



@MODEL_REGISTRY.register()
class ClipByClip2(nn.Module):

    def __init__(self, cfg):
        super(ClipByClip2, self).__init__()

        self.cfg = cfg
       
        self.d_model = cfg.MODEL.MODEL_DIM
        self.mem_len = cfg.MODEL.MEM_LEN
        self.n_layer = cfg.MODEL.NUM_LAYER
        self.n_head = cfg.MODEL.NUM_HEAD
        self.window_size = cfg.DATA.WINDOW_SIZE
        self.c_mem_len = cfg.MODEL.C_MEM_LEN
        self.compress_rate = cfg.MODEL.COMPRESS_RATE
        self.compress_type = cfg.MODEL.COMPRESS_TYPE
        self.num_buckets = cfg.MODEL.NUM_BUCKETS
        self.dropout = cfg.MODEL.DROPOUT
        self.dropatt = cfg.MODEL.DROPATT
        self.bidirectional = cfg.MODEL.BIDIRECTIONAL

        self.spa_enc = SpatialEncoder(self.d_model)
        self.mem_trm = MemTransformerLM(
            n_layer=self.n_layer, n_head=self.n_head,d_model=self.d_model,
            d_head=self.d_model // self.n_head, d_inner=self.d_model * 4,
            dropout=self.dropout, dropatt=self.dropatt, pre_lnorm=False, tgt_len=self.window_size,
            mem_len=self.mem_len, same_length=False, attn_type=0, clamp_len=-1,
            compress_rate=self.compress_rate, compress_type=self.compress_type,
            c_mem_len=self.c_mem_len, num_buckets=self.num_buckets, bidirectional=self.bidirectional
        )
        
        self.clip_dec = AnchorFreeDecoder(self.d_model)
        self.txt_enc = TextEncoder(
            n_layer=2, in_dim=cfg.DATA.WORD_EMBEDDING_DIM, out_dim=self.d_model, 
            n_head=self.n_head, d_inner=4 * self.d_model, max_len=100,
            mask_token=cfg.DATA.MLM_P > 0
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, 1)
        )
        
        self.mem_trm.apply(weights_init)

       

    def forward(self, clip, txt, txt_mask, word_mask, mems, c_mems):
        """

        :param clip: (B, C, T, H, W)
        :param txt: (B, L, d)
        :param txt_mask: (B, L)
        :return:
        """
        


        txt, txt_mask = self.txt_enc(txt, txt_mask, word_mask)
        clip = self.spa_enc.forward_image(clip)

        hidden, txt, new_mems, new_c_mems = self.mem_trm.forward_single_stream(clip, txt, txt_mask, mems=mems, c_mems=c_mems)

        loc, conf, centerness = self.clip_dec(hidden)
        clip_level_pred = self.cls_head(txt[:, 0])


        return loc, conf, centerness, clip_level_pred, new_mems, new_c_mems