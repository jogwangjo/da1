import torch.nn as nn
from sonics.layers import Transformer
from sonics.layers.tokenizer import STTokenizer


class SpecTTTra(nn.Module):
    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        t_clip,
        f_clip,
        num_heads,
        num_layers,
        pre_norm=False,
        pe_learnable=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
    ):
        super(SpecTTTra, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pre_norm = (
            pre_norm  # applied after tokenization before transformer (used in CLIP)
        )
        self.pe_learnable = pe_learnable  # learned positional encoding
        self.pos_drop_rate = pos_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.mlp_ratio = mlp_ratio

        self.st_tokenizer = STTokenizer(
            input_spec_dim,
            input_temp_dim,
            t_clip,
            f_clip,
            embed_dim,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.proj_drop_rate,
            mlp_ratio=self.mlp_ratio,
        )

    def forward(self, x):
        # Squeeze the channel dimension if it exists
        if x.dim() == 4:
            x = x.squeeze(1)

        # Spectro-temporal tokenization
        spectro_temporal_tokens = self.st_tokenizer(x)

        # Positional dropout
        spectro_temporal_tokens = self.pos_drop(spectro_temporal_tokens)

        # Transformer
        output = self.transformer(spectro_temporal_tokens)  # shape: (B, T/t + F/f, dim)

        return output


# Example usage:
input_spec_dim = 384
input_temp_dim = 128
embed_dim = 512
t_clip = 20  # This means t
f_clip = 10  # This means f
num_heads = 8
num_layers = 6
dim_feedforward = 512
num_classes = 10
