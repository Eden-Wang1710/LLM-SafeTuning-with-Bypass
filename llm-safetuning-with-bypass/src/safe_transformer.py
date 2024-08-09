import torch
import torch.nn.functional as F
from torch import nn


class SafeTransformer(nn.Module):
    """
    several transformer encoder blocks, with auto-regressive attention mask.
    
    Example:
    safe_transformer = SafeTransformer(d_model=4096, nhead=32, nlayer=6, dropout_rate=0.2).to(device)
    """
    def __init__(self, d_model, nhead, nlayer, dropout_rate, activation="gelu"):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_rate, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayer, enable_nested_tensor=False)

    def forward(self, hidden, pad_mask, device):
        hidden = hidden.permute(1, 0, 2)
        # shape: (seq_len, batch_size, d_model)
        seq_len = hidden.size(0)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(device)
        
        # 将 hf的tokenizer的pad_mask转化为torch的pad mask
        pad_mask = ~pad_mask.bool()

        # 传入的是所有position的
        output = self.transformer_encoder(src=hidden, mask=attn_mask, src_key_padding_mask=pad_mask)
        return output # shape: (seq_len, batch_size, d_model)


class SafeFFN(nn.Module):
    """
    safe_mlp = SafeMLP(input_dim=4096, hidden_size=512).to(device)
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        
    def forward(self, hidden):
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SafeBypass(nn.Module):
    """
    safe_by_pass = SafeBypass(safe_transformer, safe_mlp).to(device)
    """
    def __init__(self, safe_transformer, safe_ffn):
        super().__init__()
        self.safe_transformer = safe_transformer
        self.safe_ffn = safe_ffn

    def forward(self, hidden, pad_mask, device):
        """
        input is hidden state extracted from LLM representation
        """
        # shape: (batch_size, seq_len, d_model)
        hidden = self.safe_transformer(hidden, pad_mask, device) 
        # shape: (seq_len, batch_size, d_model)

        ##########################
        # use last valid position
        batch_size = hidden.size(1)
        seq_len = hidden.size(0)
        
        # 将 pad_mask 转换为最后一个非填充 token 的索引
        valid_lengths = pad_mask.sum(dim=1)  # Get the lengths of valid tokens
        last_indices = valid_lengths - 1     # Get the last valid index for each batch
        # print(last_indices)
        
        # 取最后一个有效位置的隐藏状态
        last_token_output = torch.stack([hidden[last_indices[i], i, :] for i in range(batch_size)], dim=0)
        # (batch_size, emd_len)
        ##########################
        
        # # Get the last token's hidden state for classification
        # last_token_output = hidden[-1]  # Get the output of the last token (batch_size, d_model)

        # print(last_token_output)
        # print(last_token_output.shape)

        return self.safe_ffn(last_token_output)

