import torch
import torch.nn as nn
import torch.nn.functional as F
from nerdlm_app.model.inputs_preprocessing import InputPreprocessing

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, device='cuda'):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model).to(device)
        self.key = nn.Linear(d_model, d_model).to(device)
        self.value = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=query.device))

        if mask is not None:
            scores = scores.transpose(0, 1).masked_fill(mask == 0, -1e9).transpose(0, 1)

        attention = self.dropout(F.softmax(scores, dim=-1))

        output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return output

class DeepMultiHeadAttention(nn.Module): 
    def __init__(self, d_model: int, num_heads, num_layers: int, dropout_rate: float = 0.1,):
        super(DeepMultiHeadAttention, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.q_g = nn.GRU(d_model, d_model, num_layers, batch_first=True).to(device)
        self.k_g = nn.GRU(d_model, d_model, num_layers, batch_first=True).to(device)
        self.v_g = nn.GRU(d_model, d_model, num_layers, batch_first=True).to(device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        EPSILON = 1e-6

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        q_g, _ = self.q_g(query)
        k_g, _ = self.k_g(key)
        v_g, _ = self.v_g(value)

        q_g = q_g.view(batch_size, -1, self.num_heads, self.d_k)
        k_g = k_g.view(batch_size, -1, self.num_heads, self.d_k)
        v_g = v_g.view(batch_size, -1, self.num_heads, self.d_k)

        attn1 = F.softmax(q_g*k_g, dim=-1) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=query.device))
        attn2 = F.softmax(q_g*v_g, dim=-1) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=query.device))
        double_attention = self.dropout(attn1*attn2)

        if mask is not None:
            double_attention = double_attention.masked_fill(mask == 0, -1e9)

        linear_combination = query + key + value + EPSILON
        tanh_output = torch.tanh(linear_combination)

        attention_output = (double_attention.view(double_attention.size(0), double_attention.size(1), -1)*tanh_output).contiguous()

        return attention_output

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout_rate: float = 0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class DeepTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers_gru: int, vocab_size: int, d_ff: int = 2048, dropout_rate: float = 0.1, num_attention_heads: int = 8):
        super(DeepTransformer, self).__init__()
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_model = d_model
        self.num_heads = num_heads
        self.input_preprocessing = InputPreprocessing(d_model, vocab_size).to(device)
        self.fc = nn.Linear(d_model, vocab_size).to(device)
        self.deep_multi_head_attention = nn.ModuleList([DeepMultiHeadAttention(d_model, num_heads, num_layers_gru, dropout_rate).to(device) for _ in range(num_attention_heads)])
        self.multi_head_attention = nn.ModuleList([MultiHeadAttention(d_model, num_heads, dropout_rate, device).to(device) for _ in range(num_attention_heads)])
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate).to(device)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6).to(device)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6).to(device)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6).to(device)
        self.dropout = nn.Dropout(dropout_rate).to(device)

    def forward(self, tgt, previous_tgt: list[torch.Tensor] = None):
        if previous_tgt is not None:
            if len(previous_tgt) > 0:
                previous_tgt = torch.cat(previous_tgt, dim=1)
                tgt = torch.cat((previous_tgt, tgt), dim=1)
        x, x_mask = self.input_preprocessing(tgt.to(self.device))
        x1 = self.multi_head_attention(x, x, x, None)
        x2 = self.multi_head_attention(x, x, x, x_mask)
        x = self.norm1(self.dropout(x1*x2 + x) + x)
        x = self.deep_multi_head_attention(x, x, x, None)
        x = self.norm2(self.dropout(x) + x)
        x = self.feed_forward(x)
        x = self.norm3(self.dropout(x) + x)

        return F.log_softmax(self.fc(x), dim=-1), x_mask

if __name__ == "__main__":
    deep_transformer = DeepTransformer(d_model=512, num_heads=8, num_layers_gru=8, vocab_size=10000)
    total_params = sum(p.numel() for p in deep_transformer.parameters())
    trainable_params = sum(p.numel() for p in deep_transformer.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    sample_input = torch.randint(0, 10000, (1, 10))
    output, output_mask = deep_transformer(sample_input)
    print("Output shape:", torch.argmax(output, dim=1))
