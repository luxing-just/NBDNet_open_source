import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("stdev", torch.ones(1))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


class CausalDWConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=0,
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        return self.dwconv(F.pad(x, (self.padding, 0)))


class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        self.dw_conv = CausalDWConv1d(d_model, kernel_size)
        self.activation = nn.GELU()
        self.pw_conv = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = x.transpose(1, 2)
        h = self.dw_conv(h)
        h = self.activation(h)
        h = self.pw_conv(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)
        return self.norm(residual + h)


class LocalStabilizer(nn.Module):
    def __init__(self, d_model, n_layers=3, kernel_size=7, local_window=9, dropout=0.1):
        super().__init__()
        self.pad = local_window - 1
        self.mean_filter = nn.Conv1d(
            d_model,
            d_model,
            local_window,
            padding=0,
            groups=d_model,
            bias=False,
        )
        with torch.no_grad():
            self.mean_filter.weight.fill_(1.0 / local_window)
        for param in self.mean_filter.parameters():
            param.requires_grad = False
        self.blocks = nn.ModuleList(
            CausalConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)
        )
        self.mu_proj = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def _local_mean(self, x):
        h = x.transpose(1, 2)
        h = self.mean_filter(F.pad(h, (self.pad, 0)))
        return h.transpose(1, 2)

    def forward(self, x):
        mu = self._local_mean(x)
        h = x - mu
        for block in self.blocks:
            h = block(h)
        return self.norm_out(h + self.mu_proj(mu))


class GlobalTrendEncoder(nn.Module):
    def __init__(self, d_model, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h, _ = self.gru(x)
        h = self.proj(h)
        return self.norm(h + x)


class SDMC(nn.Module):
    def __init__(self, d_model, n_trend=2, n_transition=2, n_fluct=2, dropout=0.1):
        super().__init__()
        self.n_trend = n_trend
        self.n_transition = n_transition
        self.n_fluct = n_fluct
        self.n_slots = n_trend + n_transition + n_fluct
        self.q_trend = nn.Parameter(torch.randn(n_trend, d_model) * 0.02)
        self.q_transition = nn.Parameter(torch.randn(n_transition, d_model) * 0.02)
        self.q_fluct = nn.Parameter(torch.randn(n_fluct, d_model) * 0.02)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _cross_attn(self, q, k, v):
        batch, _, dim = k.shape
        memory_slots = q.shape[0]
        q = q.unsqueeze(0).expand(batch, memory_slots, dim)
        q = self.w_q(q)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
        attn = self.dropout(torch.softmax(score, dim=-1))
        return torch.matmul(attn, v)

    def forward(self, h_global):
        k = self.w_k(h_global)
        v = self.w_v(h_global)
        m_trend = self._cross_attn(self.q_trend, k, v)
        m_transition = self._cross_attn(self.q_transition, k, v)
        m_fluct = self._cross_attn(self.q_fluct, k, v)
        memory = torch.cat([m_trend, m_transition, m_fluct], dim=1)
        memory = self.norm(memory)
        return memory, (m_trend, m_transition, m_fluct)


class LGBI(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.wq_l = nn.Linear(d_model, d_model)
        self.wk_m = nn.Linear(d_model, d_model)
        self.wv_m = nn.Linear(d_model, d_model)
        self.wo_l = nn.Linear(d_model, d_model)
        self.wq_m = nn.Linear(d_model, d_model)
        self.wk_l = nn.Linear(d_model, d_model)
        self.wv_l = nn.Linear(d_model, d_model)
        self.wo_m = nn.Linear(d_model, d_model)
        self.norm_l = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    @staticmethod
    def _attn(q, k, v):
        dim = q.shape[-1]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
        attn = torch.softmax(score, dim=-1)
        return torch.matmul(attn, v), attn

    def forward(self, h_local, memory):
        q_local = self.wq_l(h_local)
        k_memory = self.wk_m(memory)
        v_memory = self.wv_m(memory)
        local_update, _ = self._attn(q_local, k_memory, v_memory)
        local_update = self.wo_l(self.dropout(local_update))
        local_new = self.norm_l(h_local + local_update)
        q_memory = self.wq_m(memory)
        k_local = self.wk_l(h_local)
        v_local = self.wv_l(h_local)
        memory_update, _ = self._attn(q_memory, k_local, v_local)
        memory_update = self.wo_m(self.dropout(memory_update))
        memory_new = self.norm_m(memory + memory_update)
        context, _ = self._attn(local_new, memory_new, memory_new)
        fused = self.fuse(torch.cat([local_new, context], dim=-1))
        return self.norm_out(fused)


class NBDNet(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        d_model=64,
        n_local_layers=3,
        local_kernel=7,
        local_window=9,
        n_global_layers=2,
        n_trend_slots=2,
        n_transition_slots=2,
        n_fluct_slots=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.input_embedding = nn.Linear(enc_in, d_model)
        self.revin = RevIN(d_model, affine=True)
        self.local_enc = LocalStabilizer(
            d_model=d_model,
            n_layers=n_local_layers,
            kernel_size=local_kernel,
            local_window=local_window,
            dropout=dropout,
        )
        self.global_enc = GlobalTrendEncoder(
            d_model=d_model,
            n_layers=n_global_layers,
            dropout=dropout,
        )
        self.sdmc = SDMC(
            d_model=d_model,
            n_trend=n_trend_slots,
            n_transition=n_transition_slots,
            n_fluct=n_fluct_slots,
            dropout=dropout,
        )
        self.lgbi = LGBI(d_model=d_model, dropout=dropout)
        self.prediction_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        batch = x.shape[0]
        h = self.input_embedding(x)
        h = self.revin(h, mode="norm")
        h_local = self.local_enc(h)
        h_global = self.global_enc(h)
        memory, _ = self.sdmc(h_global)
        z = self.lgbi(h_local, memory)
        return self.prediction_head(z.reshape(batch, -1))


__all__ = [
    "RevIN",
    "CausalDWConv1d",
    "CausalConvBlock",
    "LocalStabilizer",
    "GlobalTrendEncoder",
    "SDMC",
    "LGBI",
    "NBDNet",
]
