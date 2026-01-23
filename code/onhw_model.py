import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import string
def _rand_uniform(a, b):
    return (b - a) * torch.rand(1).item() + a

def aug_jitter(x, sigma=0.02):

    return x + torch.randn_like(x) * sigma

def aug_scale(x, scale_std=0.1):

    s = 1.0 + torch.randn(1).item() * scale_std
    return x * s

def aug_shift(x, shift_std=0.05):

    b = torch.randn(1).item() * shift_std
    return x + b

def aug_time_mask(x, max_ratio=0.1):

    T, Fea = x.shape
    w = max(1, int(T * _rand_uniform(0.02, max_ratio)))
    start = torch.randint(0, max(T - w, 1), (1,)).item()
    x2 = x.clone()
    x2[start:start + w, :] = 0.0
    return x2

def aug_circ_shift(x, max_ratio=0.1):

    T, Fea = x.shape
    k = int(_rand_uniform(-T*max_ratio, T*max_ratio))
    return torch.roll(x, shifts=k, dims=0)

def aug_time_warp_resample(x, rate_min=0.8, rate_max=1.25):

    T, Fea = x.shape
    r = _rand_uniform(rate_min, rate_max)
    new_T = max(4, int(T * r))
    x_chw = x.t().unsqueeze(0)          # (1, Fea, T)
    x_resamp = F.interpolate(x_chw, size=new_T, mode="linear", align_corners=False)  # (1, Fea, new_T)
    x_back = F.interpolate(x_resamp, size=T, mode="linear", align_corners=False)     # (1, Fea, T)
    return x_back.squeeze(0).t()        # (T, Fea)

def compose_augs(x,
                 p_jitter=0.7, p_scale=0.5, p_shift=0.3,
                 p_mask=0.5, p_circ=0.3, p_warp=0.7,
                 jitter_sigma=0.02, scale_std=0.1, shift_std=0.05,
                 mask_max_ratio=0.12, circ_max_ratio=0.1,
                 warp_min=0.85, warp_max=1.2):
    if torch.rand(1).item() < p_jitter:
        x = aug_jitter(x, jitter_sigma)
    if torch.rand(1).item() < p_scale:
        x = aug_scale(x, scale_std)
    if torch.rand(1).item() < p_shift:
        x = aug_shift(x, shift_std)
    if torch.rand(1).item() < p_mask:
        x = aug_time_mask(x, mask_max_ratio)
    if torch.rand(1).item() < p_circ:
        x = aug_circ_shift(x, circ_max_ratio)
    if torch.rand(1).item() < p_warp:
        x = aug_time_warp_resample(x, warp_min, warp_max)
    return x

def char_to_bymerge_id(ch: str, char2idx: dict):

    if ch in char2idx:
        return char2idx[ch]
    return -1

def char_to_letter_id(ch: str):
    ch_up = ch.upper()
    if ch_up in string.ascii_uppercase:  # 'A'..'Z'
        return ord(ch_up) - ord('A')
    return -1
from torch.utils.data import Dataset
import torch
import numpy as np
import string

idx2char = np.load("emnist_idx2char.npy", allow_pickle=True)
idx2char = [c.decode() if isinstance(c, bytes) else c for c in idx2char]

# 构建 char -> id 的反向表
char2idx = {ch: i for i, ch in enumerate(idx2char)}
class OnHWDataset(Dataset):
    def __init__(self, root="onhw-chars_2021-06-30/onhw2_upper_indep_2", split="train"):
        """
        root 下有:
          - X_train.npy / y_train.npy
          - X_test.npy  / y_test.npy
        y 里是单字符: 'A','b', ...
        """
        if split == "train":
            X_path = os.path.join(root, "X_train.npy")
            y_path = os.path.join(root, "y_train.npy")
        else:
            X_path = os.path.join(root, "X_test.npy")
            y_path = os.path.join(root, "y_test.npy")

        X = np.load(X_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)

        self.samples = []
        for seq, ch in zip(X, y):
            ch = str(ch)           # 确保是字符串
            cid = char_to_bymerge_id(ch,char2idx)
            if cid < 0:
                # 非 A-Z 的直接丢掉（数字/符号）
                continue
            ts = torch.tensor(seq, dtype=torch.float32)  # (T, 13)
            self.samples.append({
                "time_series": ts,
                "letter_id": cid,         # 0..25
                "symbol_char": ch
            })

        print(f"[OnHW-{split}] kept {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import torch
import torch.nn.functional as F

class MultiModalRealDataset(Dataset):
    def __init__(self, 
                 photonic_digit_path="../data/dataset_photonic.npz", 
                 photonic_letter_dir="../data", 
                 use_mnist=True, use_emnist=True,
                 transform=None, mode="train", split_ratio=0.8,
                 subset="all"):   # 🔑 新增参数
        """
        subset: {"all", "digit", "letter", "photo"}
        """
        self.samples = []
        self.mode = mode
        self.split_ratio = split_ratio
        self.subset = subset.lower()
        self.transform = transform or transforms.Normalize((0.1307,), (0.3081,))

        # 记录来源
        self.source_stats = {
            "photonic_digit": [],
            "photonic_letter": [],
            "mnist": [],
            "emnist": []
        }

        # -------- 1. photonic 数字 --------
        if os.path.exists(photonic_digit_path) and self.subset in ["all", "digit"]:
            data = np.load("dataset_photonic.npz", allow_pickle=True)
            if mode == "train":
                X, y = data["X_train"], data["y_train"]
            elif mode == "val":
                X, y = data["X_val"], data["y_val"]
            elif mode == "test":
                X, y = data["X_test"], data["y_test"]

            for s, l in zip(X, y):
                self.samples.append({
                    "time_series": torch.tensor(s, dtype=torch.float32),
                    "reference_image": None,
                    "symbol_idx": int(l),
                    "symbol_type": "digit"
                })
        n_aug_per_sample = 20  
        if os.path.isdir(photonic_letter_dir) and self.subset in ["all", "letter"]:
            X = np.load(os.path.join(photonic_letter_dir, f"X_{mode}.npy"), allow_pickle=True)
            y = np.load(os.path.join(photonic_letter_dir, f"y_{mode}.npy"), allow_pickle=True)
            for s, l in zip(X, y):

                symbol_idx = int(l) + 10 
                ts = torch.tensor(s, dtype=torch.float32) 
                self.samples.append({
                    "time_series": ts,
                    "reference_image": None,
                    "symbol_idx": symbol_idx,
                    "symbol_type": "letter",
                    "is_aug": False
                })
                self.source_stats["photonic_letter"].append(symbol_idx)

                for _ in range(n_aug_per_sample):
                    ts_aug = compose_augs(ts)
                    self.samples.append({
                        "time_series": ts_aug,
                        "reference_image": None,
                        "symbol_idx": symbol_idx,
                        "symbol_type": "letter",
                        "is_aug": True
                    })
                    self.source_stats["photonic_letter"].append(symbol_idx)

        # -------- 3. MNIST --------
        if use_mnist and os.path.exists("mnist.npz") and self.subset in ["all", "digit", "photo"]:
            mnist = np.load("mnist.npz")
            X, y = mnist["X"], mnist["y"]
            n_train = int(len(X) * self.split_ratio)
            if mode == "train":
                X, y = X[:n_train], y[:n_train]
            else:
                X, y = X[n_train:], y[n_train:]
            for img, label in zip(X, y):
                img_tensor = torch.tensor(img, dtype=torch.float32)
                if img_tensor.ndim == 2:
                    img_tensor = img_tensor.unsqueeze(0)
                elif img_tensor.ndim == 3 and img_tensor.shape[-1] == 1:
                    img_tensor = img_tensor.permute(2,0,1)
                img_tensor = img_tensor / 255.0
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                label_idx = int(np.argmax(label))
                self.samples.append({
                    "time_series": None,
                    "reference_image": img_tensor,
                    "symbol_idx": label_idx,
                    "symbol_type": "digit"
                })
                self.source_stats["mnist"].append(label_idx)

        # -------- 4. EMNIST --------
        if use_emnist and os.path.exists("emnist_bymerge.npz") and self.subset in ["all", "letter", "photo"]:
            emnist = np.load("emnist_bymerge.npz")
            X, y = emnist["X"], emnist["y"]
            n_train = int(len(X) * self.split_ratio)
            if mode == "train":
                X, y = X[:n_train], y[:n_train]
            else:
                X, y = X[n_train:], y[n_train:]
            for img, label in zip(X, y):
                img_tensor = torch.tensor(img, dtype=torch.float32)
                if img_tensor.ndim == 2:
                    img_tensor = img_tensor.unsqueeze(0)
                elif img_tensor.ndim == 3 and img_tensor.shape[-1] == 1:
                    img_tensor = img_tensor.permute(2,0,1)
                img_tensor = img_tensor / 255.0
                if self.transform:
                    img_tensor = self.transform(img_tensor)

                label_idx = int(np.argmax(label))
                symbol_idx = label_idx
                self.samples.append({
                    "time_series": None,
                    "reference_image": img_tensor,
                    "symbol_idx": symbol_idx,
                    "symbol_type": "letter"
                })
                self.source_stats["emnist"].append(symbol_idx)

        # -------- 打印统计 --------
        self._print_stats()

    def _print_stats(self):
        print(f"\n[{self.mode}] 数据源标签分布:")
        for src, labels in self.source_stats.items():
            if labels:
                uniq = sorted(set(labels))
                print(f"  - {src}: min={min(labels)}, max={max(labels)}, "
                      f"num_unique={len(uniq)}, total={len(labels)}, "
                      f"unique={uniq[:20]}{'...' if len(uniq)>20 else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ImageOnhwPairDataset(Dataset):
    def __init__(self, emnist_ds, onhw_ds):
        """
        emnist_ds: EMNISTLetterImageDataset
        onhw_ds:   OnHWDataset
        """
        self.emnist_ds = emnist_ds
        self.onhw_ds = onhw_ds

        # letter_id -> indices 列表
        self.letter2img_idx = {}
        for i in range(len(emnist_ds)):
            cid = emnist_ds[i]["letter_id"]
            self.letter2img_idx.setdefault(cid, []).append(i)

        self.letter2onhw_idx = {}
        for j in range(len(onhw_ds)):
            cid = onhw_ds[j]["letter_id"]
            self.letter2onhw_idx.setdefault(cid, []).append(j)

        # 只保留两边都存在的字母
        self.common_ids = sorted(set(self.letter2img_idx) & set(self.letter2onhw_idx))
        print("common letter ids:", self.common_ids)

        # 构建 pair 列表
        self.pairs = []
        for cid in self.common_ids:
            img_idxs = self.letter2img_idx[cid]
            onhw_idxs = self.letter2onhw_idx[cid]
            # 简单做法：每个 EMNIST 样本配一个随机 OnHW 同类样本
            for img_i in img_idxs:
                onhw_j = np.random.choice(onhw_idxs)
                self.pairs.append((img_i, onhw_j, cid))

        print(f"[Pair] total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_i, onhw_j, cid = self.pairs[idx]
        img_sample  = self.emnist_ds[img_i]
        onhw_sample = self.onhw_ds[onhw_j]
        return {
            "image": img_sample["image"],              # (1,H,W)
            "onhw_seq": onhw_sample["time_series"],    # (T,13)
            "letter_id": cid                           # 0..25
        }

class TimeSeriesVAE(nn.Module):
    def __init__(self,
                 input_dim: int,   # F
                 hidden_dim: int = 64,
                 latent_dim: int = 16,
                 num_layers: int = 1,
                 bidirectional: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # ----- Encoder -----
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        enc_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc_mu     = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # ----- Decoder -----
        # 输入是 [x_t, z]，所以 input_dim + latent_dim
        self.decoder_rnn = nn.GRU(
            input_size=input_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # 解码一般单向即可
        )
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, lengths):
        """
        x:       (B, T_max, F)
        lengths: (B,)
        返回: mu, logvar, z
        """
        # pack 去掉 padding 的影响
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, h_n = self.encoder_rnn(packed)
        # h_n: (num_layers * num_directions, B, H)
        h_n = h_n.transpose(0, 1).contiguous()  # (B, L*D, H)
        h_n = h_n.view(h_n.size(0), -1)         # (B, L*D*H)

        mu     = self.fc_mu(h_n)     # (B, latent_dim)
        logvar = self.fc_logvar(h_n) # (B, latent_dim)
        z      = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, x, z, lengths=None):
        """
        x: (B, T_max, F)   原始序列，作为 teacher forcing 输入
        z: (B, latent_dim)
        返回: x_recon: (B, T_max, F)
        """
        B, T_max, F = x.shape
        # 把 z broadcast 到每个 time step
        z_expanded = z.unsqueeze(1).repeat(1, T_max, 1)  # (B, T_max, latent_dim)
        dec_in = torch.cat([x, z_expanded], dim=-1)      # (B, T_max, F + latent_dim)

        dec_out, _ = self.decoder_rnn(dec_in)            # (B, T_max, H)
        x_recon = self.dec_out(dec_out)                  # (B, T_max, F)
        return x_recon

    def forward(self, x, lengths):
        """
        训练时调用:
          x: (B, T_max, F)
          lengths: (B,)
        返回:
          x_recon, mu, logvar
        """
        mu, logvar, z = self.encode(x, lengths)
        x_recon = self.decode(x, z, lengths)
        return x_recon, mu, logvar
import torch
import torch.nn as nn

class MultiDomainVAE(nn.Module):
    def __init__(self,
                 F_onhw: int,
                 F_photonic: int,
                 F_common: int,
                 latent_dim: int,
                 vae_kwargs: dict):
        """
        F_onhw:     onhw 域的特征维度
        F_photonic: photonic 域的特征维度
        F_common:   共用的 VAE 输入特征维度
        latent_dim: 潜变量维度
        vae_kwargs: 传给 TimeSeriesVAE 的其它参数（hidden_size 等）
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.F_common   = F_common

        # 1) 域 → common 的投影
        self.proj_onhw     = nn.Linear(F_onhw,     F_common)
        self.proj_photonic = nn.Linear(F_photonic, F_common)

        # 2) common → 域 的“反投影”
        self.deproj_onhw     = nn.Linear(F_common, F_onhw)
        self.deproj_photonic = nn.Linear(F_common, F_photonic)

        # 3) 共用的 VAE 主体
        self.vae = TimeSeriesVAE(
            input_dim=F_common,
            latent_dim=latent_dim,
            **vae_kwargs
        )

    # ---- 内部工具函数 ----
    def _get_proj(self, domain: str):
        if domain == "onhw":
            return self.proj_onhw
        elif domain == "photonic":
            return self.proj_photonic
        else:
            raise ValueError(f"Unknown domain {domain}")

    def _get_deproj(self, domain: str):
        if domain == "onhw":
            return self.deproj_onhw
        elif domain == "photonic":
            return self.deproj_photonic
        else:
            raise ValueError(f"Unknown domain {domain}")

    # ---- 编码：任意域 → latent ----
    def encode(self, x, lengths, domain: str):
        """
        x:       (B, T, F_domain)
        lengths: (B,)
        domain:  "onhw" 或 "photonic"
        返回: mu, logvar, z  （都是 (B, latent_dim)）
        """
        proj = self._get_proj(domain)
        x_proj = proj(x)  # (B, T, F_common)
        mu, logvar, z = self.vae.encode(x_proj, lengths)
        return mu, logvar, z

    # ---- 解码：latent → 指定域 序列 ----
    def decode(self, x_context, z, lengths, domain: str):
        """
        x_context: (B, T, F_domain)
            - 作为 decoder 的 teacher forcing 输入
            - 一般是“同域”的原始序列，或你构造的某个参考序列
        z:        (B, latent_dim)
        lengths:  (B,) 或 None
        domain:   目标域（"onhw" 或 "photonic"）

        返回:
            x_recon: (B, T, F_domain)
        """
        proj   = self._get_proj(domain)
        deproj = self._get_deproj(domain)

        # 把 teacher forcing 输入投影到 common 维度
        x_ctx_proj = proj(x_context)  # (B, T, F_common)

        # 在 common 空间里做解码
        x_recon_proj = self.vae.decode(x_ctx_proj, z, lengths)  # (B, T, F_common)

        # 再反投影回该域的原始维度
        x_recon = deproj(x_recon_proj)  # (B, T, F_domain)
        return x_recon

    # ---- 单域重构 forward（训练用） ----
    def forward(self, x, lengths, domain: str):
        """
        标准 VAE 训练路径：给定某域的 x，做重构。

        x:       (B, T, F_domain)
        lengths: (B,)
        domain:  "onhw" / "photonic"

        返回:
            x_recon_domain: (B, T, F_domain)
            mu, logvar
        """
        proj   = self._get_proj(domain)
        deproj = self._get_deproj(domain)

        x_proj = proj(x)  # (B, T, F_common)

        # TimeSeriesVAE 自己内部会 encode + decode
        x_recon_proj, mu, logvar = self.vae(x_proj, lengths)  # (B, T, F_common), (B, latent_dim), ...

        x_recon = deproj(x_recon_proj)  # (B, T, F_domain)
        return x_recon, mu, logvar

    # ---- 简单的“围绕样本”的采样增强接口 ----
    @torch.no_grad()
    def sample_around(self,
                      x,
                      lengths,
                      domain: str,
                      k: int = 5,
                      scale: float = 0.5):
        """
        对给定域的真实样本 x, 在其对应的 latent μ 周围做扰动，生成增强样本。

        x:       (B, T, F_domain)
        lengths: (B,)
        domain:  "onhw" / "photonic"
        k:       每个样本生成多少个变体
        scale:   对标准差的缩放，避免太离谱（0.3~0.7 一般比较稳）

        返回:
            x_aug: (B*k, T, F_domain)
        """
        mu, logvar, z = self.encode(x, lengths, domain)  # (B, latent_dim)
        B, latent_dim = mu.shape

        std = torch.exp(0.5 * logvar)  # (B, latent_dim)

        # eps: (k, B, latent_dim)
        eps = torch.randn(k, B, latent_dim, device=x.device, dtype=x.dtype)
        z_samples = mu.unsqueeze(0) + scale * eps * std.unsqueeze(0)  # (k, B, latent_dim)

        # teacher forcing 用的还是原始 x（同一域）
        x_all = []
        for i in range(k):
            z_i = z_samples[i]  # (B, latent_dim)
            x_recon_i = self.decode(x, z_i, lengths, domain=domain)  # (B, T, F_domain)
            x_all.append(x_recon_i)

        x_aug = torch.cat(x_all, dim=0)  # (k*B, T, F_domain)
        return x_aug
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ConditionalTimeSeriesVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,       # 新增：类别总数
                 label_emb_dim: int = 8, # 新增：类别嵌入维度
                 hidden_dim: int = 64,
                 latent_dim: int = 16,
                 num_layers: int = 1,
                 bidirectional: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.label_emb_dim = label_emb_dim
        
        # 1. Label Embedding
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)

        # 2. Encoder: 输入不仅是 x，还有 label embedding
        # 输入维度: F + label_emb_dim
        self.encoder_rnn = nn.GRU(
            input_size=input_dim + label_emb_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        enc_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc_mu     = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # 3. Decoder: 输入是 [x_t, z, label_emb]
        # input_dim + latent_dim + label_emb_dim
        self.decoder_rnn = nn.GRU(
            input_size=input_dim + latent_dim + label_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False, 
        )
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, lengths, labels):
        # labels: (B,)
        c = self.label_embedding(labels)  # (B, label_emb_dim)
        if x.dim() == 4:
            x = x.squeeze(1)
        # 将 c 扩展到时间维度: (B, T, label_emb_dim)
        B, T, _ = x.shape
        c_expanded = c.unsqueeze(1).repeat(1, T, 1)
        
        # 拼接: (B, T, F + label_emb_dim)
        enc_in = torch.cat([x, c_expanded], dim=-1)
        
        packed = pack_padded_sequence(enc_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder_rnn(packed)
        
        h_n = h_n.transpose(0, 1).contiguous().view(B, -1)
        mu     = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z      = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, x_ctx, z, labels):
        """
        x_ctx: Teacher Forcing 的上下文输入 (B, T, F)
        z:     Latent vector (B, latent_dim)
        labels:(B,)
        """
        if x_ctx.dim() == 4:
            x_ctx = x_ctx.squeeze(1)
        B, T, _ = x_ctx.shape
        
        # 1. Expand z -> (B, T, latent_dim)
        z_expanded = z.unsqueeze(1).repeat(1, T, 1)
        
        # 2. Expand label -> (B, T, label_emb_dim)
        c = self.label_embedding(labels)
        c_expanded = c.unsqueeze(1).repeat(1, T, 1)
        
        # 3. Concat all: [x, z, c]
        dec_in = torch.cat([x_ctx, z_expanded, c_expanded], dim=-1)
        
        dec_out, _ = self.decoder_rnn(dec_in)
        x_recon = self.dec_out(dec_out)
        return x_recon

    def forward(self, x, lengths, labels):
        mu, logvar, z = self.encode(x, lengths, labels)
        x_recon = self.decode(x, z, labels)
        return x_recon, mu, logvar

class ConditionalMultiDomainVAE(nn.Module):
    def __init__(self, F_onhw, F_photonic, F_common, num_classes, latent_dim, vae_kwargs):
        super().__init__()
        self.proj_onhw = nn.Linear(F_onhw, F_common)
        self.proj_photonic = nn.Linear(F_photonic, F_common)
        
        self.deproj_onhw = nn.Linear(F_common, F_onhw)
        self.deproj_photonic = nn.Linear(F_common, F_photonic)

        # 核心 VAE 换成 Conditional 版本
        self.vae = ConditionalTimeSeriesVAE(
            input_dim=F_common,
            num_classes=num_classes,
            latent_dim=latent_dim,
            **vae_kwargs
        )

    def forward(self, x, lengths, labels, domain="photonic"):
        # 1. 投影
        proj = self.proj_onhw if domain == "onhw" else self.proj_photonic
        deproj = self.deproj_onhw if domain == "onhw" else self.deproj_photonic
        
        x_proj = proj(x)
        
        # 2. CVAE Forward (带 label)
        x_recon_proj, mu, logvar = self.vae(x_proj, lengths, labels)
        
        # 3. 反投影
        x_recon = deproj(x_recon_proj)
        return x_recon, mu, logvar
    # ============================================================
    # [最终修改版] 适配 x_ctx 接口的 decode
    # ============================================================
    def decode(self, z, labels, domain="photonic", seq_len=100):
        """
        Args:
            z: (Batch, latent_dim) 风格向量
            labels: (Batch, ) 内容标签
            domain: "photonic" 或 "onhw"
            seq_len: 你希望生成的序列长度 (必须指定，因为全零矩阵需要形状)
        """
        device = z.device
        batch_size = z.size(0)
        
        # 1. 获取 Common Space 的特征维度 (F_common)
        # 我们可以通过投影层的输出维度直接获得
        f_common = self.proj_photonic.out_features 
        
        # 2. 构造 Dummy Context (x_ctx)
        # 形状: (Batch, seq_len, F_common)
        # 全零意味着不引入任何额外的时序信息，完全由 z 和 labels 驱动生成
        x_ctx_dummy = torch.zeros(batch_size, seq_len, f_common).to(device)
        
        # 3. 调用内部 VAE 的 decode
        # 现在的参数完美对应了: (x_ctx, z, labels)
        x_common = self.vae.decode(x_ctx_dummy, z, labels)
        
        # 4. 反投影回目标域
        if domain == "photonic":
            recon_x = self.deproj_photonic(x_common)
        elif domain == "onhw":
            recon_x = self.deproj_onhw(x_common)
        else:
            raise ValueError(f"Unknown domain: {domain}")
            
        return recon_x

    @torch.no_grad()
    def sample_conditioned(self, x, lengths, labels, domain, k=5, scale=0.5):
        """
        生成增强样本：
        1. 获取 input 的 mu
        2. 在 mu 附近采样 k 次
        3. 强制使用 input 的 label 进行解码 (保持类别一致性)
        """
        proj = self.proj_onhw if domain == "onhw" else self.proj_photonic
        deproj = self.deproj_onhw if domain == "onhw" else self.deproj_photonic

        x_proj = proj(x)
        mu, logvar, z = self.vae.encode(x_proj, lengths, labels) # Encode 原始数据
        std = torch.exp(0.5 * logvar)
        
        # 生成 k 个扰动
        # shape: (K, B, dim)
        z_k = mu.unsqueeze(0) + scale * torch.randn(k, *mu.shape, device=x.device) * std.unsqueeze(0)
        
        x_aug_list = []
        for i in range(k):
            # 解码时，使用原始 x_proj 作为 teacher forcing context
            # 这里的关键是：虽然 x 是原始的，但 z 变了，且 label 依然是对的
            x_recon_proj = self.vae.decode(x_proj, z_k[i], labels)
            x_recon = deproj(x_recon_proj)
            x_aug_list.append(x_recon)
            
        return torch.cat(x_aug_list, dim=0) # (K*B, T, F)
import torch
import torch.nn.functional as F

def vae_loss_time_freq(x_recon, x, lengths, mu, logvar,
                       beta=1e-2,      # KL 权重：比之前 1e-3 大一点
                       lambda_freq=0.1 # 频域损失权重
                       ):
    """
    x_recon, x: (B, T_max, F)    已经做过标准化
    lengths:    (B,)
    mu, logvar: (B, latent_dim)

    beta:        KL 权重（β-VAE）
    lambda_freq: 频域重构权重
    """
    if x.dim() == 4:
        x = x.squeeze(1)
    B, T_max, F = x.size()
    device = x.device

    # ========= 1) 时间域重构 =========
    mask = torch.arange(T_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.float()  # (B, T_max)

    diff_time = (x_recon - x) ** 2       # (B, T_max, F)
    diff_time = diff_time.sum(dim=-1)    # (B, T_max)
    diff_time = diff_time * mask         # 只在有效时间步
    recon_time = diff_time.sum() / mask.sum().clamp_min(1.0)

    # ========= 2) 频域重构 =========
    # 对时间维做 rFFT，得到 (B, T_fft, F)，使用 log-magnitude 更稳定
    X      = torch.fft.rfft(x, dim=1)         # complex, (B, T_fft, F)
    X_rec  = torch.fft.rfft(x_recon, dim=1)   # complex, (B, T_fft, F)

    mag     = torch.abs(X)
    mag_rec = torch.abs(X_rec)

    log_mag     = torch.log1p(mag)
    log_mag_rec = torch.log1p(mag_rec)

    diff_freq = (log_mag_rec - log_mag) ** 2   # (B, T_fft, F)
    # 频域这里可以整体平均，不再按 mask，因为 rFFT 本身是对整段信号处理
    recon_freq = diff_freq.mean()

    # ========= 3) KL (β-VAE) =========
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    total = recon_time + lambda_freq * recon_freq + beta * kld

    # 方便打印监控
    return total, recon_time.detach(), recon_freq.detach(), kld.detach()
from torch.nn.utils.rnn import pad_sequence
def collate_timeseries(batch):
    """
    batch: list of dicts, each has
      - "time_series": (T_i, F)
      - "letter_id": int (可选)
    输出:
      - x_reshaped: (B, 1, T_max, F)  <--- 修复了这里
      - lengths:    (B,)
      - labels:     (B,) or None
    """
    seqs = [b["time_series"] for b in batch]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    
    # 1. Padding: 得到 (B, T_max, F)
    x_padded = pad_sequence(seqs, batch_first=True)  
    
    # 2. Reshape: 增加一个维度以适配模型输入 (B, S, L, F)
    # 这里我们设 S=1 (Steps), L=T_max (Length)
    # 结果变成 (B, 1, T_max, F)
    x_reshaped = x_padded.unsqueeze(1) 

    if "letter_id" in batch[0]:
        labels = torch.tensor([b["letter_id"] for b in batch], dtype=torch.long)
    else:
        # 尝试检查 symbol_idx (部分 dataset 用这个名字)
        if "symbol_idx" in batch[0]:
            labels = torch.tensor([b["symbol_idx"] for b in batch], dtype=torch.long)
        else:
            labels = None
            
    return x_reshaped, lengths, labels
def collate_photonic_timeseries(batch):
    """
    batch: list[dict] from MultiModalRealDataset
    返回: x_padded, lengths
        x_padded: (B, T_max, F_photonic)
        lengths:  (B,)
    """
    # 1) 取出每条序列 & 长度
    xs = []
    lengths = []
    labels = []
    for b in batch:
        x = torch.as_tensor(b["time_series"], dtype=torch.float32)  # (T_i, F_photonic)
        xs.append(x)
        lengths.append(x.shape[0])
        labels.append(b["symbol_idx"]-10)


    lengths = torch.tensor(lengths, dtype=torch.long)  # (B,)
    labels = torch.tensor(labels, dtype=torch.long)
    # 2) 按时间维度 pad 到 T_max
    x_padded = pad_sequence(xs, batch_first=True)  # (B, T_max, F_photonic)

    return x_padded, lengths, labels
