import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import torch
import torch.nn.functional as F
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
import numpy as np
import string

idx2char = np.load("emnist_idx2char.npy", allow_pickle=True)
idx2char = [c.decode() if isinstance(c, bytes) else c for c in idx2char]

# 构建 char -> id 的反向表
char2idx = {ch: i for i, ch in enumerate(idx2char)}
def char_to_bymerge_id(ch: str, char2idx: dict):

    if ch in char2idx:
        return char2idx[ch]
    return -1
class OnHWDataset(Dataset):
    def __init__(self, root="onhw-chars_2021-06-30/onhw2_upper_indep_2", split="train"):

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
            ch = str(ch)           
            cid = char_to_bymerge_id(ch,char2idx)
            if cid < 0:

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
from torch.nn.utils.rnn import pad_sequence
import torch

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
    

    x_padded = pad_sequence(seqs, batch_first=True)  
    

    x_reshaped = x_padded.unsqueeze(1) 

    if "letter_id" in batch[0]:
        labels = torch.tensor([b["letter_id"] for b in batch], dtype=torch.long)
    else:

        if "symbol_idx" in batch[0]:
            labels = torch.tensor([b["symbol_idx"] for b in batch], dtype=torch.long)
        else:
            labels = None
            
    return x_reshaped, lengths, labels
class MultiModalRealDataset(Dataset):
    def __init__(self, 
                 photonic_digit_path="../data/dataset_photonic.npz", 
                 photonic_letter_dir="../data", 
                 use_mnist=True, use_emnist=True,
                 transform=None, mode="train", split_ratio=0.8,
                 aug_npz_path=None, angle = 0,
                 subset="all"):   # 🔑 新增参数
        """
        subset: {"all", "digit", "letter", "photo"}
        """
        self.samples = []
        self.mode = mode
        self.split_ratio = split_ratio
        self.subset = subset.lower()
        self.transform = transform or transforms.Normalize((0.1307,), (0.3081,))
        self.sample_weights = []

        self.source_stats = {
            "photonic_digit": [],
            "photonic_letter": [],
            "mnist": [],
            "emnist": []
        }

        if os.path.exists(photonic_digit_path) and self.subset in ["all", "digit"]:
            data = np.load("dataset_photonic.npz", allow_pickle=True)
            if mode == "train":
                X, y = data["X_train"], data["y_train"]
                # X1 = np.load("./data/X_data_0.npy")
                # y1 = np.load("./data/y_label_0.npy")
                # X = np.concatenate((X0, X1), axis=0)
                # y = np.concatenate((y0, y1), axis=0)
            elif mode == "val":
                X, y = data["X_val"], data["y_val"]
            elif mode == "test" and angle == 0:
                X, y = data["X_test"], data["y_test"]
            elif mode == "test" and angle == 30:
                X = np.load("./data/X_data_30.npy")
                y = np.load("./data/y_label_30.npy")
            elif mode == "test" and angle == 60:
                X = np.load("./data/X_data_60.npy")
                y = np.load("./data/y_label_60.npy")
            for s, l in zip(X, y):
                self.samples.append({
                    "time_series": torch.tensor(s, dtype=torch.float32),
                    "reference_image": None,
                    "symbol_idx": int(l),
                    "symbol_type": "digit",
                    "is_aug": False
                })
                self.source_stats["photonic_digit"].append(int(l))
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

        if mode == "train" and aug_npz_path is not None and os.path.exists(aug_npz_path) and self.subset in ["all", "letter"]:
            print(f"[Dataset] Loading augmented data from {aug_npz_path} ...")
            aug_data = np.load(aug_npz_path, allow_pickle=True)
            

            X_aug = aug_data['X']
            y_aug = aug_data['y']

            count_aug = 0
            for s, l in zip(X_aug, y_aug):

                symbol_idx = int(l) + 10 
                s = s.astype(np.float32)
                ts = torch.tensor(s, dtype=torch.float32)
                
                self.samples.append({
                    "time_series": ts,
                    "reference_image": None,
                    "symbol_idx": symbol_idx,
                    "symbol_type": "letter",
                    "is_aug": True  
                })
                self.source_stats["photonic_letter"].append(symbol_idx)
                count_aug += 1
            
            print(f"[Dataset] Added {count_aug} augmented samples.")


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

        self._print_stats()

    def _print_stats(self):
        print(f"\n[{self.mode}] label distribution:")
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqSelfAttentionTorch(nn.Module):
    """
    严格对齐 Keras SeqSelfAttention (attention_type='additive', attention_activation='sigmoid')
    e_{t,t'} = v^T tanh( Wt x_t + Wx x_{t'} + b_h ) + b_a
    a_{t} = sigmoid(e_t)  
    v_t   = sum_{t'} a_{t,t'} * x_{t'}
    """
    def __init__(self, feature_dim, units=32, use_additive_bias=True, use_attention_bias=True):
        super().__init__()
        self.units = units
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias

        self.Wt = nn.Linear(feature_dim, units, bias=False)
        self.Wx = nn.Linear(feature_dim, units, bias=False)
        self.bh = nn.Parameter(torch.zeros(units)) if use_additive_bias else None

        self.Wa = nn.Linear(units, 1, bias=False)
        self.ba = nn.Parameter(torch.zeros(1)) if use_attention_bias else None

    def forward(self, x):
        """
        x: (B, T, D)
        return: (B, T, D)  
        """
        B, T, D = x.shape

        # q: (B, T, 1, U), k: (B, 1, T, U)
        q = self.Wt(x).unsqueeze(2)
        k = self.Wx(x).unsqueeze(1)
        h = torch.tanh(q + k + (self.bh if self.bh is not None else 0.0))  # (B, T, T, U)

        # e: (B, T, T)
        e = self.Wa(h).squeeze(-1)
        if self.ba is not None:
            e = e + self.ba

        a = torch.sigmoid(e)                    # (B, T, T)

        # v_t = sum_{t'} a_{t,t'} x_{t'}
        v = torch.bmm(a, x)                     # (B, T, D)
        return v
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, n_fft=256, hop_length=64, n_mels=40, n_mfcc=20, sr=1000.0, fmin=20.0, fmax=None, use_tecc=False):

        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.fmin = fmin
        self.fmax = sr / 2.0 if fmax is None else fmax
        self.use_tecc = use_tecc


        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)


        n_freqs = n_fft // 2 + 1


        mel_fb = self._build_mel_filter(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                        fmin=self.fmin, fmax=self.fmax, n_freqs=n_freqs)
        self.register_buffer("mel_fb", mel_fb)


        dct = self._build_dct(self.n_mfcc, self.n_mels)
        self.register_buffer("dct_mat", dct)

    @staticmethod
    def _hz_to_mel(f):
        return 2595.0 * torch.log10(torch.tensor(1.0) + f / 700.0)

    @staticmethod
    def _mel_to_hz(m):
        return 700.0 * (10.0**(m / 2595.0) - 1.0)

    def _build_mel_filter(self, sr, n_fft, n_mels, fmin, fmax, n_freqs):

        mmin = self._hz_to_mel(torch.tensor(fmin, dtype=torch.float32))
        mmax = self._hz_to_mel(torch.tensor(fmax, dtype=torch.float32))
        m_pts = torch.linspace(mmin, mmax, n_mels + 2, dtype=torch.float32)
        f_pts = self._mel_to_hz(m_pts)

        bins = torch.floor((n_fft + 1) * f_pts / sr).long()
        bins = bins.clamp(0, n_fft // 2)

        fb = torch.zeros(n_mels, n_freqs, dtype=torch.float32)
        for i in range(1, n_mels + 1):
            left  = bins[i - 1].item()
            center= bins[i].item()
            right = bins[i + 1].item()

            if center <= left:
                center = min(left + 1, n_freqs - 1)
            if right <= center:
                right = min(center + 1, n_freqs - 1)


            if center > left:
                steps = center - left
                fb[i - 1, left:center] = torch.arange(
                    steps, dtype=fb.dtype
                ) / max(steps, 1)


            if right > center:
                steps = right - center
                fb[i - 1, center:right] = torch.arange(
                    steps, 0, -1, dtype=fb.dtype
                ) / max(steps, 1)


        fb_sum = fb.sum(dim=1, keepdim=True)
        fb = fb / (fb_sum + 1e-10)
        return fb
    
    def _delta(self, feat, win=2):

            N, C, T = feat.shape
            kernel = torch.arange(-win, win+1, device=feat.device, dtype=feat.dtype)
            kernel = kernel / (kernel.abs().sum() + 1e-8)        
            kernel = kernel.view(1, 1, -1)
            pad = (kernel.size(-1)//2, kernel.size(-1)//2)
            f = F.pad(feat, (pad[0], pad[1]), mode='replicate')
            out = F.conv1d(f, kernel.expand(C, 1, -1), groups=C)
            return out

    def _build_dct(self, n_mfcc, n_mels):

        n = torch.arange(n_mels).float()
        k = torch.arange(n_mfcc).float().unsqueeze(1)
        dct = torch.cos(math.pi / n_mels * (n + 0.5) * k)
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / n_mels)
        return dct  # (n_mfcc, n_mels)

    @staticmethod
    def _teager_energy(x):
        # x: (N, L)
        # Ψ[x[n]] = x[n]^2 - x[n-1]*x[n+1]
        x_prev = F.pad(x[:, :-1], (1, 0))
        x_next = F.pad(x[:, 1:], (0, 1))
        return x * x - x_prev * x_next

    def forward(self, x, output_format="vector"):
        """
        x: (B, S, L, Fea)
        返回: (B, n_mfcc)  —— 已在通道 Fea 与步长 S 维做均值聚合
        """
        if x.ndim == 4:
            B, S, L, Fea = x.shape
            SL = S * L
            x_flat = x.reshape(B, SL, Fea)
        else:
            B, SL, Fea = x.shape
            x_flat = x

        if output_format == "image":

            sig = x_flat.mean(dim=-1) 
            N = B
        else:

            sig = x_flat.permute(0, 2, 1).reshape(B * Fea, SL)
            N = B * Fea

        if self.use_tecc:
            sig = self._teager_energy(sig)  # (N, T)

        max_nfft = int(self.n_fft)

        nfft_pow = 2 ** int(math.floor(math.log2(max(4, SL))))
        n_fft_eff = min(max_nfft, max(4, nfft_pow))
        hop_eff = max(1, min(self.hop_length, n_fft_eff // 4))

        window = torch.hann_window(n_fft_eff, device=sig.device, dtype=sig.dtype)

        stft = torch.stft(
            sig,
            n_fft=n_fft_eff,
            hop_length=hop_eff,
            win_length=n_fft_eff,
            window=window,
            center=False,
            return_complex=True,
        )  # (N, n_freqs_eff, T_frames)
        power = (stft.real**2 + stft.imag**2).clamp_min(1e-12)


        n_freqs_eff = power.size(1)
        mel_fb = self.mel_fb.to(power.device, power.dtype)  # (n_mels, n_freqs_orig=self.n_fft//2+1)

        if mel_fb.size(1) != n_freqs_eff:

            src = torch.linspace(0, 1, mel_fb.size(1), device=power.device, dtype=power.dtype)
            dst = torch.linspace(0, 1, n_freqs_eff, device=power.device, dtype=power.dtype)
            mel_fb = torch.interp(dst[None, :].expand(self.n_mels, -1), src, mel_fb)  # (n_mels, n_freqs_eff)

        # (N, n_mels, T_frames)
        mel_spec = torch.einsum("mf,nft->nmt", mel_fb, power)
        log_mel = torch.log(mel_spec + 1e-12)
        if output_format == "image":

            return log_mel.unsqueeze(1)
        dct = self.dct_mat.to(log_mel.device, log_mel.dtype)

        mfcc_time = torch.einsum("km,nmt->nkt", dct, log_mel)  # (N, n_mfcc, T)
        d1 = self._delta(mfcc_time, win=2)
        d2 = self._delta(d1, win=2)

        mfcc = mfcc_time.mean(-1)
        d1m  = d1.mean(-1)
        d2m  = d2.mean(-1)
        mfcc_cat = torch.cat([mfcc, d1m, d2m], dim=1)  # (N, 3*n_mfcc)


        mfcc_cat = mfcc_cat.view(B, Fea, -1).mean(dim=1)  # (B, 3*n_mfcc)
        return mfcc_cat

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_normal_(self.W)

    def forward(self, x, labels):
        # x: (B, D), labels: (B,)
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = x @ W.t()               # cos(theta)

        theta = torch.acos(logits.clamp(-1+1e-7, 1-1e-7))
        target_logits = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, num_classes=logits.size(1)).float()
        logits_m = logits * (1 - onehot) + target_logits * onehot
        return logits_m * self.s

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision.models as models
class CNN_GRU_Attn_Strict_Proto(nn.Module):

    def __init__(self,
                 n_features,
                 n_outputs,
                 num_filter=64,
                 num_kernel=3,
                 num_lstm_neuron=512,
                 num_fcn_neuron=512,
                 attn_units=32,
                 dropout_p=0.5,
                 embed_dim=128,
                 beta=0.5,
                 beta_onhw=0.5,
                 tau=0.07,
                 momentum=0.9,
                 spec_n_fft=256,
                 spec_hop=64,
                 spec_n_mels=40,
                 spec_n_mfcc=20,
                 spec_sr=1000.0,
                 spec_fmin=20.0,
                 spec_fmax=None,
                 use_tecc=False):

        super().__init__()

        self.beta = beta
        self.beta_onhw = beta_onhw
        self.tau = tau
        self.use_tecc = use_tecc

        self.spec = SpectralFeatureExtractor(
            n_fft=spec_n_fft, hop_length=spec_hop,
            n_mels=spec_n_mels, n_mfcc=spec_n_mfcc,
            sr=spec_sr, fmin=spec_fmin, fmax=spec_fmax,
            use_tecc=use_tecc
        )
        self.spec_dim = spec_n_mfcc

        self.conv = nn.Conv1d(in_channels=n_features, out_channels=num_filter,
                              kernel_size=num_kernel, padding=0)
        self.bn_conv = nn.BatchNorm1d(num_filter) 
        self.pool = nn.MaxPool1d(kernel_size=num_kernel)
        self.dropout_td = nn.Dropout(dropout_p)

        self.gru = nn.GRU(input_size=num_filter,
                          hidden_size=num_lstm_neuron,
                          batch_first=True,
                          bidirectional=False)
        self.dropout_after_gru = nn.Dropout(dropout_p)

        # Attention
        self.attn = SeqSelfAttentionTorch(feature_dim=num_lstm_neuron,
                                          units=attn_units,
                                          use_additive_bias=True,
                                          use_attention_bias=True)

        fc_in_dim = num_lstm_neuron + self.spec_dim * 3
        self.fc1 = nn.Linear(fc_in_dim, num_fcn_neuron)
        self.bn_fc1 = nn.BatchNorm1d(num_fcn_neuron) 
        self.fc2 = nn.Linear(num_fcn_neuron, n_outputs)
        self.proj = nn.Linear(num_fcn_neuron, embed_dim)
        self.dl_head = nn.Linear(num_fcn_neuron, 2)  
        self.lambda_dl = 1.0

        if os.path.exists("prototypes.npy"):
            self.register_buffer("prototypes", torch.tensor(np.load("prototypes.npy"), dtype=torch.float))
        if os.path.exists("prototypes_fused.npy"):
            self.register_buffer("onhw_prototypes", torch.tensor(np.load("prototypes_fused.npy"), dtype=torch.float))

       
        if os.path.exists("proto_labels.npy"):
            self.register_buffer("proto_labels", torch.tensor(np.load("proto_labels.npy"), dtype=torch.long))
        if os.path.exists("proto_labels_fused.npy"):
            self.register_buffer("onhw_proto_labels", torch.tensor(np.load("proto_labels_fused.npy"), dtype=torch.long))


        self._init_xavier_like_keras()



    def _class_balanced_weight(self, labels, num_classes, beta=0.9999):
        counts = torch.bincount(labels, minlength=num_classes).float().to(labels.device)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights[effective_num == 0] = 0.0
        weights = weights * (num_classes / (weights.sum() + 1e-8))
        return weights

    def _build_letter_mask(self, labels, letter_set=None, letter_start=10):
        if letter_set is not None:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for v in letter_set:
                mask |= (labels == v)
            return mask
        else:
            return labels >= letter_start

    def _init_xavier_like_keras(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)


    def forward(self, x, return_emb=False):
        """

        x: (B, S, L, Fea)

        """
        B, S, L, Fea = x.shape
        h = x.view(B * S, L, Fea).permute(0, 2, 1) 
        h = self.conv(h)
        h = self.bn_conv(h) 
        h = F.relu(h)
        h = self.pool(h)
        h = self.dropout_td(h)
        h = h.flatten(start_dim=1)                        # (B*S, *)
        h = h.view(B, S, -1)                              # (B, S, *)
        h, _ = self.gru(h)                                # (B, S, H)
        h = self.dropout_after_gru(h)
        h = self.attn(h)                                  # (B, S, H)
        h_time = h.mean(dim=1)                            # (B, H=num_lstm_neuron)
        h_spec = self.spec(x)                             # (B, spec_dim=n_mfcc)
        h_cat = torch.cat([h_time, h_spec], dim=-1)       # (B, H + spec_dim)
        h_fc = self.fc1(h_cat)
        h_fc = self.bn_fc1(h_fc) 
        h_fc = F.relu(h_fc)
        logits = self.fc2(h_fc)                           # (B, n_outputs)
        logits_dl = self.dl_head(h_fc)  # (B,2)
        emb = self.proj(h_fc)                             # (B, embed_dim)
        if return_emb:
            return emb, logits, logits_dl

        return logits
    def compute_loss(self, x, labels, is_aug=None, mode="single", k=3, update_proto=True, 
                     digit_only=False, letter_teacher=None, letter_set=None, letter_start=10, current_beta = 0.0, current_beta_onhw = 0.0):
        

        seq_emb, logits, logits_dl = self.forward(x, return_emb=True)
        seq_emb_proj = self.proto_projector(seq_emb) 
        seq_emb_norm = F.normalize(seq_emb_proj, dim=-1) 

        losses = {}

        losses["cls"] = F.cross_entropy(logits, labels, label_smoothing=0.1)

        proto = self.prototypes

        logits_proto = seq_emb_norm @ proto.T / self.tau
        
        if is_aug is not None:

            real_mask = ~is_aug 
            
            if real_mask.sum() > 0:

                losses["contrast"] = F.cross_entropy(logits_proto[real_mask], labels[real_mask])
            else:

                losses["contrast"] = 0.0
        else:

            losses["contrast"] = F.cross_entropy(logits_proto, labels)

       

        total = losses["cls"] + current_beta * losses.get("contrast", 0.0)

        if hasattr(self, 'dl_head'):
            dl_labels = (labels >= 10).long()
            losses["dl"] = F.cross_entropy(logits_dl, dl_labels)
            total += self.lambda_dl * losses["dl"]
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim_matrix = torch.mm(proto_norm, proto_norm.t()) # (235, 235)

        return total, logits, losses, seq_emb_norm
class CNN_GRU_ResNet_Wrapper(nn.Module):
    def __init__(self,
                 n_features,
                 n_outputs,
                 fusion_mode='hybrid', 
                 num_filter=64,
                 num_kernel=3,
                 num_lstm_neuron=128,
                 num_fcn_neuron=128,
                 attn_units=32,
                 dropout_p=0.5,
                 embed_dim=128,
                 beta=0.0,
                 beta_onhw=0.0,
                 use_tecc_flag=True, 

                 spec_n_fft=256, spec_hop=64, spec_n_mels=40, spec_n_mfcc=20,
                 spec_sr=1000.0, spec_fmin=20.0, spec_fmax=None):
        
        super().__init__()
        self.fusion_mode = fusion_mode
        self.beta = beta
        self.beta_onhw = beta_onhw
        self.proto_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        if os.path.exists("prototypes.npy"):
            self.register_buffer("prototypes", torch.tensor(np.load("prototypes.npy"), dtype=torch.float))
        if os.path.exists("prototypes_fused.npy"):
            self.register_buffer("onhw_prototypes", torch.tensor(np.load("prototypes_fused.npy"), dtype=torch.float))
        
        if os.path.exists("proto_labels.npy"):
            self.register_buffer("proto_labels", torch.tensor(np.load("proto_labels.npy"), dtype=torch.long))
        if os.path.exists("proto_labels_fused.npy"):
            self.register_buffer("onhw_proto_labels", torch.tensor(np.load("proto_labels_fused.npy"), dtype=torch.long))
        self.tau = 0.07

        use_tecc_flag = True if fusion_mode in ['tecc', 'hybrid'] else False

        current_n_mels = 64 if fusion_mode == 'resnet' else spec_n_mels

        self.spec_extractor = SpectralFeatureExtractor(
            n_fft=spec_n_fft, hop_length=spec_hop, 
            n_mels=current_n_mels, n_mfcc=spec_n_mfcc,
            sr=spec_sr, fmin=spec_fmin, fmax=spec_fmax,
            use_tecc=use_tecc_flag 
        )
        
        self.spec_dim = spec_n_mfcc * 3

        if fusion_mode in ['time', 'hybrid']:
            self.conv = nn.Conv1d(in_channels=n_features, out_channels=num_filter,
                                  kernel_size=num_kernel, padding=0)
            self.bn_conv = nn.BatchNorm1d(num_filter)
            self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
            self.dropout_td = nn.Dropout(dropout_p)

            self.gru = nn.GRU(input_size=num_filter,
                              hidden_size=num_lstm_neuron,
                              batch_first=True,
                              bidirectional=False)
            self.dropout_after_gru = nn.Dropout(dropout_p)

            self.attn = SeqSelfAttentionTorch(feature_dim=num_lstm_neuron,
                                              units=attn_units,
                                              use_additive_bias=True,
                                              use_attention_bias=True)
            time_out_dim = num_lstm_neuron
        else:
            time_out_dim = 0

        if fusion_mode == 'resnet':
            self.resnet = models.resnet18(pretrained=False)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Identity()
            resnet_out_dim = 512
        else:
            resnet_out_dim = 0

        if fusion_mode == 'hybrid':
            fc_in_dim = time_out_dim + self.spec_dim
        elif fusion_mode == 'time':
            fc_in_dim = time_out_dim
        elif fusion_mode == 'tecc':
            fc_in_dim = self.spec_dim
        elif fusion_mode == 'resnet':
            fc_in_dim = resnet_out_dim
        else:
            raise ValueError(f"Unknown mode: {fusion_mode}")

        print(f"Model initialized in [{fusion_mode}] mode. FC Input Dim: {fc_in_dim}")

        self.fc1 = nn.Linear(fc_in_dim, num_fcn_neuron)
        self.bn_fc1 = nn.BatchNorm1d(num_fcn_neuron)
        self.fc2 = nn.Linear(num_fcn_neuron, n_outputs)
        self.proj = nn.Linear(num_fcn_neuron, embed_dim)
        

        self._init_xavier_like_keras()

    def _init_xavier_like_keras(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.xavier_normal_(param)
                    elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x, return_emb=False):
        """
        x: (B, S, L, Fea) 或 (B, Total_L, Fea)
        """

        if x.ndim == 4:
            B, S, L, Fea = x.shape
            x_flat = x.reshape(B, S*L, Fea) 
        else:
            x_flat = x
            B = x.shape[0]

        h_final = None

        if self.fusion_mode in ['time', 'hybrid']:
            # Permute -> (B, Fea, L)
            h = x_flat.permute(0, 2, 1)
            h = self.conv(h)
            h = self.bn_conv(h)
            h = F.relu(h)
            h = self.pool(h)
            h = self.dropout_td(h)
            
            # Permute back -> (B, L', Fea')
            h = h.permute(0, 2, 1)
            h, _ = self.gru(h)
            h = self.dropout_after_gru(h)
            h = self.attn(h)
            h_time = h.mean(dim=1) # Global Pooling
            
            h_final = h_time

        if self.fusion_mode in ['tecc', 'hybrid']:
            h_spec = self.spec_extractor(x, output_format="vector") 
            
            if self.fusion_mode == 'hybrid':
                h_final = torch.cat([h_time, h_spec], dim=-1)
            else:
                h_final = h_spec

        if self.fusion_mode == 'resnet':
            spec_img = self.spec_extractor(x, output_format="image")
            
            h_res = self.resnet(spec_img)
            h_final = h_res
        h_fc = self.fc1(h_final)
        h_fc = self.bn_fc1(h_fc)
        h_fc = F.relu(h_fc)
        
        logits = self.fc2(h_fc)
        
        if return_emb:
            emb = self.proj(h_fc)
            return emb, logits, None
            
        return logits
    def compute_loss(self, x, labels, is_aug=None, mode="single", k=3, update_proto=True, 
                     digit_only=False, letter_teacher=None, letter_set=None, letter_start=10, current_beta = 0.0, current_beta_onhw = 0.0):
        
        seq_emb, logits, logits_dl = self.forward(x, return_emb=True)
        seq_emb_proj = self.proto_projector(seq_emb) 
        seq_emb_norm = F.normalize(seq_emb_proj, dim=-1) 

        losses = {}
        
        losses["cls"] = F.cross_entropy(logits, labels, label_smoothing=0.1)

        proto = self.prototypes
        logits_proto = seq_emb_norm @ proto.T / self.tau
        
        if is_aug is not None:
            real_mask = ~is_aug 
            
            if real_mask.sum() > 0:
                losses["contrast"] = F.cross_entropy(logits_proto[real_mask], labels[real_mask])
            else:
                losses["contrast"] = 0.0
        else:
            losses["contrast"] = F.cross_entropy(logits_proto, labels)

        if hasattr(self, 'onhw_prototypes'):
             proto_onhw = self.onhw_prototypes.to(seq_emb_norm.device)
             onhw_labels_ref = self.onhw_proto_labels.to(labels.device) 
             
             logits_proto_onhw = seq_emb_norm @ proto_onhw.T / self.tau
             
             if is_aug is not None:
                 mask_final = ~is_aug
             else:
                 mask_final = torch.ones_like(labels, dtype=torch.bool)
                 
             valid_in_onhw = torch.isin(labels, onhw_labels_ref)
             mask_final = mask_final & valid_in_onhw  
             
             if mask_final.sum() > 0:
                 sub_logits = logits_proto_onhw[mask_final]     
                 sub_labels_raw = labels[mask_final]             
                 
                 target_indices = (sub_labels_raw.unsqueeze(1) == onhw_labels_ref.unsqueeze(0)).nonzero()[:, 1]
                 
                 losses["contrast_onhw"] = F.cross_entropy(sub_logits, target_indices)
             else:
                 losses["contrast_onhw"] = 0.0

        total = losses["cls"] + current_beta * losses.get("contrast", 0.0) + \
                current_beta_onhw * losses.get("contrast_onhw", 0.0)
                
        if hasattr(self, 'dl_head'):
            dl_labels = (labels >= 10).long()
            losses["dl"] = F.cross_entropy(logits_dl, dl_labels)
            total += self.lambda_dl * losses["dl"]
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim_matrix = torch.mm(proto_norm, proto_norm.t()) # (235, 235)

        num_protos = sim_matrix.size(0)
        
        if hasattr(self, "proto_labels") and self.proto_labels.numel() == num_protos:
            labels = self.proto_labels.view(-1, 1)
            mask_diff_cls = (labels != labels.T).float()

            loss_ortho = torch.sum((sim_matrix * mask_diff_cls) ** 2) / (mask_diff_cls.sum() + 1e-8)
            
        else:

            I = torch.eye(num_protos, device=sim_matrix.device)
            off_diagonal = sim_matrix - I
            loss_ortho = torch.sum(off_diagonal ** 2) / (num_protos * (num_protos - 1))

        losses["ortho"] = loss_ortho

        total += 0.0 * loss_ortho
        return total, logits, losses, seq_emb_norm

from torch.nn.utils.rnn import pad_sequence
def collate_photonic_timeseries(batch):
    """
    batch: list[dict] from MultiModalRealDataset

    """

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

    x_padded = pad_sequence(xs, batch_first=True)  # (B, T_max, F_photonic)

    return x_padded, lengths, labels
def collate_with_aug_flag(batch, n_steps=200, n_length=5):

    xs = []
    ys = []
    is_augs = []
    feat_dim = batch[0]["time_series"].size(1)

    for sample in batch:
        x = sample["time_series"]  # (T, F)

        total_len = n_steps * n_length
        if x.size(0) < total_len:
            pad_len = total_len - x.size(0)
            x = torch.cat([x, torch.zeros(pad_len, feat_dim)], dim=0)
        elif x.size(0) > total_len:
            x = x[:total_len, :]

        # reshape -> (n_steps, n_length, feat_dim)
        x = x.view(n_steps, n_length, feat_dim)
        xs.append(x)
        ys.append(sample["symbol_idx"])
        is_augs.append(sample["is_aug"])
        
    x_pad = pad_sequence(xs, batch_first=True) # (B, T, F)
    y = torch.tensor(ys, dtype=torch.long)
    is_aug_tensor = torch.tensor(is_augs, dtype=torch.bool)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    
    return x_pad, lengths, y, is_aug_tensor

import numpy as np
import torch

def train_one_epoch(model, loader, optimizer, device, epoch, mode="single", k=3, update_proto=False, mixup_alpha=0.4, mixup_prob=0.5):

    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_data in loader:
        if len(batch_data) == 4:
            seqs, lengths, labels, is_aug = batch_data
            is_aug = is_aug.to(device)
        elif len(batch_data) == 3:
            seqs, lengths, labels = batch_data
            is_aug = None
        else:
            seqs, labels = batch_data
            is_aug = None
        labels = labels
        seqs, labels = seqs.to(device), labels.to(device)
        
        if epoch < 20:
            current_beta = 0.0
            current_beta_onhw = 0.0
        else:
            current_beta = 0.0
            current_beta_onhw = 0.0

        optimizer.zero_grad()
        use_mixup = (np.random.random() < mixup_prob) and (labels.size(0) > 1)
        
        if use_mixup:
            lam = np.random.beta(mixup_alpha, mixup_alpha)

            index = torch.randperm(seqs.size(0)).to(device)

            mixed_seqs = lam * seqs + (1 - lam) * seqs[index]

            is_aug_a = is_aug
            is_aug_b = is_aug[index] if is_aug is not None else None

            
            if hasattr(model, "compute_loss"):
                loss_a, logits, _, _ = model.compute_loss(
                    mixed_seqs, labels, 
                    is_aug=is_aug_a, 
                    mode=mode, k=k, update_proto=False,
                    current_beta=current_beta, current_beta_onhw=current_beta_onhw
                )
                
                loss_b, _, _, _ = model.compute_loss(
                    mixed_seqs, labels[index], 
                    is_aug=is_aug_b, 
                    mode=mode, k=k, update_proto=False,
                    current_beta=current_beta, current_beta_onhw=current_beta_onhw
                )
            else:
                logits = model(mixed_seqs)
                loss_a = F.cross_entropy(logits, labels)
                loss_b = F.cross_entropy(logits, labels[index])
            
            loss = lam * loss_a + (1 - lam) * loss_b
            
        else:
            if hasattr(model, "compute_loss"):
                loss, logits, losses_dict, _ = model.compute_loss(
                    seqs, labels, 
                    is_aug=is_aug, 
                    mode=mode, k=k, update_proto=update_proto, 
                    current_beta=current_beta, current_beta_onhw=current_beta_onhw
                )
            else:
                logits = model(seqs)
                loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        
        preds = logits.argmax(dim=1)
        if use_mixup:
            if lam > 0.5:
                correct += (preds == labels).sum().item()
            else:
                correct += (preds == labels[index]).sum().item()
        else:
            correct += (preds == labels).sum().item()
            
        total += labels.size(0)

    return total_loss / total, correct / total

import torch
import torch.nn.functional as F
import numpy as np

def evaluate(model, loader, device, mode="single", k=3, eval_topk=3):

    model.eval()
    correct_1 = 0    
    correct_k = 0    
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 4:
                seqs, lengths, labels, is_aug = batch_data
                is_aug = is_aug.to(device)
            elif len(batch_data) == 3:
                seqs, lengths, labels = batch_data
                is_aug = None
            else:
                seqs, labels = batch_data
                is_aug = None
            
            seqs, labels = seqs.to(device), labels.to(device)

            if seqs.dim() == 4:
                seqs = seqs.squeeze()

            if hasattr(model, "compute_loss"):
                loss, logits, _, _ = model.compute_loss(
                    seqs, labels, 
                    is_aug=is_aug, 
                    mode=mode, k=k, update_proto=False
                )
            else:
                outputs = model(seqs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = F.cross_entropy(logits, labels)

            
            # A. Top-1 Accuracy
            preds = logits.argmax(dim=1)
            correct_1 += (preds == labels).sum().item()
            
            # B. Top-K Accuracy 
            _, topk_preds = logits.topk(eval_topk, dim=1, largest=True, sorted=True)
            correct_k += torch.eq(topk_preds, labels.view(-1, 1)).sum().item()

            total += labels.size(0)
            total_loss += loss.item() * seqs.size(0)

    return total_loss / total, correct_1 / total, correct_k / total


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os

def plot_confusion_matrix(model, loader, device, class_names, mode="best", save_path="./"):

    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 4:
                seqs, lengths, labels, is_aug = batch_data
            elif len(batch_data) == 3:
                seqs, lengths, labels = batch_data
            else:
                seqs, labels = batch_data
            labels = labels
            seqs = seqs.to(device)

            logits = model(seqs) 
            if isinstance(logits, tuple):
                logits = logits[1]
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    plt.figure(figsize=(14, 12)) 
    sns.heatmap(
        cm, 
        annot=False,        
        fmt=".2f", 
        cmap="Blues",      
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy (Recall)'} 
    )
    
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.title(f"Confusion Matrix ({mode})", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8) 
    plt.yticks(rotation=0, fontsize=8)

    os.makedirs(save_path, exist_ok=True)
    fname = os.path.join(save_path, f"confusion_{mode}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close() 
    print(f"✅ Confusion matrix saved to {fname}")

    np.fill_diagonal(cm, 0)
 
    flattened = cm.flatten()
    indices = np.argsort(flattened)[::-1][:5] # Top 5 errors
    print("-" * 40)
    print(f"🚨 Top 5 Confusing Pairs in {mode}:")
    num_classes_actual = cm.shape[0]
    current_class_names = class_names
    for idx in indices:
        row = idx // len(class_names)
        col = idx % len(class_names)
        if row >= num_classes_actual or col >= num_classes_actual:
            continue
        val = cm[row, col]
        if val > 0.01: 
            true_char = current_class_names[row]
            pred_char = current_class_names[col]
            print(f"  True: '{true_char}' --> Pred: '{pred_char}' (Error Rate: {val:.2%})")
    print("-" * 40)

import torch

def compute_photonic_stats(dataset):

    sum_x = None
    sum_x2 = None
    count = 0

    for i in range(len(dataset)):
        item = dataset[i]
        x = torch.as_tensor(item["time_series"], dtype=torch.float32)  # (T_i, F_photonic)

        if sum_x is None:
            F = x.shape[-1]
            sum_x  = torch.zeros(F, dtype=torch.float32)
            sum_x2 = torch.zeros(F, dtype=torch.float32)

        sum_x  += x.sum(dim=0)
        sum_x2 += (x ** 2).sum(dim=0)
        count  += x.shape[0]  

    mean = sum_x / count
    var  = sum_x2 / count - mean ** 2
    std  = torch.sqrt(torch.clamp(var, min=1e-6))

    return mean, std