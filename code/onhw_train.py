import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from onhw_model import OnHWDataset, ConditionalMultiDomainVAE, MultiModalRealDataset, collate_timeseries, collate_photonic_timeseries, vae_loss_time_freq


onhw_train_ds = OnHWDataset(split="train")
onhw_loader = DataLoader(
    onhw_train_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_timeseries,  
)

photonic_train_ds = MultiModalRealDataset(
    mode="train",
    use_mnist=False,
    use_emnist=False,
    subset="letter",  
)

photonic_loader = DataLoader(
    photonic_train_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_photonic_timeseries,  # 用我们刚刚写的
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_on_example, len_on_example, _ = next(iter(onhw_loader))
x_ph_example, len_ph_example, _ = next(iter(photonic_loader))

F_onhw     = x_on_example.shape[-1]
F_photonic = x_ph_example.shape[-1]
F_common   = 64         
latent_dim = 32         

vae_kwargs = dict(

)

model = ConditionalMultiDomainVAE(
    F_onhw=F_onhw,
    F_photonic=F_photonic,
    F_common=F_common,
    latent_dim=latent_dim,
    vae_kwargs=vae_kwargs,
    num_classes= 37,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_steps = 10_000  
onhw_iter = iter(onhw_loader)
ph_iter   = iter(photonic_loader)
beta_start = 1e-4
beta_end   = 1e-2
stats = torch.load("timeseries_norm_stats.pt")
mean, std = stats["mean"], stats["std"]
ph_norm_stats = torch.load("photonic_norm_stats.pt")
PH_MEAN = ph_norm_stats["mean"]          # shape: (F_photonic,)
PH_STD  = ph_norm_stats["std"]           # shape: (F_photonic,)
for step in range(num_steps):
    # --- 1) OnHW batch ---
    try:
        x_on, len_on, label_on = next(onhw_iter)
    except StopIteration:
        onhw_iter = iter(onhw_loader)
        x_on, len_on, label_on = next(onhw_iter)
    x_on = (x_on - mean) / (std + 1e-8)
    x_on   = x_on.to(device)           # (B, T_on, F_onhw)
    len_on = len_on.to(device)         # (B,)
    label_on = label_on.to(device)
    t = step / num_steps
    beta_on = beta_start + (beta_end - beta_start) * t
    x_recon_on, mu_on, logvar_on = model(x_on, len_on, labels = label_on, domain="onhw")
    loss_on, t_on, f_on, kld_on = vae_loss_time_freq(
        x_recon_on, x_on, len_on,
        mu_on, logvar_on,
        beta=beta_on,
        lambda_freq=0.1,
    )

    # --- 2) Photonic batch ---
    try:
        x_ph, len_ph, label_ph = next(ph_iter)
    except StopIteration:
        ph_iter = iter(photonic_loader)
        x_ph, len_ph, label_ph = next(ph_iter)
    x_ph = (x_ph - PH_MEAN) / (PH_STD + 1e-8)
    x_ph   = x_ph.to(device)           # (B, T_ph, F_photonic)
    len_ph = len_ph.to(device)
    label_ph = label_ph.to(device)

    x_recon_ph, mu_ph, logvar_ph = model(x_ph, len_ph, labels = label_ph, domain="photonic")
    loss_ph, t_ph, f_ph, kld_ph = vae_loss_time_freq(
        x_recon_ph, x_ph, len_ph,
        mu_ph, logvar_ph,
        beta=beta_on,
        lambda_freq=0.1,
    )

    loss = loss_on + loss_ph

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        print(
            f"[Step {step}] "
            f"OnHW: loss={loss_on.item():.4f}, time={t_on:.4f}, freq={f_on:.4f}, kld={kld_on:.4f} | "
            f"PH: loss={loss_ph.item():.4f}, time={t_ph:.4f}, freq={f_ph:.4f}, kld={kld_ph:.4f}"
        )

import os
import torch
import json

def save_conditional_vae(model, 
                         save_path, 
                         config, 
                         norm_stats=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'norm_stats': norm_stats
    }
    
    torch.save(checkpoint, save_path)


vae_config = {
    'F_onhw': F_onhw,            
    'F_photonic': F_photonic,   
    'F_common': F_common,      
    'num_classes': 37,         
    'latent_dim': latent_dim,    
    'vae_kwargs': vae_kwargs,   
    'label_emb_dim': 8           
}

stats_to_save = {
    'photonic_mean': PH_MEAN.cpu(), 
    'photonic_std': PH_STD.cpu(),
}

# 3. 执行保存
save_path = "checkpoints/best_cvae_conditional.pt"
save_conditional_vae(model, save_path, vae_config, stats_to_save)

import torch
import numpy as np
import os
from torch.utils.data import DataLoader


def generate_offline_augmentation(
    original_dataset, 
    vae_model, 
    save_path="./data/dataset_photonic_augmented.npz", 
    k=5,                 
    batch_size=32,
    device="cuda"
):

    vae_model.to(device)
    vae_model.eval()
    
    loader = DataLoader(
        original_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_photonic_timeseries 
    )
    
    all_X = []
    all_y = []
    all_is_aug = [] 

    
    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)      # (B, T, F)
            lengths = lengths.to(device)
            y = y.to(device)      # (B,)

            for i in range(x.shape[0]):
                real_seq = x[i, :lengths[i], :].cpu().numpy()
                all_X.append(real_seq)
                all_y.append(y[i].item())
                all_is_aug.append(0) 

            x_aug = vae_model.sample_conditioned(
                x, lengths, y, 
                domain="photonic", k=k, scale=0.5
            ) 
            
            x_aug = x_aug.view(k, x.size(0), x.size(1), x.size(2)).permute(1, 0, 2, 3)
            
            for b in range(x.size(0)):
                valid_len = lengths[b].item()
                label_val = y[b].item()
                
                for j in range(k): 
                    gen_seq = x_aug[b, j, :valid_len, :].cpu().numpy()
                    all_X.append(gen_seq)
                    all_y.append(label_val)
                    all_is_aug.append(1) 

    np.savez(
        save_path, 
        X=np.array(all_X, dtype=object), 
        y=np.array(all_y),
        is_aug=np.array(all_is_aug)
    )

generate_offline_augmentation(photonic_train_ds, model, k=5)