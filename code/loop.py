import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from onhw_model import OnHWDataset, ConditionalMultiDomainVAE, collate_timeseries, collate_photonic_timeseries, vae_loss_time_freq
from model import MultiModalRealDataset
# ==========================================
# 1. 封装核心训练与生成函数
# ==========================================
def train_and_generate_vae(experiment_name, selected_train_id, num_steps=10000, k_aug=5):
    print(f"\n{'='*50}")
    print(f"🚀 Starting VAE Experiment: {experiment_name} | Target UID: {selected_train_id}")
    print(f"{'='*50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据加载 ---
    onhw_train_ds = OnHWDataset(split="train")
    onhw_loader = DataLoader(
        onhw_train_ds, batch_size=64, shuffle=True, collate_fn=collate_timeseries
    )

    # 🚨 动态传入 selected_train_id，控制当前使用哪个新手的数据
    photonic_train_ds = MultiModalRealDataset(
        selected_train_id=selected_train_id, 
        mode="train",
        use_mnist=False,
        use_emnist=False,
        subset="letter",
    )
    photonic_loader = DataLoader(
        photonic_train_ds, batch_size=64, shuffle=True, collate_fn=collate_photonic_timeseries
    )

    # --- 模型初始化 ---
    x_on_example, len_on_example, _ = next(iter(onhw_loader))
    x_ph_example, len_ph_example, _ = next(iter(photonic_loader))

    F_onhw     = x_on_example.shape[-1]
    F_photonic = x_ph_example.shape[-1]
    F_common   = 64         
    latent_dim = 32         
    vae_kwargs = dict()

    model = ConditionalMultiDomainVAE(
        F_onhw=F_onhw,
        F_photonic=F_photonic,
        F_common=F_common,
        latent_dim=latent_dim,
        vae_kwargs=vae_kwargs,
        num_classes=37,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # --- 加载归一化统计量 ---
    stats = torch.load("/mnt/d/shadow_to_ink/code/timeseries_norm_stats.pt")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    
    ph_norm_stats = torch.load("/mnt/d/shadow_to_ink/code/photonic_norm_stats.pt")
    PH_MEAN = ph_norm_stats["mean"].to(device)
    PH_STD  = ph_norm_stats["std"].to(device)

    # --- 训练循环 ---
    onhw_iter = iter(onhw_loader)
    ph_iter   = iter(photonic_loader)
    beta_start = 1e-4
    beta_end   = 1e-2

    model.train()
    for step in range(num_steps):
        # 1) OnHW batch
        try:
            x_on, len_on, label_on = next(onhw_iter)
        except StopIteration:
            onhw_iter = iter(onhw_loader)
            x_on, len_on, label_on = next(onhw_iter)
            
        x_on = x_on.to(device)
        x_on = (x_on - mean) / (std + 1e-8)
        len_on = len_on.to(device)
        label_on = label_on.to(device)
        
        t = step / num_steps
        beta_on = beta_start + (beta_end - beta_start) * t
        
        x_recon_on, mu_on, logvar_on = model(x_on, len_on, labels=label_on, domain="onhw")
        loss_on, t_on, f_on, kld_on = vae_loss_time_freq(
            x_recon_on, x_on, len_on, mu_on, logvar_on, beta=beta_on, lambda_freq=0.1
        )

        # 2) Photonic batch
        try:
            x_ph, len_ph, label_ph = next(ph_iter)
        except StopIteration:
            ph_iter = iter(photonic_loader)
            x_ph, len_ph, label_ph = next(ph_iter)
            
        x_ph = x_ph.to(device)
        x_ph = (x_ph - PH_MEAN) / (PH_STD + 1e-8)
        len_ph = len_ph.to(device)
        label_ph = label_ph.to(device)

        x_recon_ph, mu_ph, logvar_ph = model(x_ph, len_ph, labels=label_ph, domain="photonic")
        loss_ph, t_ph, f_ph, kld_ph = vae_loss_time_freq(
            x_recon_ph, x_ph, len_ph, mu_ph, logvar_ph, beta=beta_on, lambda_freq=0.1
        )

        loss = loss_on + loss_ph

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:  # 降低打印频率，避免日志太长
            print(f"   [Step {step:05d}] OnHW Loss: {loss_on.item():.4f} | PH Loss: {loss_ph.item():.4f}")

    # --- 保存模型 ---
    save_path = f"checkpoints/cvae_{experiment_name}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'F_onhw': F_onhw, 'F_photonic': F_photonic, 'F_common': F_common, 
            'num_classes': 37, 'latent_dim': latent_dim, 'vae_kwargs': vae_kwargs
        },
        'norm_stats': {'photonic_mean': PH_MEAN.cpu(), 'photonic_std': PH_STD.cpu()}
    }, save_path)
    print(f"✅ Model saved to {save_path}")

    # --- 生成离线增强数据 ---
    gen_path = f"./data/photonic_aug_{experiment_name}.npz"
    print(f"⏳ Generating augmented data for {experiment_name}...")
    generate_offline_augmentation(photonic_train_ds, model, save_path=gen_path, k=k_aug, device=device)
    print(f"✅ Augmented data saved to {gen_path}")


# ==========================================
# 2. 增强数据生成函数 (保持原样，略微优化传参)
# ==========================================
def generate_offline_augmentation(original_dataset, vae_model, save_path, k=5, batch_size=32, device="cuda"):
    vae_model.eval()
    loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_photonic_timeseries)
    
    all_X, all_y, all_is_aug = [], [], []

    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            for i in range(x.shape[0]):
                all_X.append(x[i, :lengths[i], :].cpu().numpy())
                all_y.append(y[i].item())
                all_is_aug.append(0) 

            # VAE 生成
            x_aug = vae_model.sample_conditioned(x, lengths, y, domain="photonic", k=k, scale=0.5) 
            x_aug = x_aug.view(k, x.size(0), x.size(1), x.size(2)).permute(1, 0, 2, 3)
            
            for b in range(x.size(0)):
                valid_len, label_val = lengths[b].item(), y[b].item()
                for j in range(k): 
                    all_X.append(x_aug[b, j, :valid_len, :].cpu().numpy())
                    all_y.append(label_val)
                    all_is_aug.append(1) 

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, X=np.array(all_X, dtype=object), y=np.array(all_y), is_aug=np.array(all_is_aug))

# ==========================================
# 3. 执行轮询实验
# ==========================================
if __name__ == "__main__":
    # 假设你想看前 3 个不同用户数据训练出来的 VAE 生成结果有何差异
    test_uids = [3, 4, 5,6,7,8,9,10,11] 
    
    for uid in test_uids:
        exp_name = f"UID_{uid}_k5"
        # 为了快速测试，你可以把 num_steps 先调小（比如 2000），跑通后再调回 10000
        train_and_generate_vae(experiment_name=exp_name, selected_train_id=uid, num_steps=10000, k_aug=5)

    print("\n🎉 所有实验生成完毕！你可以加载生成的 npz 文件进行对比了。")