import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import CNN_GRU_ResNet_Wrapper, train_one_epoch, evaluate, MultiModalRealDataset, collate_with_aug_flag, plot_confusion_matrix, CNN_GRU_Attn_Strict_Proto
class_names = [ '0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a','b','d','e','f','g','h','n','q','r','t']
def get_dataloaders(batch_size=64, aug_path='../data/dataset_photonic_augmented.npz'):

    # aug_path = None
    print(f"Loading datasets... (Augmentation: {aug_path})")
    digit_train = MultiModalRealDataset(mode="train", use_mnist=False, use_emnist=False, subset="letter", aug_npz_path=aug_path)
    digit_test  = MultiModalRealDataset(mode="test",  use_mnist=False, use_emnist=False, subset="letter", angle = 60)
    train_loader = DataLoader(digit_train, batch_size=batch_size, shuffle=True, collate_fn=collate_with_aug_flag)
    test_loader  = DataLoader(digit_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_with_aug_flag)
    return train_loader, test_loader

def run_experiment(
    mode: str,
    num_neurons: int, 
    dropout_p: float = 0.6,
    batch_size: int = 64,
    experiment_name: str = None
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = f"Exp_{mode}_Neu{num_neurons}"

    save_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(save_dir, exist_ok=True)


    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = CNN_GRU_ResNet_Wrapper(
        n_features=9,
        n_outputs=47,
        fusion_mode=mode,
        embed_dim=128,
        beta=0.0,             
        num_lstm_neuron=num_neurons, 
        num_fcn_neuron=num_neurons,  
        dropout_p=dropout_p
    ).to(device)


    global_best_acc = 0.0

    # ====================================================
    # Phase 1: BN + No Proto + High Mixup (Epoch 1-60)
    # ====================================================
    print("\n[Phase 1] Strategy: BN + No Proto + High Mixup")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    for epoch in range(1, 61):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.8
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5        
        )
        scheduler.step(val_loss)
        
        print(f"[P1 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}")
        
        if val_acc > 0.5 and val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase1_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    # ====================================================
    # Phase 2: Relax & Refine (Epoch 61-100)
    # ====================================================
    print("\n[Phase 2] Relax & Refine (Lower Dropout & Mixup)")
    

    model.dropout_td.p = 0.4
    if hasattr(model, 'dropout_after_gru'): model.dropout_after_gru.p = 0.4
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(61, 101):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.4  
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5       
        )
        print(f"[P2 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}")

        if val_acc > 0.6 and val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase2_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    print("\n[Phase 3] Final Sprint (No Mixup, Low Dropout)")
    
    model.dropout_td.p = 0.2
    if hasattr(model, 'dropout_after_gru'): model.dropout_after_gru.p = 0.2
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    for epoch in range(101, 131):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.0 
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5       
        )
        print(f"[P3 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}")

        if val_acc > 0.63 and val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase3_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    # ====================================================
    # Phase 4: Unleash (Epoch 131-160)
    # ====================================================
    print("\n[Phase 4] Unleash (Zero Dropout, High LR)")
    
    model.dropout_td.p = 0.0
    if hasattr(model, 'dropout_after_gru'): model.dropout_after_gru.p = 0.0
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    for epoch in range(131, 161):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.0
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,        
            test_loader, 
            device, 
            eval_topk=5        
        )
        print(f"[P4 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}")

        if val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase4_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader,
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    print(f"\n✅ Experiment ends: {experiment_name}")
    print(f"🏆 Test Acc: {global_best_acc:.4f}")
    
    return global_best_acc

def run_experiment_ori(
    use_tecc: bool, 
    num_neurons: int, 
    dropout_p: float = 0.6,
    batch_size: int = 64,
    experiment_name: str = None
):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if experiment_name is None:
        tecc_str = "TECC" if use_tecc else "TimeDomain"
        experiment_name = f"{tecc_str}_Neurons{num_neurons}_all"
    
    save_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(save_dir, exist_ok=True)


    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = CNN_GRU_Attn_Strict_Proto(
        n_features=9,
        n_outputs=47,
        embed_dim=128,
        beta=0.0,             
        use_tecc=use_tecc,   
        num_lstm_neuron=num_neurons, 
        num_fcn_neuron=num_neurons,  
        dropout_p=dropout_p
    ).to(device)

    global_best_acc = 0.0

    # ====================================================
    # Phase 1: BN + No Proto + High Mixup (Epoch 1-60)
    # ====================================================
    print("\n[Phase 1] Strategy: BN + No Proto + High Mixup")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    for epoch in range(1, 61):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.0
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5       
        )
        scheduler.step(val_loss)
        
        print(f"[P1 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}")
        
        if val_acc > 0.5 and val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase1_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    # ====================================================
    # Phase 2: Relax & Refine (Epoch 61-100)
    # ====================================================
    print("\n[Phase 2] Relax & Refine (Lower Dropout & Mixup)")
    
    model.dropout_td.p = 0.4
    if hasattr(model, 'dropout_after_gru'): model.dropout_after_gru.p = 0.4
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(61, 101):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.0 
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5        
        )
        print(f"[P2 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}, Test_3={acc_top3:.3f}")

        if val_acc > 0.6 and val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase2_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    # ====================================================
    # Phase 3: Final Sprint (Epoch 101-130)
    # ====================================================
    print("\n[Phase 3] Final Sprint (No Mixup, Low Dropout)")
    
    model.dropout_td.p = 0.2
    if hasattr(model, 'dropout_after_gru'): model.dropout_after_gru.p = 0.2
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    for epoch in range(101, 131):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.0 
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5      
        )
        print(f"[P3 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}, Test_3={acc_top3:.3f}")

        if val_acc > 0.63 and val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase3_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader,
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    # ====================================================
    # Phase 4: Unleash (Epoch 131-160)
    # ====================================================
    print("\n[Phase 4] Unleash (Zero Dropout, High LR)")
    
    model.dropout_td.p = 0.0
    if hasattr(model, 'dropout_after_gru'): model.dropout_after_gru.p = 0.0
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    for epoch in range(131, 161):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            update_proto=False, 
            mixup_prob=0.0
        )
        val_loss, val_acc, acc_top3 = evaluate(
            model,         
            test_loader, 
            device, 
            eval_topk=5        
        )
        print(f"[P4 Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}, Test_3={acc_top3:.3f}")

        if val_acc > global_best_acc:
            global_best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/Best_Phase4_{val_acc:.3f}.pt")
            plot_confusion_matrix(
                model=model, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{val_acc:.3f}", 
                save_path=save_dir
            )

    print(f"\n✅ 实验结束: {experiment_name}")
    print(f"🏆 最高 Test Acc: {global_best_acc:.4f}")
    
 
    return global_best_acc

if __name__ == "__main__":
    

    # experiments = [
    #     (False, 256),  
        # (True, 1024),  
        # (False, 256), 
        # (True, 64),
    # ]
    experiments = [
    #     # ('time', 256),  
        ('time', 512),   
        ('time', 128),   
        ('time', 64), 
    #     ('hybrid', 256),
    ]
    results = {}
    # for use_tecc, neurons in experiments:

    #     exp_name = f"{'TECC' if use_tecc else 'Time'}_Neu{neurons}_new"
        
    #     best_acc = run_experiment_ori(
    #         use_tecc=use_tecc, 
    #         num_neurons=neurons, 
    #         experiment_name=exp_name
    #     )
        
    #     results[exp_name] = best_acc
    for mode, neurons in experiments:
        exp_name = f"Exp_{mode}_Neu{neurons}"
        
        best_acc = run_experiment(
            mode=mode,
            num_neurons=neurons, 
            experiment_name=exp_name,
        )
        results[exp_name] = best_acc
    
    print("\n" + "="*40)
    print("📊 Ablation Study")
    print("="*40)
    print(f"{'Experiment':<20} | {'Best Test Acc':<15}")
    print("-" * 40)
    for name, acc in results.items():
        print(f"{name:<20} | {acc:.4f}")
    print("="*40)