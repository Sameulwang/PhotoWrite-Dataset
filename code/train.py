import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import collections
import os
from model import CNN_GRU_ResNet_Wrapper, train_one_epoch, evaluate 
from model import MultiModalRealDataset, collate_with_aug_flag, plot_confusion_matrix 
from model import CNN_GRU_Attn_Strict_Proto, evaluate_word_recognition, UserDataCache
class_names = [ '0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a','b','d','e','f','g','h','n','q','r','t']
def get_dataloaders(selected_train_id, batch_size=64 ):

    # aug_path = None
    aug_path=f'/mnt/d/shadow_to_ink/data/photonic_aug_UID_{selected_train_id}_k5.npz'
    print(f"Loading datasets... (Augmentation: {aug_path})")
    train = MultiModalRealDataset(selected_train_id, mode="train", use_mnist=False, use_emnist=False, subset="letter", aug_npz_path=aug_path)
    test  = MultiModalRealDataset(selected_train_id, mode="test",  use_mnist=False, use_emnist=False, subset="letter", angle = 60)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_with_aug_flag)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, collate_fn=collate_with_aug_flag)
    return train_loader, test_loader

def run_experiment(
    mode: str,
    num_neurons: int, 
    selected_train_id: int,
    dropout_p: float = 0.6,
    batch_size: int = 64,
    experiment_name: str = None
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = f"Exp_{mode}_Neu{num_neurons}"

    save_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(save_dir, exist_ok=True)


    train_loader, test_loader = get_dataloaders(selected_train_id=selected_train_id, batch_size=batch_size)

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
    best_model_path = os.path.join(save_dir, f"Best_Model_{experiment_name}.pt")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    for phase, epochs, drop_p, lr, weight_decay, eval_interval, mixup_prob in [
        (1, range(1, 61),   None, None, None, 5, 0.8), # Phase 1: 前60个epoch变化不大，每5个epoch测一次即可
        (2, range(61, 101), 0.4,  2e-4, 1e-4, 2, 0.4), # Phase 2: 每2个epoch测一次
        (3, range(101, 131),0.2,  5e-5, 1e-4, 1, 0.0), # Phase 3: 关键冲刺期，每个epoch测一次
        (4, range(131, 161),0.0,  2e-4, 1e-5, 1, 0.0)  # Phase 4: 极限压榨期，每个epoch测一次
    ]:
        print(f"\n====================================================")
        print(f"[Phase {phase}] Epochs: {epochs.start} to {epochs.stop-1}")
        print(f"====================================================")
        
        # 动态调整 Dropout 和 Optimizer (Phase 1 保持默认，直接沿用外面的)
        if phase > 1:
            if hasattr(model, 'dropout_td'): 
                model.dropout_td.p = drop_p
            if hasattr(model, 'dropout_after_gru'): 
                model.dropout_after_gru.p = drop_p
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in epochs:
            loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, epoch, 
                update_proto=False, 
                mixup_prob=mixup_prob
            )
            
            # 优化点 1：控制验证频率 (Eval Interval)
            if epoch % eval_interval == 0 or epoch == epochs.stop - 1:
                val_loss, val_acc, acc_top3 = evaluate(
                    model,         
                    test_loader, 
                    device, 
                    eval_topk=5       
                )
                
                # 只有 Phase 1 需要更新 scheduler (假设你用的是 ReduceLROnPlateau)
                if phase == 1:
                    scheduler.step(val_loss)
                
                print(f"[P{phase} Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}, Test_3={acc_top3:.3f}")

                acc = val_acc + acc_top3
                # 优化点 2：只保存/覆盖全局最好的那一个模型
                if acc > global_best_acc:
                    global_best_acc = acc
                    record = val_acc
                    # 直接覆盖同一个文件，避免产生几十个冗余的 .pt 文件
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  --> 🌟 New Best Accuracy: {global_best_acc:.3f}! Model saved.")
                    
            else:
                # 不 evaluate 的 epoch，只打印 train 的结果
                print(f"[P{phase} Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}")

    print(f"\n✅ Training Finished: {experiment_name}")
    print(f"🏆 Global Best Test Acc: {global_best_acc:.4f}")

    # 优化点 3：整个实验完全结束后，才画最后一次（也是最好的一次）混淆矩阵
    print(f"\n📊 Generating final confusion matrix for best model...")
    # 加载刚才存下来的最佳权重
    model.load_state_dict(torch.load(best_model_path))
    plot_confusion_matrix(
        model=model, 
        loader=test_loader, 
        device=device, 
        class_names=class_names, 
        mode=f"Final_Best_{global_best_acc:.3f}", 
        save_path=save_dir
    )
    
    return record, model

def run_experiment_ori(
    use_tecc: bool, 
    num_neurons: int, 
    selected_train_id: int,
    dropout_p: float = 0.6,
    batch_size: int = 256,
    experiment_name: str = None
):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if experiment_name is None:
        tecc_str = "TECC" if use_tecc else "TimeDomain"
        experiment_name = f"{tecc_str}_Neurons{num_neurons}_all"
    
    save_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(save_dir, exist_ok=True)


    train_loader, test_loader = get_dataloaders(selected_train_id=selected_train_id, batch_size=batch_size)

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
    record = 0.0
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    # 在所有 Phase 开始前，记录最佳模型保存路径
    best_model_path = os.path.join(save_dir, f"Best_Model_{experiment_name}.pt")

    for phase, epochs, drop_p, lr, weight_decay, eval_interval, mixup_prob in [
        (1, range(1, 61),   None, None, None, 5, 0.8), # Phase 1: 前60个epoch变化不大，每5个epoch测一次即可
        (2, range(61, 101), 0.4,  2e-4, 1e-4, 2, 0.4), # Phase 2: 每2个epoch测一次
        (3, range(101, 131),0.2,  5e-5, 1e-4, 1, 0.0), # Phase 3: 关键冲刺期，每个epoch测一次
        (4, range(131, 161),0.0,  2e-4, 1e-5, 1, 0.0)  # Phase 4: 极限压榨期，每个epoch测一次
    ]:
        print(f"\n====================================================")
        print(f"[Phase {phase}] Epochs: {epochs.start} to {epochs.stop-1}")
        print(f"====================================================")
        
        # 动态调整 Dropout 和 Optimizer (Phase 1 保持默认，直接沿用外面的)
        if phase > 1:
            model.dropout_td.p = drop_p
            if hasattr(model, 'dropout_after_gru'): 
                model.dropout_after_gru.p = drop_p
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in epochs:
            loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, epoch, 
                update_proto=False, 
                mixup_prob=mixup_prob
            )
            
            # 优化点 1：控制验证频率 (Eval Interval)
            if epoch % eval_interval == 0 or epoch == epochs.stop - 1:
                val_loss, val_acc, acc_top3 = evaluate(
                    model,         
                    test_loader, 
                    device, 
                    eval_topk=5       
                )
                
                # 只有 Phase 1 需要更新 scheduler (假设你用的是 ReduceLROnPlateau)
                if phase == 1:
                    scheduler.step(val_loss)
                
                print(f"[P{phase} Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}, Test={val_acc:.3f}, Test_3={acc_top3:.3f}")
                acc = val_acc + acc_top3
                # 优化点 2：只保存/覆盖全局最好的那一个模型
                if acc > global_best_acc:
                    global_best_acc = acc
                    record = val_acc
                    # 直接覆盖同一个文件，避免产生几十个冗余的 .pt 文件
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  --> 🌟 New Best Accuracy: {global_best_acc:.3f}! Model saved.")
                    
            else:
                # 不 evaluate 的 epoch，只打印 train 的结果
                print(f"[P{phase} Epoch {epoch}] Loss={loss:.3f}, Train={train_acc:.3f}")

    print(f"\n✅ Training Finished: {experiment_name}")
    print(f"🏆 Global Best Test Acc: {record:.4f}")

    # 优化点 3：整个实验完全结束后，才画最后一次（也是最好的一次）混淆矩阵
    print(f"\n📊 Generating final confusion matrix for best model...")
    # 加载刚才存下来的最佳权重
    model.load_state_dict(torch.load(best_model_path))
    class_names = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a','b','d','e','f','g','h','n','q','r','t']
    plot_confusion_matrix(
        model=model, 
        loader=test_loader, 
        device=device, 
        class_names=class_names, 
        mode=f"Final_Best_{record:.3f}", 
        save_path=save_dir
    )
    
 
    return record, model

if __name__ == "__main__":
    

    experiments = [
        # (True, 512),  
        (True, 256),  
        # (True, 128),  
        # (False, 256), 
        # (False, 512), 
        # (False, 128),  
    ]
    # experiments = [
    # #     # ('time', 256),  
    #     ('mfcc', 512),   
    #     ('mfcc', 128),   
    #     ('mfcc', 256), 
    #     ('tecc', 512),   
    #     ('tecc', 128),   
    #     ('tecc', 256), 
    #     ('resnet', 512),   
    #     ('resnet', 128),   
    #     ('resnet', 256), 
    # #     ('hybrid', 256),
    # ]
    results = {}
    # for use_tecc, neurons in experiments:

        # exp_name = f"{'TECC' if use_tecc else 'Time'}_Neu{neurons}_new"
        
    #     best_acc = run_experiment_ori(
    #         use_tecc=use_tecc, 
    #         num_neurons=neurons, 
    #         experiment_name=exp_name
    #     )
        
    #     results[exp_name] = best_acc
    # all_fold_results = {f"Exp_{mode}_Neu{neurons}": [] for mode, neurons in experiments}
    all_fold_results = {f"Exp_{'TECC' if mode else 'Time'}_Neu{neurons}": [] for mode, neurons in experiments}
    all_fold_char_results = collections.defaultdict(list)   # 单字母准确率
    all_fold_wordA_results = collections.defaultdict(list)  # Lexicon A (指令集)
    all_fold_wordB_results = collections.defaultdict(list)  # Lexicon B (1000词)
    # all_fold_wordC_results = collections.defaultdict(list)  # Lexicon C (5000词)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # 核心外层循环：12-Fold 交叉验证
    # =====================================================================
    for current_train_id in range(3):
        print("\n" + "="*70)
        print(f"🚀 Starting Fold {current_train_id + 1}/12 | Train Target ID: {current_train_id}")
        print("="*70)
        
        # -------------------------------------------------------------
        # 步骤 1：为当前 Fold 准备“纯净”的测试集和缓存
        # -------------------------------------------------------------
        # 严格的 One-shot 协议：测试用户必须排除参与训练的那个人 (current_train_id)
        test_uids = [uid for uid in range(12) if uid != current_train_id]
        
        # 实例化当前 Fold 的测试 Dataset
        test_dataset = MultiModalRealDataset(
            selected_train_id=current_train_id, 
            mode="test",  
            use_mnist=False, 
            use_emnist=False, 
            subset="letter", 
            angle=60
        )
        
        # 初始化数据缓存 (每个 Fold 只需构建一次，极其节省时间)
        user_cache = UserDataCache(test_dataset, class_names)
        
        # -------------------------------------------------------------
        # 步骤 2：遍历不同的实验/模型配置
        # -------------------------------------------------------------
        for mode, neurons in experiments:
            # exp_name = f"{mode}_Neu{neurons}"
            exp_name = f"Exp_{'TECC' if mode else 'Time'}_Neu{neurons}"
            current_run_name = f"{exp_name}_Fold{current_train_id}"
            
            # 训练模型，并获取当前 Fold 跑出来的 best_acc 和 加载了最佳权重的 model
            best_char_acc, best_model = run_experiment_ori(
                selected_train_id=current_train_id, 
                use_tecc=mode,
                num_neurons=neurons, 
                experiment_name=current_run_name, 
            )
            # best_char_acc, best_model = run_experiment(
            #     selected_train_id=current_train_id, 
            #     mode=mode,
            #     num_neurons=neurons, 
            #     experiment_name=current_run_name, 
            # )
            # 记录底层的单字母准确率
            all_fold_char_results[exp_name].append(best_char_acc*100)
            
            # -------------------------------------------------------------
            # 步骤 3：立刻在刚刚训练好的模型上进行词汇级评估！
            # -------------------------------------------------------------
            print(f"\n--- 进行词汇级测试 ({current_run_name}) ---")
            # 评估 Lexicon A (指令集)
            raw_A, sys_A = evaluate_word_recognition(
                best_model, user_cache, "/mnt/d/shadow_to_ink/code/lexicon_A_commands.txt", device, class_names, test_uids)
            all_fold_wordA_results[exp_name].append(sys_A)
            
            # 评估 Lexicon B (基础常用词 1000)
            raw_B, sys_B = evaluate_word_recognition(
                best_model, user_cache, "/mnt/d/shadow_to_ink/code/lexicon_B_1000.txt", device, class_names, test_uids)
            all_fold_wordB_results[exp_name].append(sys_B)
            
            # 评估 Lexicon C (通用扩展词 5000)
            # 如果你生成了 C，取消注释；如果没有，可以先跑 A 和 B
            # raw_C, sys_C = evaluate_word_recognition(
            #     best_model, user_cache, "/mnt/d/shadow_to_ink/code/lexicon_C_5000.txt", device, class_names, test_uids)
            # all_fold_wordC_results[exp_name].append(sys_C)


    # =====================================================================
    # 终极汇总打印：用于填入论文表格的数据 (Mean ± Std)
    # =====================================================================
    print("\n" + "🌟"*30)
    print("   Final 12-Fold Cross-Validation Statistics")
    print("🌟"*30)

    for mode, neurons in experiments:
        exp_name = f"Exp_{'TECC' if mode else 'Time'}_Neu{neurons}"
        # exp_name = f"{mode}_Neu{neurons}"
        
        # 提取列表
        char_list = all_fold_char_results[exp_name]
        wordA_list = all_fold_wordA_results[exp_name]
        wordB_list = all_fold_wordB_results[exp_name]
        
        print(f"\n📌 Configuration: {exp_name}")
        print(f"  - [底层] 孤立字母 Acc    : {np.mean(char_list):.2f}% ± {np.std(char_list):.2f}%")
        print(f"  - [系统] Lexicon A (指令): {np.mean(wordA_list):.2f}% ± {np.std(wordA_list):.2f}%")
        print(f"  - [系统] Lexicon B (1000): {np.mean(wordB_list):.2f}% ± {np.std(wordB_list):.2f}%")
    # # =====================================================================
    # # 1. 外层轮询：遍历 0 到 11 作为新手参与者进行 12-Fold 交叉验证
    # # =====================================================================
    # for current_train_id in range(12):
    #     print("\n" + "="*60)
    #     print(f"🚀 Starting Fold {current_train_id + 1}/12 | Train Target ID: {current_train_id}")
    #     print("="*60)
        
    #     # 内层循环：遍历你定义的不同模型架构/实验参数
    #     for mode, neurons in experiments:
    #         # exp_name = f"Exp_{mode}_Neu{neurons}"
    #         exp_name = f"Exp_{'TECC' if mode else 'Time'}_Neu{neurons}"
    #         # 给当前实验的 log 加上 Fold 编号，防止 TensorBoard 或权重文件覆盖
    #         current_run_name = f"{exp_name}_Fold{current_train_id}"
    #         print(f"   -> Running {current_run_name} ...")
            
    #         # 传入当前的 current_train_id 
    #         best_acc = run_experiment_ori(
    #             selected_train_id=current_train_id, 
    #             # mode=mode,
    #             use_tecc=mode, 
    #             num_neurons=neurons, 
    #             experiment_name=current_run_name, 
    #         )
            
    #         # 记录结果
    #         all_fold_results[exp_name].append(best_acc)
    #         print(f"   [Result] {exp_name} on Fold {current_train_id} Acc: {best_acc:.2f}%")

    # # =====================================================================
    # # 2. 全局统计：计算均值 (Mean) 和标准差 (Std)
    # # =====================================================================
    # print("\n" + "🌟"*25)
    # print("   Final 12-Fold Cross-Validation Statistics")
    # print("🌟"*25)

    # for exp_name, acc_list in all_fold_results.items():
    #     # 计算统计值
    #     mean_acc = np.mean(acc_list)
    #     std_acc = np.std(acc_list)
        
    #     print(f"\n📌 Configuration: {exp_name}")
    #     # 打印所有 fold 的具体数值，方便你写论文时画 Boxplot 或者 Scatter
    #     formatted_list = [f"{a:.2f}%" for a in acc_list]
    #     print(f"  - Individual Folds: {formatted_list}")
    #     # 打印最终用于论文表格的鲁棒性结果
    #     print(f"  - Robust Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # print("\n" + "="*40)
    # print("📊 Ablation Study")
    # print("="*40)
    # print(f"{'Experiment':<20} | {'Best Test Acc':<15}")
    # print("-" * 40)
    # for name, acc in results.items():
    #     print(f"{name:<20} | {acc:.4f}")
    # print("="*40)