
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import os
from model import MultiModalRealDataset, CNN_GRU_Attn_Strict_Proto, collate_with_aug_flag, plot_confusion_matrix, evaluate
class_names = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a','b','d','e','f','g','h','n','q','r','t']
num_neurons = 256
device = torch.device("cuda:0")
model_final = CNN_GRU_Attn_Strict_Proto(
        n_features=9,
        n_outputs=47,
        embed_dim=128,
        use_tecc = True,
        beta=0.0,             
        num_lstm_neuron=num_neurons, 
        num_fcn_neuron=num_neurons,  
    ).to(device) 
checkpoint_path = "checkpoints/best_model.pth"
state_dict = torch.load(checkpoint_path, map_location=device)
model_final.load_state_dict(state_dict)
digit_test  = MultiModalRealDataset(mode="test",  use_mnist=False, use_emnist=False, subset="letter")
test_loader  = DataLoader(digit_test,  batch_size=64, shuffle=False, collate_fn=collate_with_aug_flag)
avg_loss, acc_top1, acc_top3 = evaluate(
    model_final,         
    test_loader, 
    device, 
    eval_topk=5        
)
plot_confusion_matrix(
                model=model_final, 
                loader=test_loader, 
                device=device, 
                class_names=class_names, 
                mode=f"best_acc_{acc_top1:.3f}", 
                save_path='./'
            )
