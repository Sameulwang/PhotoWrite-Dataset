# PhotoWrite-Dataset
## Reproducibility Guide

Due to the large size of the augmented dataset (1.3GB), we provide the **raw sensor data** and the **generation script**. Please follow the steps below to reproduce the training data and results.

### Step 1: Prepare Data
The raw data is located in `data/shadow_data.npz`. 
Run the following command to generate the full augmented dataset (including mixup and domain adaptation samples):

```bash
python code/onhw_train.py
python code/train.py
