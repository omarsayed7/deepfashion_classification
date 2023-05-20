import os, sys, cv2
import pandas as pd 
from pathlib import Path


def show_sample_img(fashion_df:pd.DataFrame, data_dir:Path):
    sample_raw = fashion_df.sample()
    img = cv2.imread(f"{data_dir}/{sample_raw['img_path'].values[0]}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    print(f"{sample_raw['label_str'].values[0]}")
    return img


def show_from_batch(data_loader, label_mapping_dict):
    img_batch, label_batch = next(iter(train_loader))
    img = img_batch[0].permute(1, 2, 0)
    label = label_batch[0].numpy()
    label_str = label_mapping[int(label)]
    return img, label_str