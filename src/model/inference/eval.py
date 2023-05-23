import os, gc, sys
import cv2, time
import numpy as np
import pandas as pd 
from tqdm import tqdm
from operator import itemgetter

import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F

import wandb
import glob
import argparse
import numpy as np
import pandas as pd 
from pathlib import Path
from tqdm import tqdm
from typing import List
from datetime import datetime
import random
import json, os, warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, OneCycleLR

#Local imports
sys.path.append(r"../training")
from models import FashionNet
from dataset import FashionDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(args):
    transform = transforms.Compose(
                [transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    eval_data_dir = args.eval_data_dir 
    # Using readlines()
    file1 = open(f"{eval_data_dir}/raw_annotations.txt", 'r')
    Lines = file1.readlines()
    X_test = []
    y_test = []
    # Strips the newline character
    for line in Lines:
        label = int(line.split("\t")[-1].strip())
        img_path = line.split("\t")[0].strip().split('data/')[-1]
        y_test.append(label)
        X_test.append(img_path)
        
    test_dataset = FashionDataset(X_test, y_test, args.main_data_dir, transform)

    # ref https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#1-consider-using-another-learning-rate-schedule
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Opening mapping file
    with open(f'{args.main_data_dir}/classification_data/mapping.json') as json_file:
        mapping_dict = json.load(json_file)

    model = FashionNet(model_name = args.model_name, num_classes = 22, dropout = 0.5, freeze_backbone = False)
    best_model_metadata = torch.load(args.weights_path, map_location='cpu')
    model.load_state_dict(best_model_metadata['model_state_dict'])
    model.to(device)

    model.eval()
    print('Starting to Evaluate on the data {args.eval_data_dir}')
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            # forward pass
            #data, labels = data.to(device), target.to(device)
            data, labels = data, target
            outputs = model(data)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
        # ⭐️⭐️ Garbage Collection
        torch.cuda.empty_cache()
        _ = gc.collect()
    # loss and accuracy for the complete epoch
    acc = 100. * (valid_running_correct / len(test_loader.dataset))
    # Confusion matrix
    cls_report = metrics.classification_report(lbllist.numpy(), predlist.numpy())
    print(acc)
    print("\n")
    print(cls_report)

if __name__ == "__main__":
    '''
    Main function, used to parse the arguments and call the main function
    '''
    parser = argparse.ArgumentParser(description="Evaluation on a data")
    parser.add_argument('-weights_path', '--weights_path', type= str, help= 'path to the weights of the model', default = "../../../models/resnet-50_0.0001_True_64_categorical_crossentropy_0.5_1_V1/best_model.pth")
    parser.add_argument('-model_name', '--model_name', type= str, help= 'model name', default="resnet-50")
    parser.add_argument('-eval_data_dir', '--eval_data_dir', type= str, help= 'path to the evaluation data', default="../../../data/eval_dataset")
    parser.add_argument('-main_data_dir', '--main_data_dir', type= str, help= 'path to the main data directory of the project', default="../../../data")
    args = parser.parse_args()

    args1 = args
    eval(args1)
