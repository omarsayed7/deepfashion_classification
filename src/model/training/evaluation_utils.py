import os, gc
import cv2, time
import numpy as np
import pandas as pd 
from tqdm import tqdm
from operator import itemgetter

import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F

from data_utils import *
from models import FashionNet

def model_evaluate(model, testloader, device):
    '''
    This function is used to calculate the accuracy of the model on the test set.
    '''
    model.eval()
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            # forward pass
            data, labels = data.to(device), target.to(device)
            outputs = model(data)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # ⭐️⭐️ Garbage Collection
            torch.cuda.empty_cache()
            _ = gc.collect()
        
    # loss and accuracy for the complete epoch
    acc = 100. * (valid_running_correct / len(testloader.dataset))
    return acc


def get_preds(model, testloader, device, threshold):
    '''
    This function is used to calculate the predictions of the model on certain dataset.
    '''
    model.eval()
    print('Calculating the confusion matrix')
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            # forward pass
            data, labels = data.to(device), target.to(device)
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
    return valid_running_correct, predlist, lbllist


def model_report(model, testloader, device, threshold=0.1):
    '''
    This function is used to calculate the accuracy of the model on the test set, and return the predictions.
    return: accuracy, confusion matrix, per-class accuracy, classification report.
    '''
    valid_running_correct, predlist, lbllist = get_preds(model, testloader, device, threshold)    
    # loss and accuracy for the complete epoch
    acc = 100. * (valid_running_correct / len(testloader.dataset))
    # Confusion matrix
    conf_mat = metrics.confusion_matrix(lbllist.numpy(), predlist.numpy())
    cls_report = metrics.classification_report(lbllist.numpy(), predlist.numpy())
    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    return acc, conf_mat, class_accuracy, cls_report


def top_losses(model, criterion, test_dataset, device):
    '''
    This function is used to calculate the top losses of the model on the train set.
    return: top losses dataframe.
    '''
    model.eval()
    print('Calculating the top losses of the data')
    top_losses_df = pd.DataFrame(columns=['Target', 'Predicted', 'Loss','Image'])
    predlist = torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0,dtype=torch.long, device='cpu')
    losseslist = []
    infolist = []
    valid_running_correct = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            data, target, img_info = test_dataset.__getitem__(i)
            # forward pass
            data, labels = data, target
            outputs = model(data.unsqueeze(0))
            # calculate the loss
            loss = criterion(outputs.squeeze(0), torch.tensor(labels))
            # calculate the accuracy
            _, preds = torch.max(outputs.squeeze(0).data, 0)
            valid_running_correct += (preds == labels).sum().item()
            if preds != labels:
                # Append top losses
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
                losseslist.append(loss.view(-1).cpu().numpy()[0])
                infolist.append(img_info)
        # ⭐️⭐️ Garbage Collection
        torch.cuda.empty_cache()
        _ = gc.collect()
                
    top_losses_df['Target'] = lbllist
    top_losses_df['Predicted'] = predlist
    top_losses_df['Loss'] = losseslist
    top_losses_df['Image'] = infolist
    acc = 100. * (valid_running_correct / len(test_dataset))
    return acc, top_losses_df