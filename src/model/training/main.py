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
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, OneCycleLR

#Local imports
from data_utils import *
from dataset import FashionDataset
from training_utils import *
from evaluation_utils import *


warnings.filterwarnings('ignore')
torch.backends.cudnn.bencmark = True

SEED = 4
random.seed(SEED)

# Log in to your W&B account
#wandb.login()
os.environ["WANDB_CACHE_DIR "] = "wandb/cache"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    '''
    Main function to train the model, and save the model and plots to disk, and log the results to W&B, if enabled.
    [Args] args: The arguments passed to the script.
    '''
    VERSION = args.experiment_version
    print(f"[INFO] Starting to training the {VERSION} experiment")
    main_data_dir = Path("../../../data")
    classification_data_dir = Path(main_data_dir / "classification_data")
    models_dir = Path("../../../models")
    fashion_df_path = Path( classification_data_dir / "fashion_data.csv")
    label_mapping_path = Path(classification_data_dir / "mapping.json")

    use_wandb = args.use_wandb

    #load the mapping file
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    #read the fashion dataframe file 
    fashion_df = pd.read_csv(fashion_df_path)
    #split the dataset into train, valid, and test
    X_train, X_test, y_train, y_test = train_test_split(fashion_df['img_path'].tolist(), fashion_df['label_id'].tolist(),
                                                    stratify=fashion_df['label_id'].tolist(), 
                                                    test_size=0.13,
                                                    random_state = SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        stratify=y_train, 
                                                        test_size=0.14,
                                                        random_state = SEED)

    print(f"Train data: {len(X_train)}, Validation data: {len(X_val)}, Test data: {len(X_val)}")

    #Update the number of classes of the network
    num_classes = len(set(fashion_df['label_id'].tolist()))
    if args.use_wandb:
        wandb.init(
            project="fashion-classification",
            # Set entity to specify your username or team name
            # ex: entity="wandb",
            config={
                "version":args.experiment_version,
                "dropout": args.dropout,
                "loss": args.loss,
                "epoch": args.epochs,
                "batch_size": args.batch_size,
                'lr': args.learning_rate,
                'lr_scheduler': args.lr_scheduler,
                'num_classes': num_classes,
                'model_name': args.model_name,
                'is_freeze_backbone': args.freeze_backbone
            })
        config = wandb.config

    '''
    log the experiment params here
    '''
    net = FashionNet(model_name = args.model_name, num_classes = num_classes, dropout = args.dropout, freeze_backbone = args.freeze_backbone)
    net.to(device)
    
    #Create version name for saving experiment logs and model
    version_loss = args.loss
    exp_params = f"{args.model_name}_{args.learning_rate}_{args.lr_scheduler}_{args.batch_size}_{version_loss}_{args.dropout}_{args.epochs}"
    VERSION_NAME = f'{exp_params}_V{VERSION}'

        
    #Create experiment directory
    if not os.path.exists(f"{models_dir}/{VERSION_NAME}"):
        os.makedirs(f"{models_dir}/{VERSION_NAME}")

    #write the args to a json file for future reference
    with open(f"{models_dir}/{VERSION_NAME}/args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    #Dataset 
    transform = transforms.Compose(
                    [transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = FashionDataset(X_train, y_train, classification_data_dir, transform)
    valid_dataset = FashionDataset(X_val, y_val, classification_data_dir, transform)
    test_dataset = FashionDataset(X_test, y_test, classification_data_dir, transform)

    # ref https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#1-consider-using-another-learning-rate-schedule
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    validation_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True
    )


    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )


    epochs = args.epochs
    steps_per_epoch = len(train_loader)

    if args.loss == "categorical_crossentropy":
        train_criterion= nn.CrossEntropyLoss()
        val_criterion = nn.CrossEntropyLoss()
        

    optimizer= optim.Adam(net.parameters(),lr= args.learning_rate)
    if args.lr_scheduler:
        scheduler = OneCycleLR(optimizer, 
                            max_lr = 0.01, # Upper learning rate boundaries in the cycle for each parameter group
                            steps_per_epoch = steps_per_epoch, # The number of steps per epoch to train for.
                            epochs = epochs, # The number of epochs to train for.
                            anneal_strategy = 'linear') # Specifies the annealing strategy

    save_best_model = SaveBestModel()

    # lists to keep track of losses and accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_val_loss = None
    best_val_acc = None
    # start the training
    print("Start training experiment ", args.experiment_version, " : ", " ".join(VERSION_NAME.split("_")))
    if args.use_wandb:
        wandb.watch(model, log_freq=epochs)
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs} -- V{args.experiment_version}")
        train_epoch_loss, train_epoch_acc = train(net, train_loader, optimizer, 
                                                train_criterion, device = device)
        valid_epoch_loss, valid_epoch_acc = validate(net, validation_loader, val_criterion, device = device)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        if args.use_wandb:
            wandb.log({"train_loss": train_epoch_loss, "train_acc": train_epoch_acc, 'valid_loss': valid_epoch_loss, 'valid_acc': valid_epoch_acc})
            
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.1f}% - Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.1f}%  -- V{args.experiment_version}")
        # save the best model till now if we have the least loss in the current epoch
        best_model_data = save_best_model(valid_epoch_loss, valid_epoch_acc, epoch, net, 
                                      optimizer, val_criterion, models_dir, VERSION_NAME)
        if best_model_data != None:
            best_val_loss, best_val_acc = best_model_data
        if args.lr_scheduler:
            #Update learning scheduler
            scheduler.step()
        print('-'*50)
        
    # save the trained model weights for a final time
    save_model(epochs, net, optimizer, val_criterion, models_dir, VERSION_NAME)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss, models_dir, VERSION_NAME)
    print(f'TRAINING COMPLETE  -- V{args.experiment_version}')


    logging_df = pd.read_csv(f"{models_dir}/logging/experiments_logs.csv")
    values = [datetime.now(),VERSION_NAME, best_val_loss, best_val_acc, args.model_name, args.freeze_backbone, args.loss, args]
    logging_df_extended = pd.DataFrame([values], columns = ['time', 'exp_dir', 'best_val_loss', 'best_val_acc', 'model_name', 'freeze_backbone', 'train_loss', 'args'])
    logging_df = pd.concat([logging_df, logging_df_extended])
    logging_df.to_csv(f"{models_dir}/logging/experiments_logs.csv", index=False)

    #Evaluation
    print(f"Evaluation of the best model on the validation data  -- V{args.experiment_version}")
    evaluate_mode = args.eval_mode

    transform = transforms.Compose(
                [transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if evaluate_mode == "valid":
        test_dataset = FashionDataset(X_val, y_val, classification_data_dir, transform)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle = True, pin_memory=True, num_workers = 0)
    elif evaluate_mode == "train":
        test_dataset = FashionDataset(X_train, y_train, classification_data_dir, transform)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle = True, pin_memory=True, num_workers = 0)
    else:
        test_dataset = FashionDataset(X_test, y_test, classification_data_dir, transform)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle = True, pin_memory=True, num_workers = 0)

    model = FashionNet(model_name = args.model_name, num_classes = num_classes, dropout = args.dropout, freeze_backbone = args.freeze_backbone)
    model.to(device)
    
    if args.loss == "categorical_crossentropy":
        val_criterion = nn.CrossEntropyLoss()
    
    best_model_metadata = torch.load(f"{models_dir}/{VERSION_NAME}/best_model.pth", map_location='cpu')
    model.load_state_dict(best_model_metadata['model_state_dict'])

    acc, conf_mat, class_accuracy, cls_report = model_report(model, test_loader, device = device)
    with open(f'{models_dir}/{VERSION_NAME}/cls_report.txt', 'w') as f:
        f.writelines(cls_report)

    if args.calculate_top_losses:
        test_dataset = FashionDataset(X_train, y_train, classification_data_dir, transform, True)
        _, top_losses_df = top_losses(model, val_criterion, test_dataset, device = device)
        top_losses_df.sort_values('Loss', inplace=True, ascending=[False])
        top_losses_df.to_csv(f"{models_dir}/{VERSION_NAME}/top_losses.csv", index=False)

def setup(args):
    print("[INFO] Starting to Setup the project")
    main_data_dir = Path("../../../data")
    models_dir = Path("../../../models")

    dir_list = [f"{main_data_dir}", f"{models_dir}", f"{models_dir}/logging", 
    f"{main_data_dir}/classification_data", f"{main_data_dir}/classification_data/raw_annotations"]
    for directory in dir_list:
        if not os.path.exists(directory):
            # If it doesn't exist, create it
            os.makedirs(directory)
    #Creating the experiments logging csv file 
    logging_df = pd.DataFrame(columns = ['time', 'exp_dir', 'best_val_loss', 'best_val_acc', 'model_name', 'freeze_backbone', 'train_loss', 'args'])
    logging_df.to_csv(f"{models_dir}/logging/experiments_logs.csv", index=False)
    
if __name__ == "__main__":
    '''
    Main function, used to parse the arguments and call the main function
    '''
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-experiment_version', '--experiment_version', type= str, help= 'version of experiment')
    parser.add_argument('-epochs', '--epochs', type= int, help= 'number of epochs', default = 1)
    parser.add_argument('-learning_rate', '--learning_rate', type= float, help= 'value of learning rate', default = 0.0001)
    parser.add_argument('-batch_size', '--batch_size', type= int, help= 'training/validation batch size', default = 64)
    parser.add_argument('-loss', '--loss', type= str, help= 'training/validation loss function(categorical_crossentropy or nllloss)', default='categorical_crossentropy')
    parser.add_argument('-dropout', '--dropout', type= float, help= 'value of the dropout', default=0.5)
    parser.add_argument('-lr_scheduler', '--lr_scheduler', type= bool, help= 'Use learning rate scheduler while training', default=True)
    parser.add_argument('-model_name', '--model_name', type= str, help= 'Name of the model we want to train of evaluate with', default="resnet-50")
    parser.add_argument('-freeze_backbone', '--freeze_backbone', type= bool, help= 'Either freeze the backbone layers or not', default=True)
    parser.add_argument('-use_wandb', '--use_wandb', type= bool, help= 'Track the experiments with wandb', default=False)
    parser.add_argument('-eval_mode', '--eval_mode', type= str, help= 'Evaluation subset of data (train, valid, or test)', default='valid')
    parser.add_argument('-calculate_top_losses', '--calculate_top_losses', type= bool, help= 'Calculate top losses of the training data', default=True)
    parser.add_argument('-script_mode', '--script_mode', type= str, help= 'Either train or setup the data and folder structure', default='train')
    args = parser.parse_args()

    args1 = args
    args1.experiment_version = 1
    if args1.script_mode == 'train':
        main(args1)
    elif args1.script_mode == 'setup':
        setup(args1)