import gc
import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, current_valid_acc,
        epoch, model, optimizer, criterion, save_dir, version_name
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss for epoch: {epoch+1}: {self.best_valid_loss} - Best validation acc: {current_valid_acc}")
            torch.save({
                'time': str(datetime.now()),
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'accuracy': current_valid_acc
                }, f'{save_dir}/{version_name}/best_model.pth')
            return self.best_valid_loss, current_valid_acc


def train(model, trainloader, optimizer, criterion, device):
    '''
    Train the model, and return the training loss and accuracy, and the time taken for training, in seconds.
    '''
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()
    for i, (data, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        optimizer.zero_grad()
        # ⭐️⭐️ Automatic Tensor Casting
        with torch.cuda.amp.autocast():
            # forward pass
            data, labels = data.to(device), target.to(device)
            outputs = model(data)
            # calculate the loss
            loss = criterion(outputs, labels)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(loss).backward() # ⭐️⭐️ Automatic Gradient Scaling
        
        train_running_loss += loss.item()
        
        # update the optimizer parameters
        # Unscales gradients and calls
        # or skips optimizer.step()
        scaler.step(optimizer)
        # Updates the scale for next iteration
        scaler.update()
        # ⭐️⭐️ Garbage Collection
        torch.cuda.empty_cache()
        _ = gc.collect()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    '''
    Validate the model, and return the validation loss and accuracy, and the time taken for validation, in seconds.
    '''
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            # forward pass
            data, labels = data.to(device), target.to(device)
            outputs = model(data)
            
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


def save_model(epochs, model, optimizer, criterion, save_dir, version_name):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'time': str(datetime.now()),
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{save_dir}/{version_name}/final_model__{str(datetime.now())}.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss, save_dir, version_name):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_dir}/{version_name}/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/{version_name}/loss.png')