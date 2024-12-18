from __future__ import print_function
from __future__ import division
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import shutil
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from my_utils.dataset import load_dataset
from my_utils.helper_fns import AverageMeter, accuracy, ProgressMeter
from my_models import *
# from emergencyNet import ACFFModel
from emergencyNet2 import ACFFModel

import time
import gc


num_classes = 5
batch_size = 32
num_epochs = 150     # Number of epochs to train for
INPUT_SIZE = 224   # 2
data_dir = '../../dataset/AIDER/'

""" Implemented models [ MobileNet_v2, MobileNet_v3, SqeezeNet1_0, VGG16, ShuffleNet_v2 <--
    EfficientNet_B0  ResNet50
"""
model_name = "emergencyNetv12"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
best_acc1 = 0


def train_model(model, train_dataloader, val_dataloader, criterion, device, optimizer, scheduler, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()  # Update the learning rate
        # scheduler.step(val_loss)  # in the case of ReduceLROnPlateau

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30 == 0:
            progress.display(i)
        gc.collect()


def evaluate(model, dataloader, criterion, device):
    """
    This function is similar to the validate function
    Can use it to simply get the validation losses and acc during training
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, labels in dataloader:
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            correct = int(predicted == labels)
            total_correct += correct
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 30 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    # model.module.show_params()

    return top1.avg


def save_checkpoint(state, is_best, path, checkpoint_file='../results/checkpoints.pth.tar'):
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, path)


def main():
    global best_acc1
    # LOAD DATA
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create dataloaders "train" and "val"
    train_loader, val_loader, my_datasets = load_dataset(data_dir, data_transforms, batch_size)

    # Initialize the model for this run
    # model = initialize_model(num_classes, feature_extract, use_pretrained=True)
    # model = select_model(model_name, classes=5)
    model = ACFFModel(num_classes)

    # Print the model we just instantiated
    # print(model)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be updated in this run. If we are finetuning we will be updating all parameters.
    # However, if we are doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)

    steps_per_epoch = len(train_loader)
    # Define the cosine annealing learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch, eta_min=0.001)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    class_weights = [1.0, 1.0, 1., 0.35, 1.]
    weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    model_filepath = '../results/' + model_name + '_best.pth'
    # Train and evaluate loop
    val_acc_history = []
    for epoch in range(num_epochs):

        train_model(model, train_loader, val_loader, criterion, device, optimizer, scheduler, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print('best_acc:' + str(best_acc1))
        val_acc_history.append(acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_name,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, path=model_filepath)

    # Plot the training curves of validation acc vs epoch
    if isinstance(val_acc_history, list) and len(val_acc_history) > 0:
        val_hist = [h.cpu().numpy() if isinstance(h, torch.Tensor) else h for h in val_acc_history]
        plt.title("Validation Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1, num_epochs + 1), val_hist, label=model_name)
        plt.ylim((0, 1.))
        plt.xticks(np.arange(1, num_epochs + 1, 1.0))
        plt.legend()
        train_plot_filepath = '../results/' + model_name + 'trainingPlot.png'
        plt.savefig(train_plot_filepath)
    else:
        print("Error: Can not plot: val_acc_history is not a list or is empty.")


if __name__ == '__main__':
    main()





