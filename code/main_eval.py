from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from my_models import *
from my_utils.dataset import load_dataset
from my_utils.helper_fns import print_size_of_model, print_no_of_parameter
import numpy as np
import torch
from torchvision import transforms
from emergencyNet import ACFFModel


model_name = "emergencyNet"


def eval(model, dataloaders,):
    y_pred = []
    y_true = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # iterate over test data
    model.eval()
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    conf_mtrx_filepath = '../results/' + model_name + 'confusionMatrix.png'
    plt.savefig(conf_mtrx_filepath)
    plt.clf()
    print(classification_report(y_true, y_pred, target_names=classes))


def main():
    INPUT_SIZE = 224
    data_dir = '../../dataset/AIDER/'

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
    dataloaders = load_dataset(data_dir, data_transforms)

    # model = select_model(model_name, classes=5)
    model = ACFFModel(5)
    saved_state_dict = torch.load('../results/emergencyNet_best.pth')
    model.load_state_dict(saved_state_dict['state_dict'])

    # print(model)
    print_size_of_model(model)
    print_no_of_parameter(model)

    eval(model, dataloaders)


if __name__ == '__main__':
    main()

