from Train_Main import initialize_model
from dataloader import HemorrhageLoader, transform_func
import torch

from sklearn.metrics import classification_report

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):
    
    model.eval()   # Set model to evaluate mode
    running_corrects = 0

    np_preds = np.empty([0])
    np_labels = np.empty([0])

    for i_batch, sampled_batch in enumerate(dataloader):
        inputs = sampled_batch['img']
        labels = sampled_batch['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
        np_preds = np.append(np_preds, preds.to('cpu').numpy())
        np_labels = np.append(np_labels, labels.to('cpu').numpy())

                
    acc = running_corrects.double() / len(dataloader.dataset)

    print('Evaluation Acc: {:.4f}'.format(acc))

    return np_preds, np_labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate_main(data_dir, 
                 model_name, 
                 num_classes, 
                 batch_size, 
                 load_model_path):    
    
    # Initialize the model for this evaluation
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)

    # load the model    
    model_ft.load_state_dict(torch.load(load_model_path))

    print("Initializing Datasets and Dataloaders...")    
    # Create evalutation datasets
    image_dataset = HemorrhageLoader(data_dir, 'val', transform_func())
    # Create evalutation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4)

    # use GPU
    device = torch.device("cuda")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # evaluate model
    preds, labels = evaluate_model(model_ft, dataloader, device)

    # plot confusion matrix
    if num_classes == 6:
        target_names = [ 'epidural', 'healthy', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    else:
        target_names = [ 'epidural', 'healthy', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    print(classification_report(labels, preds, target_names=target_names))
    print ("**************************************************************")

    plt.figure()
    cnf_matrix = confusion_matrix(labels, preds)
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                        title='normalized confusion matrix')

    plt.show()


if __name__ == "__main__":
    #   to the ImageFolder structure
    data_dir = "..\\Train_png_all\\photo\\"
    # Models to choose from [resnet18, resnet50]
    model_name = "resnet18"   
    # number of output classes in the model
    num_classes = 6
    # Batch size for training (change depending on how much memory you have)
    batch_size = 10
    # load the model from "load_model_path"
    load_model_path = "..\\log\\" + model_name + "_001" +".model"

    print(load_model_path)
    evaluate_main(data_dir, model_name, num_classes, batch_size, load_model_path)