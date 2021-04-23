import torch
import time
import copy
import plotly.graph_objects as go
from sklearn.model_selection import KFold
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from time import sleep
import math
def imshow(img):
   img = img / 2 + 0.5     # unnormalize
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

def train_model(model,model_name, dataloaders, criterion, optimizer, grafico, epochs_without_learning):
    x = []
    y_train_loss = []
    y_val_loss = []
    y_train_acc = []
    y_val_acc = []
    not_learn = 0
    current_epoch = 0
    max_epoch = 15
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    lowest_loss = 1000
    best_epoch = 1
    print('-' * 10)
    while(not_learn < epochs_without_learning):
        if(current_epoch >= max_epoch):
            break
        current_epoch += 1

        print('Current Epoch: {}'.format(current_epoch))
        print('Lowest loss so far is {:4f} in epoch {}'.format(
            lowest_loss, best_epoch))
        print('Epochs without learning: {}/{}'.format(not_learn,
                                                      epochs_without_learning))
        x.append(current_epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            with tqdm(total=math.ceil(len(dataloaders[phase].dataset)/32)) as pbar:
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    pbar.update(1)
                    device = torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu")
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):

                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':

                y_train_acc.append(epoch_acc.cpu())
                y_train_loss.append(epoch_loss)
            else:
                y_val_acc.append(epoch_acc.cpu())
                y_val_loss.append(epoch_loss)

            # Copia o modelo de melhor resultado
            if phase == 'val' and epoch_loss < lowest_loss:
                not_learn = 0
                best_epoch = current_epoch
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Melhor modelo encontrado.Atualizando rede',
                      lowest_loss, "->", epoch_loss)
            elif phase == 'val' and epoch_loss >= lowest_loss:
                not_learn += 1
        print('-' * 10)
        save_dir = "./modelo_treinado/"+model_name+"-epoca-"+str(current_epoch)+".pth"
        #Salva o modelo treinado para a epoca atual
        print('Salvando modelo em ',save_dir)
        torch.save(model.state_dict(), save_dir)
        print('Modelo salvo com sucesso')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest val Loss: {:4f}'.format(lowest_loss))

    grafico.add_trace(go.Scatter(x=x, y=y_train_loss, name='Conjunto de Treinamento',
                                 line=dict(color='firebrick', width=2)), row=1, col=1)
    grafico.add_trace(go.Scatter(x=x, y=y_val_loss, name='Conjunto de Validação',
                                 line=dict(color='royalblue', width=2)), row=1, col=1)

    grafico.add_trace(go.Scatter(x=x, y=y_train_acc, name='Conjunto de Treinamento', showlegend=False,
                                 line=dict(color='firebrick', width=2)), row=2, col=1)
    grafico.add_trace(go.Scatter(x=x, y=y_val_acc, name='Conjunto de Validação', showlegend=False,
                                 line=dict(color='royalblue', width=2)), row=2, col=1)

    # Carrega os pesos do epoch com maior acuracia para o val
    model.load_state_dict(best_model_wts)
    save_dir_trainning_info="./modelo_treinado/"+model_name+"-loss-e-acc"+".txt"
    with open(save_dir_trainning_info, 'w') as filetowrite:
        filetowrite.write("Quantidade de epocas = " + str(current_epoch)+" Melhor época = "+ str(best_epoch))
        
        filetowrite.write("\n\nÉpocas,Trainning Loss,Validation Loss\n")
        for i in range(len(x)):
            filetowrite.writelines("%x,%.3f,%.3f\n" % (x[i],y_train_loss[i],y_val_loss[i]))

        filetowrite.write("\nÉpocas,Trainning Acc,Validation Acc\n")
        for i in range(len(x)):
            filetowrite.writelines("%x,%.3f,%.3f\n" % (x[i],y_train_acc[i],y_val_acc[i]))
        filetowrite.write('Treinamento completo em %f' % time_elapsed)
    


    return model


def train_model_kfold(model_ft, dataloaders_dict, criterion, optimizer_ft, grafico, kfold):

    list_of_fold = []
    data = []
    mean = 0
    x= []
    y_accuracy = []
    folds_mean = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for image, label in dataloaders_dict['train']:
        data.append([image, label])


    current_fold = 0
    best_acc = 0.0
    best_fold = 1
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    kf = KFold(n_splits=kfold,shuffle=True)

    for train, test in kf.split(data):
        #print(train, test)
        current_fold += 1
        x.append(current_fold)
        list_of_fold.append(current_fold)
        print('Current Fold: {}'.format(current_fold))


        running_loss = 0.0
        running_corrects = 0

        for idx_train in train:
            model_ft.train()
            image, label = data[idx_train]
            # imshow(torchvision.utils.make_grid(image))
            image = image.to(device)
            label = label.to(device)


            optimizer_ft.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model_ft(image)
                loss = criterion(outputs, label)

                _, preds = torch.max(outputs, 1)

                # backward + optimize
                loss.backward()
                optimizer_ft.step()

                # statistics
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)
        ##print(running_loss)   
        fold_loss = running_loss / len(train)
        fold_acc = running_corrects.double(
        ) / len(train)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(
              fold_loss, fold_acc))

        running_loss = 0.0
        running_corrects = 0

        for idx_test in test:
            model_ft.eval()
            image, label = data[idx_test]
  


            image = image.to(device)
            label = label.to(device)

            optimizer_ft.zero_grad()

            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = model_ft(image)
                loss = criterion(outputs, label)

                _, preds = torch.max(outputs, 1)

                # statistics
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)
        fold_loss = running_loss / len(test)
        fold_acc = running_corrects.double(
        ) / len(test)
        y_accuracy.append(fold_acc.cpu())
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(
              fold_loss, fold_acc))
        mean+=fold_acc
        folds_mean.append(mean.cpu() /current_fold )
    mean = mean / kfold
    print('Acurácia média obtida para os {} folds : {}'.format(kfold,mean))
    
    grafico.add_trace(go.Scatter(x=x, y=y_accuracy, name='Ki accuracy,i>0', showlegend=True,
                                 line=dict(color='purple', width=2)), row=2, col=1)
    grafico.add_trace(go.Scatter(x=x, y=folds_mean, name='Mean', showlegend=True,
                                 line=dict(color='red', width=2)), row=2, col=1)                                 
                                 

    return model_ft
