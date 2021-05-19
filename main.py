from model import new_model
from loader import dataset
from loader import dataloader
from train import train_model
from train import train_model_kfold
import seaborn as sn
import pandas as pd
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
from time import sleep
import math
from matplotlib import pyplot
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torchvision

import plotly.graph_objects as go
from plotly.subplots import make_subplots



def imshow(img):
   img = img / 2 + 0.5     # unnormalize
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()



def train(model_ft,model_name,model_input_size,grafico,classes,batch,epochs_without_learning,kfold,pretrain):

   ##Parametros para o treinomento

   #Seleciona os parametros a serem atualizados.Como estamos utilizando transfer leaning, querendo selecionar apenas o classifer.
   params_to_update = []
   if(pretrain == True):
      print("Apenas última camada marcada para otimização")
      for name,param in model_ft.named_parameters():
         if param.requires_grad == True:
            params_to_update.append(param)
   elif(pretrain == False):
      print("Todos os named parameters marcados para otimização")
      for name,param in model_ft.named_parameters():
         params_to_update.append(param)      

   #optimizer_ft = optim.SGD(params_to_update, lr=0.001,momentum=0.9)
   optimizer_ft = optim.Adam(params_to_update, lr=0.0001)

   #nSamples = [0.5, 1]
   #class_weights = torch.FloatTensor(nSamples)
   #print(class_weights)
   #criterion = nn.CrossEntropyLoss(weight=class_weights)
   criterion = nn.CrossEntropyLoss()
   #Treinamento,Determina se a forma de treinamento sera por meio de KFOLD ou normal com train/val
   if kfold >= 2:
      data_dir = "./data/kfold"
      #Datasets/Dataloaders
      print('Iniciando Treinamento por cross validation KFOLD = {}'.format(kfold))
      print('Carregando dataset em',data_dir)
      image_datasets = dataset(data_dir,model_input_size)
      dataloaders_dict = dataloader(data_dir,image_datasets,1)
      model_ft = train_model_kfold(model_ft, dataloaders_dict, criterion, optimizer_ft,grafico,kfold)
   else:
      data_dir = "./data/normal_trainning"
      #Datasets/Dataloaders
      print('Iniciando Treinamento pelo método normal train/val')
      print('Carregando dataset em',data_dir)
      image_datasets = dataset(data_dir,model_input_size)
      dataloaders_dict = dataloader(data_dir,image_datasets,batch)
      model_ft = train_model(model_ft,model_name, dataloaders_dict, criterion, optimizer_ft,grafico,epochs_without_learning)
   return model_ft

def eval(grafico,model_input_size,model_name,model_ft,inside_trainning):

   #Diretorio do dataset para test
   data_dir = "./data/test"
   #Diretorio de onde o modelo treinado e carregado
   save_dir = "./modelo_treinado/"+model_name+".pth"
   save_dir_confusao = "./modelo_treinado/"+model_name+"-confusao"+".png"
   save_dir_roc = "./modelo_treinado/"+model_name+"-roc"+".png"

   batch = 1
   # Datasets/Dataloaders
   print('Modo de evaluation')
   print('Carregando dataset em',data_dir)
   image_datasets = dataset(data_dir,model_input_size)
   dataloaders_dict = dataloader(data_dir,image_datasets,batch)
     
   #Carrega o modelo salvo em disco apenas se nao estiver dentro do loop de train e estiver apenas em evaluation
   if not inside_trainning:
      model_ft.load_state_dict(torch.load(save_dir,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
   #Seta o modelo para evaluation
   model_ft.eval()



   print('Calculando a taxa de acerto para ',data_dir)
   x = []
   y = []
   true_positive = 0
   false_positive = 0
   true_negative = 0
   false_negative = 0
   correct = 0
   total = 0
   images_processed = 0
   true_labels = []
   roc_probabilities = []


   #n lembro
   with tqdm(total=math.ceil(len(dataloaders_dict['val'].dataset))) as pbar:
      with torch.no_grad():
         since = time.time()
         for data in dataloaders_dict['val']:
            images_processed+=1
            pbar.update(1)
            images, labels = data
            images_print = images
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            x.append(total)

            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(outputs)
            
            #Benigna = 0
            #Maligna = 1
            if(torch.cuda.is_available()):
               if(labels.cpu().numpy()[0] == 0 and predicted.cpu().numpy()[0] == 0):
                  true_negative = true_negative + 1
               elif(labels.cpu().numpy()[0] == 1 and predicted.cpu().numpy()[0] == 1):
                  true_positive = true_positive + 1
               elif(labels.cpu().numpy()[0] == 0 and predicted.cpu().numpy()[0] == 1):
                  false_positive = false_positive + 1        
               elif(labels.cpu().numpy()[0] == 1 and predicted.cpu().numpy()[0] == 0):
                  false_negative = false_negative + 1    
            else:
               if(labels.numpy()[0] == 0 and predicted.numpy()[0] == 0):
                  true_negative = true_negative + 1
               elif(labels.numpy()[0] == 1 and predicted.numpy()[0] == 1):
                  true_positive = true_positive + 1
               elif(labels.numpy()[0] == 0 and predicted.numpy()[0] == 1):
                  false_positive = false_positive + 1        
               elif(labels.numpy()[0] == 1 and predicted.numpy()[0] == 0):
                  false_negative = false_negative + 1    
            
            #if not((predicted == labels).sum().item()):
            #   print('Imagem classificada incorretamente:')
            #   print("Probabilidade BENIGNA : %.2f%% Probabilidade MALIGNA : %.2f%%" % (probabilities.data[0][0].item()*100,probabilities.data[0][1].item()*100) ) #Converted to probabilities
               #imshow(torchvision.utils.make_grid(images_print))
            roc_probabilities.append(probabilities.data[0][1].item())
            true_labels.append(labels)
            correct += (predicted == labels).sum().item()
            y.append(correct/total)
            #print('Imagens Processadas : {}'.format(images_processed))
   time_elapsed = time.time() - since
   roc_probabilities = np.array(roc_probabilities)
   true_labels = np.array(true_labels)

   x1, y1 = [0, 1], [0, 1]
   pyplot.plot(x1, y1,linestyle='--')
   fpr, tpr, thresholds = metrics.roc_curve(true_labels, roc_probabilities)
   auc = metrics.roc_auc_score(true_labels, roc_probabilities)
   pyplot.plot(fpr, tpr, marker='.', label='ResNet AUC = '+str(auc))
   pyplot.xlabel('1 - especificidade')
   pyplot.ylabel('Sensibilidade')
   pyplot.legend(loc=4)
   pyplot.savefig(save_dir_roc)



   print("True Positive (Imagem Maligna classificada como maligna)= %d\nTrue Negative (Imagen Benigna classificada como benigna) = %d\nFalse Positive (Imagem Benigna classificada como maligna) = %d\nFalse Negative(Imagen Maligna classificada como benigna) = %d\n" % (true_positive,true_negative,false_positive,false_negative))
   array = [[true_positive,false_negative],[false_positive,true_negative]]
   df_cm = pd.DataFrame(array, index = [i for i in ["Maligna","Benigna"]],
                     columns = ["Maligna","Benigna"])

   plt.figure(figsize = (10,7))
   plt.title("Matriz de Confusão", fontsize=25)
   sn.set(font_scale=1.4)
   sn.heatmap(df_cm, annot=True,cmap="Blues", fmt='d')
   plt.xlabel('Classe Predita',fontsize=20)
   plt.ylabel('Classe Real',fontsize=20)
   plt.tick_params(axis='both', which='major', labelsize=18, labelbottom = True, bottom=False, top = False, labeltop=False)
   plt.savefig(save_dir_confusao)
   
   save_dir_metricas="./modelo_treinado/"+model_name+"-metricas"+".txt"
   with open(save_dir_metricas, 'w') as filetowrite:
      quantidade_de_imagens = true_positive+true_negative+false_positive+false_negative
      acertos = true_positive+true_negative
      erros = false_positive+false_negative
      filetowrite.write("True Positive (Imagem Maligna classificada como maligna)= %d\nTrue Negative (Imagen Benigna classificada como benigna) = %d\nFalse Positive (Imagem Benigna classificada como maligna) = %d\nFalse Negative(Imagen Maligna classificada como benigna) = %d\n\n" % (true_positive,true_negative,false_positive,false_negative)) 
      filetowrite.write("Quantidade de Imagens Analisadas = %d\n" % (quantidade_de_imagens)) 
      filetowrite.write("Quantidade de Imagens Classificadas Corretamente = %d\n" % (acertos)) 
      filetowrite.write("Quantidade de Imagens Classificadas Incorretamente = %d\n\n" % (erros))

      acuracia = acertos/quantidade_de_imagens
      if(true_negative+false_positive == 0):
         especificidade = 0
      else:
         especificidade = true_negative/(true_negative+false_positive)
      
      if(true_positive+false_negative == 0):
         sensibilidade = 0
      else:  
         sensibilidade = true_positive/(true_positive+false_negative)
      
      if(true_positive+false_positive == 0):
         precisao = 0
      else:
         precisao = true_positive/(true_positive+false_positive)
      if(precisao + sensibilidade == 0):
         f1measure = 0
      else:
         f1measure = 2 * ((precisao * sensibilidade) / (precisao + sensibilidade))

      filetowrite.write("Acurácia (Taxa de acertos das amostras totais)=  %.3f\n" % (acuracia))  
      filetowrite.write("Especificidade (Representa a probabilidade do classificador identificar corretamente a classe benigna) = %.3f\n" % (especificidade)) 
      filetowrite.write("Sensibilidade (Representa a probabilidade do classificador identificar corretamente a classe maligna)= %.3f\n" % (sensibilidade))  
      filetowrite.write("Precisão (De todas as amostras classificadas como maligna, quantas são realmente malignas)= %.3f\n" % (precisao)) 
      filetowrite.write("F1-Measure = %.3f\n" % (f1measure))
      filetowrite.write('Evaluation completo em %f' % time_elapsed)
     

      
   grafico.add_trace(go.Scatter(x=x, y=y, name='Conjunto de Teste', line = dict(color='green', width=2)),row=3,col=1)  
   print('Taxa de acerto: %d %%' % (100 * correct / total))

#Possible values for mode: train_eval,eval,kfold
def statistics_plot(model_name,mode):
   if mode == 'train_eval':
      grafico = make_subplots(rows=3, cols=1, subplot_titles=("Curva de Aprendizado", "Curva de Acurácia","Acurácia no Conjunto de Teste"))
      grafico.update_xaxes(title_text="ÉPOCA", row=1, col=1)    
      grafico.update_yaxes(title_text="PERDA", row=1, col=1)  
      grafico.update_xaxes(title_text="ÉPOCA", row=2, col=1)    
      grafico.update_yaxes(title_text="ACC", row=2, col=1)  
      grafico.update_xaxes(title_text="Número de Amostras", row=3, col=1)    
      grafico.update_yaxes(title_text="ACC", row=3, col=1)  
      grafico.update_layout(title_text="Estatísticas "+model_name, height=800) 
   elif mode == 'eval':
      grafico = make_subplots(rows=3, cols=1, subplot_titles=("Acurácia no Conjunto de Teste"))
      grafico.update_xaxes(title_text="Número de Amostras", row=1, col=1)    
      grafico.update_yaxes(title_text="ACC", row=3, col=1)  
      grafico.update_layout(title_text="Estatísticas "+model_name, height=800) 
   elif mode == 'kfold':
      grafico = make_subplots(rows=3, cols=1, subplot_titles=("","Acurácia K-FOLD",''))
      grafico.update_xaxes(title_text="K", row=2, col=1)    
      grafico.update_yaxes(title_text="ACC", row=2, col=1)  
      # grafico.update_xaxes(title_text="Number of Samples", row=3, col=1)    
      # grafico.update_yaxes(title_text="ACC", row=3, col=1)  
      grafico.update_layout(title_text="Estatísticas "+model_name, height=800) 



   return grafico


def computate_train_and_eval(classes,batch,epochs_without_learning,model_name,kfold,pretrain):

      # Cria a arquitetura densene1t pre-treinada
      print('Inicializando a rede...')
      print('Estrutura da {}: '.format(model_name))
      model_ft, model_input_size= new_model(classes,model_name,pretrain)
      print(model_ft)
      print("Pré-Treinada = %s" % pretrain)

      #Inicializa objeto para criacao dos graficos
      if(kfold < 2):
         grafico = statistics_plot(model_name,'train_eval')
      else:
          grafico = statistics_plot(model_name,'kfold')

      model_ft = train(model_ft,model_name,model_input_size,grafico,classes,batch,epochs_without_learning,kfold,pretrain)
      #Testa a rede sobre o dataset de test
      eval(grafico,model_input_size,model_name,model_ft,inside_trainning=True)
      save_dir = "./modelo_treinado/"+model_name+"-best"+".pth"
      save_dir_grafico = "./estatisticas/"+model_name+".html"
      #Salva o modelo treinado em disco
      print('Salvando modelo em ',save_dir)
      torch.save(model_ft.state_dict(), save_dir)
      print('Modelo salvo com sucesso')
      #Exibe o grafico no navegador
      grafico.show()
      grafico.write_html(save_dir_grafico)


def computate_eval_only(classes,batch,epochs_without_learning,model_name,kfold,pretrained):
   save_dir_grafico = "./modelo_treinado/"+model_name+"eval-only"+".html"
   model_ft, model_input_size= new_model(classes,model_name,pretrained)
   grafico = statistics_plot(model_name,'eval')
   eval(grafico,model_input_size,model_name,model_ft,inside_trainning=False)
   grafico.show()
   grafico.write_html(save_dir_grafico)


def run():
   torch.multiprocessing.freeze_support()
   classes = 2
   batch =  32
   epochs_without_learning = 5
   pretrain = True
   model_name='resnet18'
   #IF k fold < 2 the trainning will be done with train/val method,otherwise with kfold validation
   kfold = 1

   ##Train the network using kfold or train/eval,then, test over the test_dataset
   computate_train_and_eval(classes,batch,epochs_without_learning,model_name,kfold,pretrain)

   ##Only eval over test_dateset
   #computate_eval_only(classes,batch,epochs_without_learning,model_name,kfold,pretrain)

if __name__ == '__main__':
   run()


   

