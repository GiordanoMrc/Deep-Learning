from torchvision import models
import torch
import torch.nn as nn
import sys

def set_parameter_requires_grad(model,pretrained):
     if(pretrained == True):
          for param in model.parameters():
               param.requires_grad = False
     elif(pretrained == False):
          for param in model.parameters():
               param.requires_grad = True

def new_model(classes,model_name,pretrained):
   if model_name == 'densenet':
        model_ft = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, classes)
        input_size = 224
   elif model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, classes)
        input_size = 224
   elif model_name == 'googlenet':
        model_ft = models.googlenet(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, classes)
        input_size = 224   
   elif model_name == 'shufflenet':
        model_ft = models.googlenet(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, classes)
        input_size = 224       
   elif model_name == 'resnext50':
        model_ft = models.resnext50_32x4d(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, classes)
        input_size = 224       
       
   elif model_name == 'alexnet':
        model_ft = models.alexnet(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,classes)
        input_size = 224
   elif model_name == 'vgg11':
        model_ft = models.vgg11_bn(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,classes)
        input_size = 224
   elif model_name == 'squeezenet':  
        model_ft = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model_ft,pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = classes
        input_size = 224        
   else:
      print('Modelo não encontrado.Selecione um modelo existente')
      sys.exit(1)

   # Verifica se existe GPU disponivel
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # Caso exista placa de video, o modelo e enviado para a gpu
   model_ft = model_ft.to(device)
   return model_ft, input_size

