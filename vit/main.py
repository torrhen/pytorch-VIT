import os
import torch
from models import ViT
from engine import train
from pathlib import Path
from torchmetrics import Accuracy
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from viz import plot_loss, plot_accuracy
from config import data_params, model_params, train_params


class Preprocessing(torch.nn.Module):
  '''
  Normalize image and convert to tensor before passing to model. No Augmentation.
  '''
  def __init__(self):
    super().__init__()
    self.transforms = torch.nn.Sequential(
      v2.ToTensor(),
      # correct values taken from https://github.com/kuangliu/pytorch-cifar/issues/19
      v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    )
  
  def forward(self, x):
    return self.transforms(x)


if __name__ == "__main__":

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)

  TRAIN_DATA_FOLDER = Path(os.getcwd()) / Path(R"data/train")
  TRAIN_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

  TEST_DATA_FOLDER = Path(os.getcwd()) / Path(R"data/test")
  TEST_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

  transforms = Preprocessing()

  # train and test ViT on CIFAR10 dataset
  train_data = CIFAR10(root=TRAIN_DATA_FOLDER, transform=transforms, train=True, download=True)
  test_data = CIFAR10(root=TEST_DATA_FOLDER, transform=transforms, train=True, download=True)
  train_loader = DataLoader(train_data, batch_size=train_params['batch_size'], shuffle=True, drop_last=False)
  test_loader = DataLoader(test_data, batch_size=train_params['batch_size'], shuffle=True, drop_last=False)

  model = ViT(
    in_channels=data_params['in_channels'],
    num_classes=data_params['num_classes'],
    image_size=data_params['image_size'],
    patch_size=data_params['patch_size'],
    embedding_dim=data_params['embedding_dim'],
    n_layers=model_params['n_layers'],
    n_heads=model_params['n_heads'],
    attn_dropout=model_params['attn_dropout'],
    hidden_dim=model_params['hidden_dim'],
    dropout=model_params['dropout']
  ).to(device)

  optimizer = torch.optim.SGD(params=model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'])
  loss_fn = torch.nn.CrossEntropyLoss()
  accuracy_fn = Accuracy(task='multiclass', num_classes=data_params['num_classes']).to(device)

  results = train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
    epochs=train_params['epochs']
  )

  plot_loss(results)
  plot_accuracy(results)




  

