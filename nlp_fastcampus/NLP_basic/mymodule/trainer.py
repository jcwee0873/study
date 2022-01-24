from copy import deepcopy

import numpy as np

import torch
from torch import nn
from torch import optim
from tqdm import tqdm

class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    
    def _train(self, batch_item):
        self.model.train()
        
        x = batch_item[0].to(self.device)
        y = batch_item[1].to(self.device)
        
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        
        return float(loss)
    
    def _validate(self, batch_item):
        self.model.eval()
        
        x = batch_item[0].to(self.device)
        y = batch_item[1].to(self.device)
        
        val_loss = 0
        with torch.no_grad():
            y_hat = self.model(x)
            val_loss = self.crit(y_hat, y)
        
        return float(val_loss)
            
    
    def train(self, train_loader, valid_loader, epochs):
        best_loss = np.inf
        best_model = None
        best_epoch = np.inf
        
        for i in range(epochs):
            total_loss, total_val_loss = 0, 0
            
            tqdm_dataset = tqdm(enumerate(train_loader))
            for batch, batch_item in tqdm_dataset:
                batch_loss = self._train(batch_item)
                total_loss += batch_loss
                
                tqdm_dataset.set_postfix({
                    'Epoch': i + 1,
                    'Loss': '%.6f' % batch_loss,
                })
            
            tqdm_dataset = tqdm(enumerate(valid_loader))
            for batch, batch_item in tqdm_dataset:
                batch_loss = self._validate(batch_item)
                total_val_loss += batch_loss
                
                tqdm_dataset.set_postfix({
                    'Epoch': i + 1,
                    'Val Loss': '%.6f' % batch_loss,
                })
            
            total_val_loss = total_val_loss / (batch + 1)
            if total_val_loss <= best_loss:
                best_loss = total_val_loss
                best_epoch = i
                best_model = deepcopy(self.model.state_dict())
         
        print('Best: Epoch= %d  val_loss= %.6f' % (best_epoch + 1, best_loss))
        
        self.model.load_state_dict(best_model)