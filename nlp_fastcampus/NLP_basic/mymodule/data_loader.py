import os

import torch
from torch.utils.data import DataLoader, Dataset

def get_loader(x, y,
               dataset,
               batch_size=256,
               is_train = True,
               train_ratio=.8, 
               valid_ratio=.2,):
        
    assert (train_ratio + valid_ratio) <= 1
    
    if is_train : 
        train_cnt = int(x.size(0) * train_ratio)
        valid_cnt = x.size(0) - train_cnt

        indices = torch.randperm(x.size(0))

        train_x, valid_x = torch.index_select(
            x,
            dim=0,
            index=indices
        ).split([train_cnt, valid_cnt])
        
        train_y, valid_y = torch.index_select(
            y,
            dim=0,
            index=indices
        ).split([train_cnt, valid_cnt])

        train_loader = DataLoader(
            dataset=dataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            dataset=dataset(valid_x, valid_y),
            batch_size=batch_size,
            shuffle=True
        )
        
        return train_loader, valid_loader

    else :
        test_loader = DataLoader(
            dataset=dataset(x, y),
            batch_size=batch_size,
            shuffle=False
        )
        
        return test_loader