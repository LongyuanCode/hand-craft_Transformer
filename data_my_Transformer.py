import torch
from data_load import load_data
from torch.utils import data

class en2cn_dataset(data.Dataset):
    def __init__(self, X_train, Y_train):
        assert len(X_train) == len(Y_train)
        self.cn = torch.LongTensor(X_train).cuda()
        self.en = torch.LongTensor(Y_train).cuda()
    
    def __getitem__(self, index):
        return self.en[index], self.cn[index]
    
    def __len__(self):
        return len(self.cn)