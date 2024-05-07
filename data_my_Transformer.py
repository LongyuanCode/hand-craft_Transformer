import torch
from data_load import load_data
from torch.utils import data

class en2cn_dataset(data.Dataset):
    def __init__(self, X_train, cn_valid_lens, Y_train, en_valid_lens):
        assert len(X_train) == len(Y_train)
        self.cn = torch.LongTensor(X_train)
        self.cn_valid_lens = torch.tensor(cn_valid_lens)
        self.en = torch.LongTensor(Y_train)
        self.en_valid_lens = torch.tensor(en_valid_lens)
    
    def __getitem__(self, index):
        return self.en[index], self.en_valid_lens[index],\
            self.cn[index], self.cn_valid_lens[index]
    
    def __len__(self):
        return len(self.cn)