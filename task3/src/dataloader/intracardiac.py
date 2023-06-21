import torch
from torch.utils.data import Dataset
from .tools import *

class IntracardiacDataset(Dataset):
    def __init__(self, DIR='data/intracardiac_dataset/', train=True):
        super().__init__()

        data_dirs = []
        regex = r'data_hearts_dd_0p2*'
        for x in os.listdir(DIR):
            if re.match(regex, x):
                data_dirs.append(DIR + x)
        self.file_pairs = read_data_dirs(data_dirs)
        self.file_pairs.sort(key=lambda x: x[0])
        if train:
            self.file_pairs = self.file_pairs[:int(len(self.file_pairs)*0.95)]
        else:
            self.file_pairs = self.file_pairs[int(len(self.file_pairs)*0.95):]
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        pECGData = np.load(self.file_pairs[idx][0])
        pECGData = get_standard_leads(pECGData)
        pECGData = (pECGData - pECGData.min(axis=0)) / (pECGData.max(axis=0) - pECGData.min(axis=0)) # normalize 
        pECGData = pECGData.transpose(1, 0)
        pECGData = torch.from_numpy(pECGData).float()
        VmData = np.load(self.file_pairs[idx][1])
        ActTime = get_activation_time(VmData)
        ActTime = torch.from_numpy(ActTime).float().squeeze()
        return pECGData, ActTime