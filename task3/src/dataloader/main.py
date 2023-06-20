from torch.utils.data import DataLoader
from .intracardiac import IntracardiacDataset

def get_loaders():
    train_dataset = IntracardiacDataset(train=True)
    test_dataset = IntracardiacDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return {
        'train': train_loader,
        'test': test_loader
    }
    