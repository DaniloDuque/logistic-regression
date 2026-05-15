from torch.utils.data import Dataset as TorchDataset


class TextDataset(TorchDataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i]
