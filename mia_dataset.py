from torch.utils.data import Dataset
from datasets import load_dataset



class MIADataset(Dataset):
    def __init__(self, name):
        self.name = name

    def _init_dataset(self):
        pass

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class WikiMIA(MIADataset):
    def __init__(self, name, length=64):
        super().__init__(name)
        self.length = length
        self._init_dataset()
    def _init_dataset(self):
        self.dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{self.length}")
        self.member = []
        self.non_member = []
        for i in range(len(self.dataset)):
            if self.dataset[i]['label'] == 1:
                self.non_member.append(self.dataset[i]['input'])
            else:
                self.member.append(self.dataset[i]['input'])
        self.member = TextDataset(self.member)
        self.non_member = TextDataset(self.non_member)

class TemporalArixivMIA(MIADataset):
    def __init__(self):
        super().__init__("TemporalArixivMIA")
        self._init_dataset()
    def _init_dataset(self):
        pass
