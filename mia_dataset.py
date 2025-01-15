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

class TemporalArxiv(MIADataset):
    def __init__(self):
        super().__init__("TemporalArixivMIA")
        self._init_dataset()
    def _init_dataset(self):
        self.dataset = load_dataset("iamgroot42/mimir", "temporal_arxiv")
        self.member = TextDataset(self.dataset["member"])
        self.non_member = TextDataset(self.dataset["non_member"])

class TemporalWiki(MIADataset):
    def __init__(self):
        super().__init__("TemporalWikiMIA")
        self._init_dataset()
    def _init_dataset(self):
        self.dataset = load_dataset("iamgroot42/mimir", "temporal_wiki")
        self.member = TextDataset(self.dataset["member"])
        self.non_member = TextDataset(self.dataset["non_member"])

class MIMIR(MIADataset):
    def __init__(self, domain="all"):
        super().__init__("MIMIR")
        self._init_dataset()
        self.domain = domain
    def _init_dataset(self):
        if self.domain == "all":
            self.dataset = {}
            for domain in ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                           "pubmed_central", "wikipedia_(en)", "full_pile", "c4"]:
                self.dataset[domain] = load_dataset("iamgroot42/mimir", domain, split = "ngram_13_0.8")
                self.member = {}
                self.non_member = {}
                for domain in self.dataset:
                    self.member[domain] = TextDataset(self.dataset[domain]["member"])
                    self.non_member[domain] = TextDataset(self.dataset[domain]["non_member"])
        else:
            self.dataset = {}
            self.dataset[self.domain] = load_dataset("iamgroot42/mimir", self.domain, split="ngram_13_0.8" )
            self.member = {}
            self.member[self.domain] = TextDataset(self.dataset[self.domain]["member"])
            self.non_member = {}
            self.non_member[self.domain] = TextDataset(self.dataset[self.domain]["non_member"])