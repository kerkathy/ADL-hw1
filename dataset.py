from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    # read-only, can't modify
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # collate the input samples into a batch for yielding from the data loader iterator
        # batch["text"] -> tensor(batch_size, seq_len, embed_size(input_size))
        # batch["intent"] -> tensor(batch_size, 1)

        batch = dict()
        batch["id"] = [sample["id"] for sample in samples]
        batch["text"] = [sample["text"].split() for sample in samples]

        # encode_batch: List[List[str]] -> List[List[int]]
        # 1. pads each sequence to the max length of the batch 
        # 2. converts text to ids
        batch["text"] = self.vocab.encode_batch(batch["text"])

        # convert to tensor
        batch["text"] = torch.tensor(batch["text"])

        # for train/val split
        # padding using smallest unused index (i.e., num_classes) to max length within this batch
        # or else, there would be dim error when creating model 
        if "intent" in samples[0]:
            batch["intent"] = [self.label2idx(sample["intent"]) for sample in samples]
            batch["intent"] = torch.tensor(batch["intent"])
            # assert type(batch["intent"][0]) == int
            
        return batch
        # raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    # read-only, can't modify
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # collate the input samples into a batch for yielding from the data loader iterator
        # batch["tokens"] -> tensor(batch_size, seq_len, embed_size(input_size))
        # batch["tags"] -> tensor(batch_size, 1)

        batch = dict()
        batch["id"] = [sample["id"] for sample in samples] # List[str]
        batch["tokens"] = [sample["tokens"] for sample in samples] # List[List[str]]

        # encode_batch: List[List[str]] -> List[List[int]]
        # 1. pads each sequence to the max length of the batch 
        # 2. converts tokens to ids
        batch["tokens"] = self.vocab.encode_batch(batch["tokens"])

        # convert to tensor
        batch["tokens"] = torch.tensor(batch["tokens"])

        # for train/val split
        if "tags" in samples[0]:
            batch["tags"] = [self.label2idx(sample["tags"]) for sample in samples] # List[List[str]]
            batch["tags"] = torch.tensor(batch["tags"])
            
        return batch
        # raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]