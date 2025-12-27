import random
import re
from typing import Dict, List
import torch

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2idx: Dict[str, int], max_len: int = 400):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        toks = _tokenize(self.texts[i])[: self.max_len]
        idx = [self.word2idx.get(w, 0) for w in toks]  # 0 = UNK
        if len(idx) == 0:
            idx = [0]
        return idx, int(self.labels[i])

def build_imdb_loaders(vocab_size=10000, batch_size=256, word_dropout=0.5, max_len=400, num_workers=2):
    try:
        from datasets import load_dataset
    except Exception as e:
        print("[WARN] datasets not available; skipping IMDB:", e)
        return None, None, None

    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    from collections import Counter
    c = Counter()
    for t in train_texts:
        c.update(_tokenize(t))
    most = c.most_common(vocab_size - 2)
    word2idx = {"<unk>": 0, "<pad>": 1}
    for i, (w, _) in enumerate(most, start=2):
        word2idx[w] = i

    train_ds = IMDBDataset(train_texts, train_labels, word2idx, max_len=max_len)
    test_ds = IMDBDataset(test_texts, test_labels, word2idx, max_len=max_len)

    def collate(batch):
        all_idx = []
        offsets = [0]
        labels = []
        for idx, y in batch:
            if word_dropout > 0:
                kept = [w for w in idx if random.random() > word_dropout]
                if len(kept) == 0:
                    kept = [idx[0]]
                idx = kept
            all_idx.extend(idx)
            offsets.append(len(all_idx))
            labels.append(y)

        idx_t = torch.tensor(all_idx, dtype=torch.long)
        offsets_t = torch.tensor(offsets[:-1], dtype=torch.long)
        y_t = torch.tensor(labels, dtype=torch.long)
        return idx_t, offsets_t, y_t

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return train_loader, test_loader, len(word2idx)
