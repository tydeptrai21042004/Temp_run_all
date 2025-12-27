import torch
import torch.nn as nn

class IMDBEmbeddingBagLogReg(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, num_classes, mode="sum", include_last_offset=False)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, idx, offsets):
        return self.emb(idx, offsets) + self.bias
