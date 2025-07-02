import torch
import torch.nn as nn

class TokenEmbedding(nn.Module): #  nn.Module
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        # 下一步我们会在这里写内容
        return self.embedding(x)

model = nn.Embedding(100, 16)
x = torch.tensor([[1, 2, 3]])

out = model(x)
print(out.shape)          # torch.Size([1, 3, 16])
print(out)                # 查看嵌入向量
print(model.weight[1])    # 第1号 token 的向量