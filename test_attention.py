import time
import torch

from meta.attention import Attention


embed_dim = 4
seq_dim = 768
batch_dim = 3072

num_head = 1
attn_pdrop = .1
resid_pdrop = .1
scale = True

x = torch.rand(batch_dim, seq_dim, embed_dim)
x = x.cuda()

attention = Attention(embed_dim, seq_dim, num_head, attn_pdrop, resid_pdrop, scale)
attention.cuda()

start_time = time.time()
val = attention(x)
print(time.time() - start_time)

print(val.shape)