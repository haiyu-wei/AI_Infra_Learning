import torch

NEG_INF = -1e10
EPSILON = 1e-10

Q_LEN = 6
K_LEN = 6
Q_BLOCK_SIZE = 3
KV_BLOCK_SIZE = 3
P_DROP = 0.2

Tr = Q_LEN
Tc = K_LEN

Q = torch.rand(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
K = torch.rand(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
V = torch.rand(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')

O = torch.zeros_like(Q, requires_grad=True)
l = torch.zeros(Q.shape[:-1])[..., None]
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BlOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

for j in range(len(Q_BLOCKS)):
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]

    for i in range(len(K_BLOCKS)):
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BlOCKS[i]

        S_ij = torch.einsum('... i d, ...j d -> ... i j', Qi, Kj)

        mask = S_ij.ge(0.5)
        S_ij = torch.masked_fill(S_ij, mask, value=0)

        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
        P_ij = torch.exp(S_ij - m_block_ij)
        l_block_ij = torch.sum(P_ij, dim=-1, keepdim=True) + EPSILON
        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)
        mi_new = torch.maximum(m_block_ij, mi)

        li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

        m = torch.nn.Dropout(p=P_DROP)
        P_ij_Vj = m(P_ij_Vj)

        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj

        print(f'----------Attention : Q{i}xK{j}-----------')
        print(O_BLOCKS[i].shape)
        print(O_BLOCKS[0])
        print(O_BLOCKS[1])
        print('\n')

        l_BLOCKS[i] = li_new
        m_BlOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
l = torch.cat(l_BLOCKS, dim=2)
m = torch.cat(m_BlOCKS, dim=2)