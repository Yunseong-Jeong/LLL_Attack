import torch
import numpy as np
import hashlib
from fractions import Fraction
from typing import Iterator, List, Sequence


def SHA512(input):
    output = hashlib.sha512(str(input).encode("utf-8")).hexdigest()

    return output


def Hypersphere_ECC(c):
    c = c.reshape(1, -1)
    _, ind = torch.topk(torch.abs(c), 4)

    output = torch.zeros(64)

    for i in range(4):
        if c[0, ind[0, i]] >= 0:
            output[ind[0, i]] = 0.500
        else:
            output[ind[0, i]] = -0.500

    output = output.reshape(-1, 1)

    return output



#dimension = 64
# n = 4
# 2 * lattice_P * input_c => c2?

input_P = torch.load("P_Reduction.tensor")
input_w1 = torch.load("gen_w1.tensor")
input_w2 = torch.load("gen_w2.tensor")
input_P1 = torch.load("gen_P1.tensor")
input_P2 = torch.load("gen_P2.tensor")

input_w1 /= input_w1.norm()
input_w1 = input_w1.reshape(-1,1)

input_w2 /= input_w2.norm()
input_w2 = input_w2.reshape(-1,1)

c1 = torch.mm(input_P1, input_w1)
c1 = Hypersphere_ECC(c1)

c2 = torch.mm(input_P2, input_w2)
c2 = Hypersphere_ECC(c2)

Proof = torch.mm(input_P, c1)
Proof *= 4

print(c2 == Proof)

ECC_Proof=Hypersphere_ECC(Proof)
print(c2 == ECC_Proof)

ECC_Proof_index = []
c2_index = []

for i in range(0, 64):
    if(abs(ECC_Proof[i])>0.499 ):
        ECC_Proof_index.append(i)
    if(abs(c2[i])>0.499):
        c2_index.append(i)
print("LLL_C1_index")
print(ECC_Proof_index)
print("c2_index")
print(c2_index)