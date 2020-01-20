import torch
import numpy as np
import hashlib
import random
import math

def SHA512(input):
    output = hashlib.sha512(str(input).encode("utf-8")).hexdigest()

    return output


def Hypersphere_ECC(c):
    c = c.reshape(1, -1)
    _, ind = torch.topk(torch.abs(c), 4)

    output = torch.zeros(64)

    for i in range(4):
        if c[0, ind[0, i]] >= 0:
            output[ind[0, i]] = 0.5
        else:
            output[ind[0, i]] = -0.5

    output = output.reshape(-1, 1)

    return output

input_w = torch.load("gen_w.tensor")
input_P = torch.load("gen_P.tensor")
input_r = torch.load("gen_r.tensor")

input_w /= input_w.norm()
input_w = input_w.reshape(-1,1)

c = torch.mm(input_P,input_w)
c = Hypersphere_ECC(c)
hash_c = SHA512(c)
print(input_r == hash_c)




