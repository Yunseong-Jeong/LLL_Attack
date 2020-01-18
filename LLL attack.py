import torch
import numpy as np
import hashlib


def SHA512(input):
    output = hashlib.sha512(str(input).encode("utf-8")).hexdigest()

    return output


def Hypersphere_ECC(c):
    c = c.reshape(1, -1)
    _, ind = torch.topk(torch.abs(c), 16)

    output = torch.zeros(512)

    for i in range(16):
        if c[0, ind[0, i]] >= 0:
            output[ind[0, i]] = 0.2500
        else:
            output[ind[0, i]] = -0.2500

    output = output.reshape(-1, 1)

    return output


Lattice_P = torch.load("file_name")
input_w = torch.load("file_name")

#dimension = 64
# n = 4
# 2 * lattice_P * input_c => c2?

