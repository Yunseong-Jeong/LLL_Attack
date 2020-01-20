import torch
import numpy as np
import hashlib
import random
import math

def GenC(level):
    C=torch.rand(1,64)
    if level==5:
        if int(torch.randint(1960000000000001,(1,)))==1:
            level -= 1
        else:
            values,index = torch.topk(C,16)
            C=torch.zeros(1,512)
            #C=C.cuda()
            i=0
            while i<16:
                C[0][int(index[0][i])]=2*(((int(10000*values[0][i]))//1)%2-0.5)*1/4
                i += 1
    if level==4:
        if int(torch.randint(6289000001,(1,)))==1:
            level -= 1
        else:
            values,index = torch.topk(C,8)
            C=torch.zeros(1,512)
            #C=C.cuda()
            i=0
            while i<8:
                C[0][int(index[0][i])]=2*(((int(10000*values[0][i]))//1)%2-0.5)*1/(2*math.sqrt(2))
                i += 1
    if level==3:
        if int(torch.randint(86530,(1,)))==1:
            level -= 1
        else:
            values,index = torch.topk(C,4)
            C=torch.zeros(1,64)
            #C=C.cuda()
            i=0
            while i<4:
                C[0][int(index[0][i])]=2*(((int(10000*values[0][i]))//1)%2-0.5)*1/2
                i += 1
    C=C.reshape(-1,1)
    return C

def random_orthonormal_matrixQ(n):
    """
    Makes a square matrix which is orthonormal by concatenating
    random Householder transformations
    """
    A = np.identity(n)
    d = np.zeros(n)
    d[n - 1] = random.choice([-1.0, 1.0])
    for k in range(n - 2, -1, -1):
        # generate random Householder transformation
        x = np.random.randn(n - k)
        s = math.sqrt((x ** 2).sum())  # norm(x)
        sign = math.copysign(1.0, x[0])
        s *= sign
        d[k] = -sign
        x[0] += s
        beta = s * x[0]
        # apply the transformation
        y = np.dot(x, A[k:n, :]) / beta
        A[k:n, :] -= np.outer(x, y)
    # change sign of rows
    A *= d.reshape(n, 1)
    A = torch.FloatTensor(A)
    return A


def rotation_matrixR(x, y, n=64):
    w = y - torch.mm(x.t(), y) * x  # x,y로 결정되는 평면에서 x와 수직한 벡터를 찾습니다
    w = w / w.norm()  # 기저로 사용하기 위해 크기로 나눠줍니다

    cost = torch.mm(x.t(), y)  # 원래 x에서 y로의 회전을 하려하는 것이므로 cosine값을 구합니다.
    sint = math.sqrt(1 - cost ** 2)  # sine값입니다
    xw = torch.cat((x, w), 1)  # x와y로 결정되는 평면으로의 projection 입니다
    rot_2dim = torch.FloatTensor([[cost, -sint], [sint, cost]])  # x와y로 결정되는 평면에서 회전행렬입니다.
    R = torch.eye(n) - torch.mm(x, x.t()) - torch.mm(w, w.t()) + torch.mm(torch.mm(xw, rot_2dim), xw.t())
    # 앞에서 x,x.t()와 w,w.t()를 빼주는 것은 n-2 dimension으로의 사영시키기 위함입니다.
    # 뒷부분의 x와y로 결정되는 평면으로 사영한 후에, 회전시켜주고 다시 원래 dimension으로 돌아옵니다.

    return R



def SHA512(input):
    output = hashlib.sha512(str(input).encode("utf-8")).hexdigest()

    return output


def GEN(z):
    z = z / z.norm()
    z = z.reshape(-1, 1)

    Q = random_orthonormal_matrixQ(64)

    Qz = torch.mm(Q, z)

    c = GenC(3)

    R = rotation_matrixR(Qz, c)

    P = torch.mm(R, Q)

    r = SHA512(c)

    return r, P


def REP(P, z_prime):
    z_prime = z_prime / z_prime.norm()
    z_prime = z_prime.reshape(-1, 1)

    Pz_prime = torch.mm(P, z_prime)

    c_prime = Hypersphere_ECC(Pz_prime)

    r_prime = SHA512(c_prime)

    return r_prime


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

dimension = 64
# n = 4
# 2 * lattice_P * input_c => c2?

gen_w = np.random.randn(dimension)
gen_w = torch.FloatTensor(gen_w)
print(gen_w)
torch.save(gen_w,"gen_w.tensor")
gen_r,gen_P = GEN(gen_w)
print(gen_r)
print(gen_P)
torch.save(gen_r,"gen_r.tensor")
torch.save(gen_P,"gen_P.tensor")





