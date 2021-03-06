import numpy as np
from scipy.ndimage import label
import torch
from torch.autograd import Function
from torch.nn import Module

from core.model.tv2d_numba import prox_tv2d

from time import perf_counter
from numba import jit
import sys

@jit(nopython=True)
def isin(x, l):
    for i in l:
        if x==i:
            return True
    return False

@jit(nopython=True)        
def back(Y, dX, dY):
    neigbhours=list([(1,1)])
    del neigbhours[-1] 
    group=[(0,0)]
    del group[-1]
    n=0
    idx_grouped = [(200,200)for x in range(196)]
    count=0
    value=0
    s=0
    while True:
        if len(neigbhours)!=0:
            while len(neigbhours)!=0:
                if Y[neigbhours[0][0],neigbhours[0][1]] == value:
                    a = neigbhours[0][0]
                    b = neigbhours[0][1]
                    del neigbhours[0]
                    count+=1
                    s+=dY[a,b]
                    group.append((a,b))
                    idx_grouped[n]=(a,b)
                    n+=1
                    if b<13 and isin((a,b+1), idx_grouped)==False and isin((a,b+1), neigbhours)==False:
                        neigbhours.append((a,b+1))
                    if a<13 and isin((a+1,b), idx_grouped)==False and isin((a+1,b), neigbhours)==False:
                        neigbhours.append((a+1,b)) 
                    if b>0 and isin((a,b-1), idx_grouped)==False and isin((a,b-1), neigbhours)==False:
                        neigbhours.append((a,b-1)) 
                    if a>0 and isin((a-1,b), idx_grouped)==False and isin((a-1,b), neigbhours)==False:
                        neigbhours.append((a-1,b)) 
                else:
                    del neigbhours[0]
        else:
            if len(group)>0:
                o=s/count
                count=0
                for x in group:
                    dX[x[0],x[1]]=o
                group=[(0,0)]
                del group[0]
            
            if n>=196:
                break
            B=False
            for i in range(14):
                for j in range(14):
                    if isin((i,j), idx_grouped)==False:
                        value = Y[i,j]
                        s = dY[i,j]
                        count+=1
                        group.append((i, j))
                        idx_grouped[n] = (i, j)
                        n+=1
                        if j<13 and isin((i,j+1), idx_grouped)==False and isin((i,j+1), neigbhours)==False:
                            neigbhours.append((i,j+1))
                        if i<13 and isin((i+1,j), idx_grouped)==False and isin((i+1,j), neigbhours)==False:
                            neigbhours.append((i+1,j)) 
                        if j>0 and isin((i,j-1), idx_grouped)==False and isin((i,j-1), neigbhours)==False:
                            neigbhours.append((i,j-1)) 
                        if i>0 and isin((i-1,j), idx_grouped)==False and isin((i-1,j), neigbhours)==False:
                            neigbhours.append((i-1,j)) 
                        B=True
                        break
                if B:
                    break
    return dX

class TV2DFunction(Function):

    @staticmethod
    def forward(ctx, X, alpha=0.01, max_iter=35, tol=1e-2):
        torch.set_num_threads(8)
        ctx.digits_tol = int(-np.log10(tol)) // 2

        X_np = X.detach().cpu().numpy()
        n_rows, n_cols = X_np.shape
        Y_np = prox_tv2d(X_np.ravel(),
                         step_size=alpha / 2,
                         n_rows=n_rows,
                         n_cols=n_cols,
                         max_iter=max_iter,
                         tol=tol)
        
        
        Y_np = Y_np.reshape(n_rows, n_cols)
        Y = torch.from_numpy(Y_np)  # double-precision
        Y = torch.as_tensor(Y, dtype=X.dtype, device=X.device)
        ctx.save_for_backward(Y.detach())

        return Y

    @staticmethod
    def backward(ctx, dY):
        torch.set_num_threads(8)
        Y, = ctx.saved_tensors
       
        Y_np = np.array(Y.cpu()).round(ctx.digits_tol)
        dY_np = np.array(dY.cpu())
        dX = np.zeros((14,14))
        dX = back(Y_np, dX, dY_np)
        dX = torch.as_tensor(dX, dtype=dY.dtype, device=dY.device)
        
        return dX, None 


_tv2d = TV2DFunction.apply


class TV2D(Module):

    def __init__(self, alpha=1, max_iter=1000, tol=1e-12):
        """2D Total Variation layer

        Computes argmax_P 0.5 ||X - P||^2 + alpha * tv_penalty(P)

        """

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, X):
        return _tv2d(X, self.alpha, self.max_iter, self.tol)


if __name__ == '__main__':
    sys.settrace 

    X = torch.randn(14, 14, requires_grad=True)

    Y = _tv2d(X)
    Y[1, 2].backward()

