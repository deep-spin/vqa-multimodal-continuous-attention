from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from core.model.tv2d_layer_2 import TV2DFunction
from entmax import sparsemax
from functools import partial
from torch import Tensor

from core.model.basis_functions import GaussianBasisFunctions
from core.model.continuous_sparsemax import ContinuousSparsemax
from core.model.continuous_softmax import ContinuousSoftmax
import math 


# --------------------------------------------------------------
# ---- Flatten the sequence (image in continuous attention) ----
# --------------------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.attention=__C.attention
        self.gen_func=gen_func

        if __C.plot_attention == 'True':
            self.plot_attention = True # plot attention at test time
        else: 
            self.plot_attention = False

        if __C.train_rnd == 'True':
            self.train_rnd = True # train model with random number of components
        else: 
            self.train_rnd = False

        self.n_iter = __C.n_iter

        if __C.count == 'True':
            self.count = True # count number of examples per K
        else: 
            self.count = False

        # for spherical covariance matrices and corrected model selectrion criterion set both to True
        self.spherical = False
        self.new_criterion = False

        self.n_modes = 4 # it is possible to change this

        if self.train_rnd==False:
            self.lbd = 5 # setting the value of lambda            
            self.n_inits = 3 # number of intializations per K
            self.n_multimodal = (self.n_modes-1)*self.n_inits 
            if self.count == True:
                self.counts = torch.zeros(1 + self.n_multimodal, 1).to('cuda')
            self.list_n_components = [2,2,2,3,3,3,4,4,4] # if self.n_modes == 4
            self.list_pi_init = []
            self.list_mu_init = []
            self.list_var_init = []
            seed = 222
            torch.manual_seed(seed)
            for i in range (0, self.n_multimodal):
                pi_init, mu_init, var_init = self.random_initialization(self.list_n_components[i])
                self.list_pi_init.append(pi_init)
                self.list_mu_init.append(mu_init)
                self.list_var_init.append(var_init)

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,__C.FLAT_OUT_SIZE)

        if (self.attention=='cont-sparsemax'):
            self.transform = ContinuousSparsemax(psi=None) # use basis functions in 'psi' to define continuous sparsemax
        else:
            self.transform = ContinuousSoftmax(psi=None) # use basis functions in 'psi' to define continuous softmax
        
        self.list_of_lengths = [] # e.g., [[19,25], [19,30], [25,19], ...]
        self.list_of_Gs = [] # e.g, [G([19,25]), G([19,30]), G([25,19]), ...]
        
        # compute psi
        device = 'cuda'
        nb_basis = 100
        self.psi = [None]
        self.psi.append([])
        self.add_gaussian_basis_functions(self.psi[1],nb_basis,device=device)


    def compute_positions_efficient(self, batch, size, L_max, device='cuda'):

        positions = torch.zeros(batch, L_max, 2, 1).to(device)
        lengths = torch.zeros(batch)
        lengths_0_all = torch.zeros(batch)
        lengths_1_all = torch.zeros(batch)

        for length in range(batch):
            length_0 = size[0][length].item()
            length_1 = size[1][length].item()

            lengths_0_all[length] = length_0
            lengths_1_all[length] = length_1

            repeated = 0
            for i in range(length):
                if ((length_0 == lengths_0_all[i]) & (length_1 == lengths_1_all[i])):
                    positions[length] = positions[i]
                    repeated = 1
                    break

            if (repeated==0):
                shift_0 = 1/float(length_0)
                positions_x = torch.linspace(shift_0, 1-shift_0, int(length_0))
                shift_1 = 1/float(length_1)
                positions_y = torch.linspace(shift_1, 1-shift_1, int(length_1))
                positions_x, positions_y=torch.meshgrid(positions_x,positions_y)
                positions_x=positions_x.flatten()
                positions_y=positions_y.flatten()

                for position in range(1,len(positions_x)+1):
                    positions[length][position-1]=torch.tensor([[positions_x[position-1]],[positions_y[position-1]]])

            lengths[length] = int(length_0 * length_1)

        return positions, lengths # [batch, L_max, 2, 1]

    def compute_Gs_with_list(self, size, positions, lengths, nb_basis, ridge_penalty, device='cuda'):
        # positions [batch, L_max, 2, 1]

        batch = positions.size(0)
        L_max = positions.size(1)

        self.Gs=[None] # list with Gs in current batch

        for positions_index in range(batch):

            length_0 = size[0][positions_index].item()
            length_1 = size[1][positions_index].item()
            repeated = 0

            for i in range(len(self.list_of_Gs)):
                if ((length_0 == self.list_of_lengths[i][0]) & (length_1 == self.list_of_lengths[i][1])):
                    G = self.list_of_Gs[i]
                    repeated = 1
                    break

            if (repeated == 0):
                F = torch.zeros(nb_basis, L_max).unsqueeze(2).unsqueeze(3).to(device) # [N, L_max, 1, 1]
                basis_functions = self.psi[1][0]

                L_example = int(lengths[positions_index]) # get current L
                for i in range(L_example): # first L_example features; 0 otherwise
                    F[:,i]=basis_functions.evaluate(positions[positions_index][i])[:]

                I = torch.eye(nb_basis).to(device)
                F=F.squeeze(-2).squeeze(-1) # [N, 196]
                G = F.t().matmul((F.matmul(F.t()) + ridge_penalty * I).inverse()) # [L_max, N]

                # update list_of_Gs & list of lengths
                self.list_of_Gs.append(G.to(device))
                self.list_of_lengths.append([length_0, length_1])
            

            self.Gs.append(G.to(device)) # [None, G1, G2, etc]
            
    def add_gaussian_basis_functions(self, psi, nb_basis, device):
        
        steps=int(math.sqrt(nb_basis))

        mu_x=torch.linspace(0,1,steps)
        mu_y=torch.linspace(0,1,steps)
        mux,muy=torch.meshgrid(mu_x,mu_y)
        mux=mux.flatten()
        muy=muy.flatten()

        mus=[]
        for mu in range(1,nb_basis+1):
            mus.append([[mux[mu-1]],[muy[mu-1]]])
        mus=torch.tensor(mus).to(device)

        sigmas=[]
        for sigma in range(1,nb_basis+1):
            sigmas.append([[0.001,0.],[0.,0.001]]) # it is possible to change this matrix
        sigmas=torch.tensor(sigmas).to(device)

        assert mus.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu=mus, sigma=sigmas))

    def value_function(self, values, batch):
        # Approximate B * F = values via multivariate regression.
        # Use a ridge penalty. The solution is B = values * G

        G = self.Gs[1].unsqueeze(0)
        for i in range(batch-1):
            G = torch.cat((G, self.Gs[i+2].unsqueeze(0)), 0)

        B = torch.transpose(values,-1,-2) @ G
        return B


    # --- EM for weighted data--- #

    def _estimate_log_prob(self, x, mu, var, n_features):
        # x: (batch, n, 1, d, 1) 
        # mu: (batch, 1, k, d, 1)
        # var: (batch, 1, k, d, d)
        
        # returns 
        # log_prob: (batch, n, k, 1, 1)

        var_inv = 1/2. * (var.inverse()+ torch.transpose(var.inverse(),-1,-2)) # to avoid numerical problems
        log_prob = (-0.5 * (torch.transpose(x-mu, -1,-2) @ var_inv @ (x-mu)) - math.log(2*math.pi) - 0.5 * torch.log(torch.det(var).unsqueeze(-1).unsqueeze(-1)))

        return log_prob

    def _e_step(self, x, mu, var, n_features, pi, batch = True):
        # if batch == True        
        # x: (batch, n, 1, d, 1)
        # mu: (batch, 1, k, d, 1)
        # var: (batch, 1, k, d, d)
        # pi: (batch, 1, k, 1, 1)
        # batch: boolean
        
        # returns 
        # torch.mean(log_prob_norm)
        # log_resp (batch, n, k, 1, 1)

        weighted_log_prob = self._estimate_log_prob(x, mu, var, n_features) + torch.log(pi)

        if batch == True:
            log_prob_norm = torch.logsumexp(weighted_log_prob, dim=2, keepdim=True)
        else:
            log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)

        log_resp = weighted_log_prob - log_prob_norm

        if batch == True:
            return torch.mean(log_prob_norm, dim = 1), log_resp
        else:
            return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp, weights, eps, batch = True, spherical=False):
        # if batch == True        
        # x: (batch, n, 1, d, 1)
        # resp: (batch, n, k, 1, 1)
        # weights: (batch, n, 1, 1, 1)
        # batch: boolean
        
        # returns 
        # pi: (batch, 1, k, 1, 1)
        # mu: (batch, 1, k, d, 1)
        # var: (batch, 1, k, d, d)

        resp = torch.exp(log_resp) * weights # include attention weights

        if batch == True:
            N_k = torch.sum(resp, dim=1, keepdim=True) + eps # [batch, 1, k, 1, 1]
            mu = torch.sum(resp * x, dim=1, keepdim=True) / N_k # [batch, 1, k, d, 1]
            if spherical==False:
                var = torch.sum( (resp * (x-mu) @ torch.transpose(x-mu, -1,-2)), dim=1, keepdim=True) / N_k # [batch, 1, k, d, d]
            else:
                var = torch.sum( (resp * (torch.norm(x-mu, dim=-2)**2).unsqueeze(-1) ), dim=1, keepdim=True) / (2 * N_k)
                var = (torch.ones(1,resp.size(2),2).unsqueeze(-1).to(x.device))*(torch.eye(2,2).to(x.device)) * var 
            var = var + (torch.tensor([[1.,0.],[0.,1.]])*1e-6).to(x.device)   
        else:
            N_k = torch.sum(resp, dim=0, keepdim=True) + eps
            mu = torch.sum(resp * x, dim=0, keepdim=True) / N_k
            if spherical == False:
                var = torch.sum( (resp * (x-mu) @ torch.transpose(x-mu, -1,-2)), dim=0, keepdim=True) / N_k
            else:
                var = torch.sum( (resp * (torch.norm(x-mu, dim=-2)**2).unsqueeze(-1) ), dim=0, keepdim=True) / (2 * N_k)
                var = torch.ones(1,n_components,n_features).unsqueeze(-1)*torch.eye(2,2) * var
            var = var + (torch.tensor([[1.,0.],[0.,1.]])*eps).to(x.device) 
                
        pi = N_k # [batch, 1, k, 1, 1]

        return pi, mu, var

    def __em(self, x, mu, var, n_features, pi, weights, eps, batch=True, spherical=False):
        # if batch == True
        # x: (batch, n, 1, d, 1)
        # mu: (batch, 1, k, d, 1)
        # n_features: int
        # var: (batch, 1, k, d, d)
        # pi: (batch, 1, k, 1, 1)
        # weights: (batch, n, 1, 1, 1)
        # eps:
        # batch: boolean

        # returns
        # updated pi, mu, var 
        
        _, log_resp = self._e_step(x, mu, var, n_features, pi, batch)
        pi, mu, var = self._m_step(x, log_resp, weights, eps, batch, spherical=spherical)

        return pi, mu, var

    def __wgt_score(self, x, mu, var, n_features, pi, weights, sum_data=True, batch = True):
        # computes the log-likelihood of the data under the model

        weighted_log_prob = self._estimate_log_prob(x, mu, var, n_features) + torch.log(pi)
        if batch==True:
            per_sample_score = weights * torch.logsumexp(weighted_log_prob, dim=2).unsqueeze(1)
        else:
            per_sample_score = weights * torch.logsumexp(weighted_log_prob, dim=1)

        if sum_data:
            if batch == True:
                return per_sample_score.sum(dim=1).sum(dim=1)
            else:
                return per_sample_score.sum()
        else:
            return torch.squeeze(per_sample_score)

    def random_initialization(self, n_components, var_const = 0.001, n_features = 2, max_seq_len = 196, batch = 1):
        # returns
        # pi_init: (batch, 1, k, 1, 1) // or 1,1,k,1,1
        # mu_init: (batch, 1, k, d, 1) // or 1,1,k,d,1
        # var_init: (batch, 1, k, d, d) // or 1,1,k,d,d

        device = 'cuda'

        pi_init = (torch.ones(batch, 1, n_components, 1, 1) / n_components).to(device) # uniform pi_init (1/K)
        mu_init = torch.rand(batch, 1, n_components, n_features, 1).to(device) # random mu_init btw 0 and 1
        var_init = (var_const * (torch.eye(n_features).unsqueeze(0)).repeat(batch, 1, n_components, 1, 1)).to(device)

        return pi_init, mu_init, var_init

    def get_positions(self, att, max_seq_len = 196):
        # returns
        # positions: (1, n, 1, d, 1); no need to do (batch, n, 1, d, 1) because positions is the same for all the elements in a batch
        positions_x = torch.linspace(0., 1., int(math.sqrt(max_seq_len)))
        positions_x, positions_y = torch.meshgrid(positions_x,positions_x)
        positions_x = positions_x.flatten()
        positions_y = positions_y.flatten()
        positions = torch.zeros(len(positions_x),2,1).to(att.device)
        for position in range(1,len(positions_x)+1):
            positions[position-1]=torch.tensor([[positions_x[position-1]],[positions_y[position-1]]])
        positions = positions.unsqueeze(0).unsqueeze(-3)

        return positions

    def run_EM(self, att, positions, n_components, pi_init, mu_init, var_init, max_seq_len, n_iter, eps):

        n_features = 2 # bidimensional data
        i = 0
        pi = pi_init
        mu = mu_init
        var = var_init

        while (i <= n_iter):
            pi, mu, var = self.__em(positions, mu, var, n_features, pi, att.unsqueeze(-1), eps, batch = True, spherical=self.spherical) # weights = att
            i += 1

        return pi, mu, var


    # --- Continuous attention --- #

    def unimodal_continuous_attention(self, x, att, positions, B, max_seq_len=608):
        # compute distribution parameters 
        Mu = torch.sum(positions @ att.unsqueeze(-1), 1) # [batch, 2, 1]
        Sigma=torch.sum(((positions @ torch.transpose(positions,-1,-2)) * att.unsqueeze(-1)),1) - (Mu @ torch.transpose(Mu,-1,-2)) # [batch, 2, 2]
        Sigma=Sigma + (torch.tensor([[1.,0.],[0.,1.]])*1e-6).to(x.device) # to avoid problems with small values
        if (self.attention=='cont-sparsemax'):
            Sigma=9.*math.pi*torch.sqrt(Sigma.det().unsqueeze(-1).unsqueeze(-1))*Sigma

        # get `mu` and `sigma` as the canonical parameters `theta`
        theta1 = ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) @ Mu).flatten(1)
        theta2 = (-1. / 2. * (1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2)))).flatten(1)
        theta = torch.zeros(x.size(0), 6, device=x.device ) # [batch, 6]
        theta[:,0:2]=theta1
        theta[:,2:6]=theta2

        r = self.transform(theta)  # batch x nb_basis
        r = r.unsqueeze(1)  # batch x 1 x nb_basis

        # get the context vector
        context = torch.matmul(B, r.transpose(-1, -2))
        context = context.transpose(-1, -2)  # batch x 1 x values_size

        return context, Mu, Sigma


    def random_multimodal_continuous_attention(self, x, att, positions, B, n_components, pi_init, mu_init, var_init, max_seq_len=608, n_iter = 5, eps=1e-12):
        # returns
        # context
        # Mu.unsqueeze(1): (batch, 1, k, d, 1)
        # Sigma.unsqueeze(1): (batch, 1, k, d, d)
        # Pi.unsqueeze(1): (batch, 1, k, 1, 1)

        Pi, Mu, Sigma = self.run_EM(att.unsqueeze(-1), positions, n_components, pi_init, mu_init, var_init, max_seq_len=max_seq_len, n_iter=n_iter, eps=eps)
        # Mu: (batch, 1, k, d, 1)
        # Sigma: (batch, 1, k, d, d)
        # Pi: (batch, 1, k, 1, 1)

        Mu = Mu.squeeze(1) # Mu: (batch k, d, 1)
        Sigma = Sigma.squeeze(1) # Sigma: (batch, k, d, d)
        Pi = Pi.squeeze(1) # Pi: (batch, k, 1, 1)

        Theta_i = ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) @ Mu).flatten(-2) # batch,K,2
        Theta_j = (-1. / 2. * (1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2)))).flatten(-2) # batch,K,4
        Theta = torch.zeros(x.size(0), n_components, 6, device = x.device)
        Theta[:,:,0:2] = Theta_i
        Theta[:,:,2:6] = Theta_j
        # Theta.size(): batch, K, 6

        for k in range(0,n_components):
            r_k = self.transform(Theta[:,k,:]) # batch x nb_basis
            r_k = r_k.unsqueeze(1) # batch x 1 x nb_basis
            # get the context vector
            context_k = torch.matmul(B, r_k.transpose(-1, -2))
            context_k = context_k.transpose(-1, -2)  # batch x 1 x values_size

            if k == 0:
                context = (Pi[:,k,:,:] @ context_k)
            else:
                context = context + (Pi[:,k,:,:] @ context_k)

        return context, Pi.unsqueeze(1), Mu.unsqueeze(1), Sigma.unsqueeze(1)


    def forward(self, x, x_mask, size, eval_=False):

        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2),-1e9)

        # for discrete/continuous attention 
        att = self.gen_func(att.squeeze(-1), dim=-1).unsqueeze(-1)

        batch = x.size(0)
        L_max = 608
        nb_basis = 100
        ridge_penalty = 0.01
        positions, lengths = self.compute_positions_efficient(batch, size, L_max, device='cuda')
        self.compute_Gs_with_list(size, positions, lengths, nb_basis, ridge_penalty, device='cuda')

        # map to a probability density over basis functions
        self.transform.psi = self.psi[1]
        # compute B using a multivariate regression
        B = self.value_function(x, batch)

        if self.train_rnd == True:

            # Generate random number for n_components
            n_components = torch.randint(1,self.n_modes+1,(1,)).item()

            # Unimodal continuous attention
            if (n_components == 1):
                # run UCA and obtain context vector
                context, Mu_UCA, Sigma_UCA = self.unimodal_continuous_attention(x, att, positions, B, max_seq_len=196)

            # Multimodal continuous attention
            else:
                pi_init, mu_init, var_init = self.random_initialization(n_components)         
                context, Pi_index, Mu_index, Sigma_index = self.random_multimodal_continuous_attention(x, att, positions.unsqueeze(-3), B, n_components, pi_init, mu_init, var_init, 608, self.n_iter, eps=1e-12)


        else: # test with model selection criterion

            # run UCA and obtain context vector
            context_UCA, Mu_UCA, Sigma_UCA = self.unimodal_continuous_attention(x, att, positions, B, max_seq_len=196)
            Pi_UCA = torch.ones(x.size(0),1,1,1,1).to(att.device)

            # compute log-lik for UCA
            log_lik_UCA= self.__wgt_score(positions.unsqueeze(-3), Mu_UCA.unsqueeze(1).unsqueeze(1), Sigma_UCA.unsqueeze(1).unsqueeze(1), 2, Pi_UCA, att.unsqueeze(-1))

            # create stack of bics
            if self.new_criterion == False:
                bic_UCA = (1*self.lbd - 2*log_lik_UCA).squeeze(-1)
            else:
                bic_UCA = (5*self.lbd - 2*log_lik_UCA).squeeze(-1)
            bics = bic_UCA
            # create stack of contexts
            context_stack = context_UCA
            
            if self.plot_attention == True:
                # create list of Pis, Mus and Sigmas
                Pi_list = [Pi_UCA] 
                Mu_list = [Mu_UCA]
                Sigma_list = [Sigma_UCA]     

            for index in range(0, self.n_multimodal):

                # atm pi/mu/var_init shape is (1,...) instead of (batch,...)
                # that is corrected in the next lines
                pi_init = (self.list_pi_init[index]).repeat(x.size(0),1,1,1,1)
                mu_init = (self.list_mu_init[index]).repeat(x.size(0),1,1,1,1)
                var_init = (self.list_var_init[index]).repeat(x.size(0),1,1,1,1)
                n_components = self.list_n_components[index]

                context_index, Pi_index, Mu_index, Sigma_index = self.random_multimodal_continuous_attention(x, att, positions.unsqueeze(-3), B, n_components, pi_init, mu_init, var_init, 608, self.n_iter, eps=1e-12)

                # compute log-lik, bic and update stack of bics
                log_lik_index = self.__wgt_score(positions.unsqueeze(-3), Mu_index, Sigma_index,2, Pi_index, att.unsqueeze(-1))
                if self.new_criterion == False:
                    bic_index = (n_components * self.lbd - 2*log_lik_index).squeeze(-1)
                else:
                    bic_index = ((4*n_components - 1) * self.lbd - 2*log_lik_index).squeeze(-1)
                bics = torch.cat((bics, bic_index), dim=1)

                # update stack of contexts
                context_stack = torch.cat((context_stack, context_index), dim=1)

                if self.plot_attention == True:
                    # update list of Pis, Mus and Sigmas
                    Pi_list.append(Pi_index)
                    Mu_list.append(Mu_index)
                    Sigma_list.append(Sigma_index)

            # get indices of minimum bic
            _,indices = torch.min(bics,1)

            # create context tensor with zeros
            context = torch.zeros(x.size(0), 1, x.size(-1))

            if self.plot_attention == True:
                best_Pi = torch.zeros(x.size(0), 1, self.n_modes , 1, 1) # self.n_modes is the maximum number os components
                best_Mu = torch.zeros(x.size(0), 1, self.n_modes , 2, 1)
                best_Sigma = torch.zeros(x.size(0), 1, self.n_modes , 2, 2)
                all_sizes = torch.zeros(x.size(0), 2) # create tensor for sizes

            # filling context tensor
            for i in range (0,x.size(0)):
                context[i] = context_stack[i,indices[i],:]
                
                if self.plot_attention == True:
                    best_n_components = Pi_list[indices[i]][i].size(1) 
                    best_Pi[i,:, 0:best_n_components ,:,:] = Pi_list[indices[i]][i]
                    best_Mu[i,:, 0:best_n_components ,:,:] = Mu_list[indices[i]][i]
                    best_Sigma[i,:, 0:best_n_components ,:,:] = Sigma_list[indices[i]][i]

                    all_sizes[i][0] = size[0][i].item() # fill tensor with sizes
                    all_sizes[i][1] = size[1][i].item() # fill tensor with sizes



            if self.count == True:
                for j in range(0,self.n_multimodal+1):
                    if j == 0:
                        zeros = torch.zeros(indices.size(0)).to(att.device)
                        self.counts[0] = self.counts[0] + torch.eq(indices, zeros).sum().to(att.device)
                    else:
                        number = torch.ones(indices.size(0)).to(att.device) * j
                        self.counts[j] = self.counts[j] + torch.eq(indices, number).sum().to(att.device) 

                print('\ncounts:', self.counts)

            context = context.to(x.device)  

        x_atted=context.squeeze(1) # for continuous softmax

        x_atted = self.linear_merge(x_atted) # linear_merge is used to compute Wx

        if self.plot_attention == True:
            return x_atted, best_Pi.squeeze(1), best_Mu.squeeze(1), best_Sigma.squeeze(1), all_sizes.unsqueeze(-1)


        else:
            return x_atted





# ----------------------------------------------------------------
# ---- Flatten the sequence (question and discrete attention) ----
# ----------------------------------------------------------------
# this is also used to flatten the image features with discrete attention
class AttFlatText(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(AttFlatText, self).__init__()
        self.__C = __C

        self.gen_func=gen_func

        if str(gen_func)=='tvmax':
            self.sparsemax = partial(sparsemax, k=512)
            self.tvmax = TV2DFunction.apply


        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,__C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask, size, eval_=False):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2),-1e9)

        temp=1
        if str(self.gen_func)=='tvmax':
            att = att.squeeze(-1)
            for i in range(att.size(0)):
                a = att[i][:size[0][i].item()*size[1][i].item()]
                a = a.view(size[0][i].item(),size[1][i].item())
                att[i][:size[0][i].item()*size[1][i].item()] = self.tvmax(a).view(-1)
            att = self.sparsemax(att.view(att.shape[0],-1), dim=-1).unsqueeze(-1)
        else:
            att = self.gen_func(att.squeeze(-1), dim=-1).unsqueeze(-1)


        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, gen_func=torch.softmax):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=__C.WORD_EMBED_SIZE)

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.attention=__C.attention #added this 

        if __C.plot_attention == 'True':
            self.plot_attention = True # plot attention
        else: 
            self.plot_attention = False

        if __C.train_rnd == 'True':
            self.train_rnd = True # train model with random number of components
        else: 
            self.train_rnd = False

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True)

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE)

        self.gen_func=gen_func
        self.backbone = MCA_ED(__C, gen_func)

        if (self.attention=='discrete'):
            self.attflat_img = AttFlatText(__C, self.gen_func)
        else: # use continuous attention 
            self.attflat_img = AttFlat(__C, self.gen_func)

        self.attflat_lang = AttFlatText(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix, size, eval_=False):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)
        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask)

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask, size, eval_)
        
        if self.plot_attention == True:
            img_feat, Pi, Mu, Sigma, all_sizes = self.attflat_img(
                img_feat,
                img_feat_mask, size, eval_)            
        else:
            img_feat = self.attflat_img(
                img_feat,
                img_feat_mask, size, eval_)

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        if self.plot_attention == True:
            return proj_feat, Pi, Mu, Sigma, all_sizes
        else:
            return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature),dim=-1) == 0).unsqueeze(1).unsqueeze(2)