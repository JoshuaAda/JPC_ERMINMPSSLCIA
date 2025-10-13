# IMPORTS
import torch 
import casadi as ca
import cvxpy as cp
import numpy as np
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import torch.optim as optim
import multiprocessing as mp
torch.manual_seed(42)
from torch.utils.data import DataLoader,random_split,TensorDataset
import itertools
import multiprocessing as mp
torch.manual_seed(42)
# CLASSES
class parametricOPz():
    def __init__(self, n_vars = 10, n_eq = 5, n_ineq = 5, obj_type = "quad", eps_fb = 1e-6, op_dict = None,N=20,N_r=1,dt=0.005,n_x=4,n_u=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.obj_type = obj_type # quad, nonconvex, rosenbrock, rosenbrock_modified, rastrigin
        self.eps_fb = eps_fb
        self.rho = 1.0
        if self.obj_type == "nonlinear":
            self.dt=dt
            self.N=N
            self.N_r=N_r
            self.c1 = 0.4#op_dict['c1']
            self.c2 = 0.2#op_dict['c2']
            self.n_vars = (n_x+n_u)*N
            self.n_eq = n_x*N
            self.n_ineq = 2*(n_x+n_u)*N
            self.n_x=n_x
            self.n_u=n_u
            self.n_x_traj = self.n_x * (self.N + 1)
            self.n_u_traj = self.n_u * self.N
            self.multistage=True

            # variables
            self.states = ['C_a','C_b','T_R','T_K']
            self.inputs = ['F']

            # bounds
            self.bounds = {
                'lower': {
                    '_x': {'C_a':0.05,
                           'C_b':0.05,
                            'T_R':0.375,
                            'T_K':0.375},
                    '_u': {'F': 0.05,
                           }
                },
                'upper': {
                    '_x': {'C_a':1,
                           'C_b':1,
                            'T_R':1,
                            'T_K':1},
                    '_u': {'F': 1.00,
                           }
                }
            }

            state_bounds = []
            for state in self.states:
                lb = torch.tensor(self.bounds['lower']['_x'][state], device=self.device)
                ub = torch.tensor(self.bounds['upper']['_x'][state], device=self.device)
                bounds = torch.hstack([lb, ub])
                state_bounds.append(bounds)
            self.state_bounds = torch.stack(state_bounds)

            input_bounds = []
            for input in self.inputs:
                lb = torch.tensor(self.bounds['lower']['_u'][input], device=self.device)
                ub = torch.tensor(self.bounds['upper']['_u'][input], device=self.device)
                bounds = torch.hstack([lb, ub])
                input_bounds.append(bounds)
            self.input_bounds = torch.stack(input_bounds)
            self.c1 = 0.4
            self.c2 = 0.2
            var = [[0.95238, 1, 0.90476],[0.90909, 1, 0.81818],[0.78947, 0.8947, 1],[2/3, 5/6, 1],[120/140, 130/140, 1],[3.8/4.6, 4.2/4.6, 1],[10/12, 11/12,12/12]]#[[1],[1],[1],[1],[1],[1],[1]]##[[0.95238, 1, 0.90476]]#[[1/1.05],[1/1.1],[5.1/5.7],[5/6],[130/140],[4.2/4.6],[11/12]]#[[0.95238, 1, 0.90476],[0.90909, 1, 0.81818],[0.78947, 0.8947, 1],[2/3, 5/6, 1],[120/140, 130/140, 1],[3.8/4.6, 4.2/4.6, 1],[10/12, 11/12,12/12]]#[[1/1.05],[1/1.1],[5.1/5.7],[5/6],[130/140],[4.2/4.6],[11/12]]#[[0.95238, 1, 0.90476],[0.90909, 1, 0.81818],[0.78947, 0.8947, 1],[2/3, 5/6, 1],[120/140, 130/140, 1],[3.8/4.6, 4.2/4.6, 1],[10/12, 11/12,12/12]]#[[1/1.05],[1/1.1],[5.1/5.7],[5/6],[130/140],[4.2/4.6]]#[[0.95238, 1, 0.90476],[0.90909, 1, 0.81818],[0.78947, 0.8947, 1],[2/3, 5/6, 1],[120/140, 130/140, 1],[3.8/4.6, 4.2/4.6, 1]]##[[1/1.05],[1/1.1],[5.1/5.7],[5/6],[130/140],[4.2/4.6]]#[[1/1.05],[1/1.1],[5.1/5.7],[5/6],[130/140],[4.2/4.6]]#[[1/1.05],[1/1.1],[5.1/5.7],[5/6],[130/140],[4.2/4.6]]#[[1],[1],[0.8947],[5/6],[130/140],[4.2/4.6]]#[[1],[1],[0.8947],[5/6],[130/140],[4.2/4.6]]#[[0.95238, 1, 0.90476],[0.90909, 1, 0.81818],[0.78947, 0.8947, 1],[2/3, 5/6, 1],[120/140, 130/140, 1],[3.8/4.6, 4.2/4.6, 1]]#[[1],[1],[0.8947],[5/6],[130/140],[4.2/4.6]]
            self.n_p=len(var)
            self.num_scen=len(var[0])
            self.x_scaling=[2,2,140,140]
            self.u_scaling=[100,8500]
            self.c_scen = self.generate_scenarios(var)#[[0.4, 0.2], [0.4, 0.2], [0.4, 0.2], [0.4, 0.2]]#[[0.35, 0.15], [0.35, 0.25], [0.45, 0.15], [0.45, 0.25]]


        self._calc_dims()

        if op_dict is None:
            self._setup_op()
        else:
            self.op_dict = op_dict
            self.Q = op_dict['Q']
            self.p = op_dict['p']
            self.A = op_dict['A']
            self.G = op_dict['G']
            self.h = op_dict['h']
        self.C, self.Q, self.G, self.R = self.gen_C_Q_G()
        self.p = torch.tensor([0.1, 0.1])
        #self._sanity_check()
        self._setup_functions()
        self._casadi_setup()
    def generate_scenarios(self,current_list):
        #new_list = list(itertools.product(*current_list))
        new_list = [list(combination) for combination in itertools.product(*current_list)]
        return new_list

    # SAVING and LOADING # -------------------------------------
    def save_config(self, folder_path=None, file_name="op_cfg"):
        cfg = {"n_vars": self.n_vars,
                "n_eq": self.n_eq,
                "n_ineq": self.n_ineq,
                "obj_type": self.obj_type,
                "eps_fb": self.eps_fb}
        cfg["op_dict"] = self.op_dict_to_json(self.op_dict)
        if folder_path is None:
            save_pth = Path(file_name+".json")
        else:
            save_pth = Path(folder_path,file_name+".json")        
        with open(save_pth,"w") as f:
            json.dump(cfg,f,indent=4)
        print("settings saved to: ", save_pth)
    
    def op_dict_to_json(self,op_dict):
        op_dict_json = op_dict.copy()
        for key, value in op_dict_json.items():
            if isinstance(value,torch.Tensor):
                op_dict_json[key] = value.tolist()
        return op_dict_json
    
    @classmethod
    def json_to_op_dict(cls,op_dict_json,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        op_dict = op_dict_json.copy()
        for key, value in op_dict.items():
            if isinstance(value,list):
                op_dict[key] = torch.tensor(value,device=device)
        return op_dict

    @classmethod
    def from_dict(cls, cfg):
        cfg["op_dict"] = cls.json_to_op_dict(cfg["op_dict"])
        return cls(**cfg)
    
    @classmethod
    def from_json(cls, folder_pth=None, file_name="op_cfg"):
        if folder_pth is None:
            load_pth = Path(file_name+".json")
        else:
            load_pth = Path(folder_pth,file_name+".json")
        with open(load_pth,"r") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)    
    
    # MAIN # -------------------------------------
    def _calc_dims(self):
        self.n_z = self.n_vars + self.n_eq + self.n_ineq
        self.n_x = 4#self.n_eq
        self.n_y = self.n_vars
        self.n_nu = self.n_eq
        self.n_lam = self.n_ineq
    def gen_C_Q_G(self):

        C = torch.zeros(self.n_eq, self.n_vars, self.n_vars)
        Q= torch.zeros(self.n_vars, self.n_vars)
        self.Q1=torch.eye(self.N,self.N)
        self.Q2=torch.eye(self.N,self.N)
        self.Q3 = torch.eye(self.N, self.N)
        R =torch.zeros(self.n_u*self.N, self.n_u*self.N)
        G = torch.zeros(2 * self.N, self.n_vars)


        for k in range(self.N):
            G[2 * k, 3 * k] = 1
            G[2 * k + 1, 3 * k] = -1
            if k == 0:
                Q[0,0]=1
                Q[1,1]=1
                R[0,0]=.1

            else:
                ind = 3 * k
                Q[4*k,4*k]=1
                Q[4*k+1,4*k+1]=1
                R[k, k] = 0.1
                C[2 * k, ind - 2, ind - 1] = -self.dt
                C[2 * k, ind - 2, ind - 1] = - self.c1 * self.dt
                C[2 * k, ind - 1] = -self.dt
                C[2 * k, ind, ind - 2] = -self.dt * self.c1
                C[2 * k + 1, ind - 2, 3 * k - 1] = self.dt
                C[2 * k + 1, 3 * k - 1, 3 * k - 2] = self.dt
                C[2 * k + 1, 3 * k - 1, 3 * k] = - self.c2 * self.dt
                C[2 * k + 1, 3 * k, 3 * k - 1] = -self.dt * self.c2

        return C,Q,G,R
    def _setup_op(self):
        if self.obj_type in  ["quad", "nonconvex"]:
            self.Q = self.gen_Q()
            self.p = self.gen_p()
        elif self.obj_type == "rosenbrock_modified":
            self.Q = None
            self.p = self.gen_p()
        else:
            self.Q = None
            self.p = None
        self.A = self.gen_A()
        self.G = self.gen_G()
        self.h = self.gen_h()        
        op_dict = {
            'Q': self.Q,
            'p': self.p,
            'A': self.A,
            'G': self.G,
            'h': self.h
        }
        self.op_dict = op_dict
        print("Optimization Problem Matrices Generated.")

    def _sanity_check(self):
        if self.obj_type in  ["quad", "nonconvex"]:        
            assert self.n_vars == self.op_dict['Q'].shape[0]
            assert self.n_vars == self.op_dict['Q'].shape[1]
            assert self.n_vars == self.op_dict['p'].shape[0]
        if self.obj_type == "rosenbrock_modified":
            assert self.n_vars == self.op_dict['p'].shape[0]
        assert self.n_eq == self.op_dict['A'].shape[0]
        assert self.n_vars == self.op_dict['A'].shape[1]
        assert self.n_ineq == self.op_dict['G'].shape[0]
        assert self.n_vars == self.op_dict['G'].shape[1]
        assert self.n_ineq == self.op_dict['h'].shape[0]

    
    def _setup_functions(self):
        self.f_nonlinear_grad = torch.func.grad(self.f_nonlinear, argnums=0)
        self.h_func_grad = torch.func.jacrev(self.h_func, argnums=0)
        self.g_func_grad = torch.func.jacrev(self.g_func, argnums=0)
        if self.obj_type == "quad":
            self.f_func = self.f_quad
            self.f_grad_func = self.f_quad_grad
        elif self.obj_type == "nonconvex":
            self.f_func = self.f_nonconvex
            self.f_grad_func = self.f_nonconvex_grad
        elif self.obj_type == "rosenbrock":
            self.f_func = self.f_rosenbrock
            self.f_grad_func = self.f_rosenbrock_grad
        elif self.obj_type == "rosenbrock_modified":
            self.f_func = self.f_rosenbrock_mod
            self.f_grad_func = self.f_rosenbrock_mod_grad
        elif self.obj_type == "rastrigin":
            self.f_func = self.f_rastrigin
            self.f_grad_func = self.f_rastrigin_grad
        elif self.obj_type == "nonlinear":
            self.f_func = self.f_nonlinear
            self.f_grad_func = self.f_nonlinear_grad

        else:
            raise ValueError("Objective Type not supported.")

        # Batch Functions
        self.f_batch_func = torch.vmap(self.f_func)
        self.h_batch_func = torch.vmap(self.h_func)
        self.g_batch_func = torch.vmap(self.g_func)
        self.KKT_batch_func = torch.vmap(self.KKT_func)
        self.Fk_batch_func = torch.vmap(self.Fk_func)
        self.Tk_batch_func = torch.vmap(self.Tk_func)
        self.Tk_Fk_batch_func = torch.vmap(self.Tk_Fk_func)
        self.Fk_lin_batch_func = torch.vmap(self.Fk_lin_func)
        self.Tk_conv_batch_func = torch.vmap(self.Tk_conv_func)
        self.Tk_Fk_conv_batch_func = torch.vmap(self.Tk_Fk_conv_func)
        self.Vk_batch_func = torch.vmap(self.Vk_func)
        self.gamma_batch_func = torch.vmap(self.gamma_func)
        self.DFk_Dz_batch_func = torch.vmap(self.DFk_Dz_func)
        self.cond_num_batch_func = torch.vmap(self.cond_num_func,chunk_size=1)
        
        # Norm Functions
        self.f_func_grad_y = torch.func.grad(self.f_func,argnums=0)
        self.f_grad_norm = lambda y: torch.norm(self.f_func_grad_y(y),p=torch.inf)
        self.f_grad_norm_batch = torch.vmap(self.f_grad_norm)

    def move_to_device(self,device):
        """Function to move all attributes to new device."""
        self.device = device
        if self.obj_type in  ["quad", "nonconvex"]:
            self.Q = self.Q.to(device)
            self.p = self.p.to(device)
        if self.obj_type == "rosenbrock_modified":
            self.p = self.p.to(device)
        self.A = self.A.to(device)
        self.G = self.G.to(device)
        self.h = self.h.to(device)
        # self.op_dict = {key: value.to(device) for key, value in self.op_dict.items()}

    def f_quad(self,y):
        """Objective function.
        0.5 * y^T Q y + p^T y"""
        f_val = 0.5 * y @ self.Q @ y + self.p @ y
        return f_val
    
    def f_nonconvex(self,y):
        """Objective function.
        0.5 * y^T Q y + p^T sin(y)"""
        f_val = 0.5 * y @ self.Q @ y + self.p @ torch.sin(y)
        return f_val
    def f_nonlinear(self,y,x):
        rho_curr = x[self.n_x + self.n_p+self.N] * 40
        self.c = x[self.n_x:self.n_x + self.n_p]

        lam_F = x[
                self.n_x + self.n_p + 1+self.N:self.n_x + self.n_p + self.n_u  + 1+self.N]
        F_tilde = x[
                  self.n_x + self.n_p + self.n_u + 1+self.N:self.n_x + self.n_p + 2 * self.n_u + 1+self.N]
        past_u = x[
                 self.n_x + self.n_p +self.N+ self.N_r * self.n_u * 2 + 1:self.n_x + self.n_p +self.N+ self.N_r * self.n_u * 3 + 1]
        u_past = torch.hstack((past_u, y[self.n_x * self.N:self.N * (self.n_x + self.n_u) - self.n_u]))
        u = y[self.n_x * self.N:self.N * (self.n_x + self.n_u)]
        c_a = y[:self.N * self.n_x:self.n_x]
        c_b = y[1:self.N * self.n_x:self.n_x]
        # T_R=y[2:self.N * self.n_x:self.n_x]
        # +(T_R-0.88).T@self.Q3@(T_R-0.88)
        f_val = 0.1 * (((c_a - 0.35).T @ self.Q1 @ (c_a - 0.35)) + ((c_b - 0.3).T @ self.Q2 @ (c_b - 0.3)) + (
                    u - u_past).T @ self.R @ (u - u_past) + lam_F[0] * (u[0] - F_tilde[0]) + rho_curr / 2 * (
                                   (u[0] - F_tilde[0]) ** 2))

        return f_val
    # Coupled Rosenbrock
    def f_rosenbrock(self,y):
        """Objective function.
        sum(100*(y_{i+1} - y_i^2)^2 + (1-y_i)^2)"""
        f_val = torch.sum(100*(y[1:] - y[:-1]**2)**2 + (1-y[:-1])**2)
        return f_val/100

    # Coupled Rosenbrock modified by adding linear term --> more active inequality constraints
    def f_rosenbrock_mod(self,y):
        f_val = torch.sum(100*(y[1:] - y[:-1]**2)**2 + (1-y[:-1])**2) + 500*self.p @ y
        return f_val/100
    
    def f_rastrigin(self,y):
        """Objective function.
        10*n_y + sum(y_i^2 - 10*cos(2*pi*y_i))"""
        f_val = torch.sum(y**2 - 10*torch.cos(2*torch.pi*y) + 10)
        return f_val

    def f_quad_grad(self,y):
        """Objective gradient function.
        Q y + p"""
        f_grad_val = self.Q @ y + self.p
        return f_grad_val
    
    def f_nonconvex_grad(self,y):
        """Objective gradient function.
        Q y + p cos(y)"""
        f_grad_val = self.Q @ y + self.p * torch.cos(y)
        return f_grad_val
    
    def f_rosenbrock_grad(self,y):
        """Objective gradient function.
        2*100*(y_{i+1} - y_i^2) + 2*(y_i - 1)"""
        f_grad_val0 = -400*y[0]*(y[1]-y[0]**2) - 2*(1 - y[0])
        f_grad_val1 = -400*y[1:-1]*(y[2:] - y[1:-1]**2) - 2*(1-y[1:-1]) + 200*(y[1:-1]-y[:-2]**2)
        f_grad_val2 = 200*(y[-1]-y[-2]**2)
        f_grad_val = torch.hstack((f_grad_val0,f_grad_val1,f_grad_val2))        
        return f_grad_val/100
    
    def f_rosenbrock_mod_grad(self,y):
        f_grad_val0 = -400*y[0]*(y[1]-y[0]**2) - 2*(1 - y[0])
        f_grad_val1 = -400*y[1:-1]*(y[2:] - y[1:-1]**2) - 2*(1-y[1:-1]) + 200*(y[1:-1]-y[:-2]**2)
        f_grad_val2 = 200*(y[-1]-y[-2]**2)
        f_grad_val = torch.hstack((f_grad_val0,f_grad_val1,f_grad_val2)) + 500*self.p        
        return f_grad_val/100

    def f_rastrigin_grad(self,y):
        """Objective gradient function.
        2*y_i + 10*2*pi*sin(2*pi*y_i)"""
        f_grad_val = 2*y + 20*torch.pi*torch.sin(2*torch.pi*y)
        return f_grad_val

    # NOTE: Initialization of Q, p, A, G, h and x as in DC3 by Donti, Rolnick, and Kolter (2021)
    def gen_Q(self):
        """Generate random Q matrix (objective quadratic term).
        random, [0,1], diagonal matrix"""
        return torch.diag(torch.rand(self.n_vars))

    def gen_p(self):
        """Generate random p vector (objective linear term).
        random, [0,1], vector"""
        return torch.rand(self.n_vars)

    def gen_A(self):
        """Generate random A matrix (equality constraints).
        random, normal distribution, mean=0, std=1"""
        return torch.randn(self.n_eq, self.n_vars)
    
    def gen_G(self):
        """Generate random G matrix (inequality constraints).
        random, normal distribution, mean=0, std=1"""
        return torch.randn(self.n_ineq, self.n_vars)

    def gen_h(self):
        """Generate h vector (inequality constraints).
        sum of absolute values of rows of G @ A_pinv
        Procedure ensures existence of feasible solution."""
        A_pinv = torch.pinverse(self.A)
        return torch.sum(torch.abs(self.G @ A_pinv), axis=1)

    def gen_x(self):
        """Generate random x vector (equality constraints).
        random, [-1,1], vector"""
        return 2*torch.rand(self.n_eq)-1
    
    def batch_gen_x(self,n_batch):
        return 2.0*torch.rand(n_batch,self.n_eq,device=self.device)-1.0
    def batch_gen_x_random(self,n_batch):
        c=self.get_random_scenario(n_batch)
        self.c = c
        lam = 2e-3*torch.rand(n_batch , self.N_r * self.n_u)-1e-3#*2
        rho_new=0.7*torch.rand(n_batch,1)
        u_tilde = torch.tensor([[0.95]],device=self.device).T*torch.rand(n_batch, self.n_u)+torch.tensor([[0.05]],device=self.device).T#torch.rand(n_batch, self.N_r * self.n_u)#self.spread_to_scenarios(self.u_tilde, n_batch)
        x0=torch.tensor([[0.95],[0.95],[0.625],[0.625]],device=self.device).T*torch.rand(n_batch,self.n_x,device=self.device)+torch.tensor([[0.05],[0.05],[0.375],[0.375]],device=self.device).T#.repeat_interleave(len(self.c_scen)**self.N_r,dim=0)
        past_u=torch.tensor([[0.95]],device=self.device).T*torch.rand(n_batch, self.n_u)+torch.tensor([[0.05]],device=self.device).T
        Q_dot=torch.randint(0,2,(n_batch,self.N),device=self.device)
        x = torch.hstack([x0, c,Q_dot,rho_new,lam, u_tilde,past_u])
        return x
    def get_x_from_old(self,n_batch,zk_batch,xk_batch):
        Q_dot_rel=zk_batch[:,self.n_x*self.N+1:(self.n_x+2)*self.N:2]
        rho_new=xk_batch[:,self.n_x+self.n_p:self.n_x+self.n_p+1]
        lam=xk_batch[:,self.n_x+self.n_p+1].reshape(-1,1)
        c=xk_batch[:,self.n_x:self.n_x+self.n_p]
        x0=xk_batch[:,0:self.n_x]
        u_tilde=xk_batch[:,self.n_x+self.n_p+3].reshape(-1,1)
        past_u=xk_batch[:,self.n_x+self.n_p+5].reshape(-1,1)
        u_int = torch.zeros((n_batch, self.N),device=self.device)


        for m in range(n_batch):

            summe = 0
            for i in range(self.N):
                summe = summe + Q_dot_rel[m, i]
                if summe > 0.5:
                    summe -= 1
                    u_int[m, i] = 1
        x = torch.hstack([x0, c, u_int, rho_new, lam, u_tilde, past_u])
        return x
    def batch_gen_x_new(self,n_batch,x0=torch.tensor([]),u0=torch.tensor([]),Q_dot=torch.tensor([])):

        c = self.get_scenarios(n_batch)
        self.num_cons = sum([len(self.c_scen) ** k for k in range(self.N_r)])
        self.rho_new = 0.01/20
        self.c = c
        self.u_tilde=u0
        self.lam=torch.zeros(n_batch*(len(self.c_scen)**self.N_r),self.N_r*self.n_u)
        u_tilde=self.spread_to_scenarios(self.u_tilde,n_batch)

        past_u = torch.zeros(n_batch, self.n_u)

        if x0.numel()==0:
            x=torch.hstack([1.5*torch.rand(n_batch,self.n_x,device=self.device).repeat_interleave(len(self.c_scen)**self.N_r,dim=0),c,torch.tensor([[self.rho_new]],device=self.device).repeat_interleave(n_batch*len(self.c_scen)**self.N_r,dim=0),self.lam,u_tilde,past_u])
        else:
            x = torch.hstack(
                [x0, c,Q_dot,torch.tensor([[self.rho_new]],device=self.device).repeat_interleave(n_batch*len(self.c_scen)**self.N_r,dim=0), self.lam, u_tilde,u0])
        return x#torch.rand(n_batch,self.n_x,device=device)
    def batch_gen_x_new_cas(self,n_batch,x0=torch.tensor([]),u0=torch.tensor([]),Q_dot=torch.tensor([])):

        c = self.get_scenarios(n_batch)
        c=np.array(c.cpu())
        self.num_cons = sum([len(self.c_scen) ** k for k in range(self.N_r)])
        self.rho_new = 0.005/2
        self.c = c
        self.u_tilde=np.zeros((n_batch*self.num_cons,self.n_u))
        self.lam=np.zeros((n_batch*(len(self.c_scen)**self.N_r),self.N_r*self.n_u))
        u_tilde=np.array(self.spread_to_scenarios(torch.tensor(self.u_tilde),n_batch).cpu())
        if x0.numel()==0:
            x=np.array([np.array(1.5*torch.rand(n_batch,self.n_x,device=self.device).repeat_interleave(len(self.c_scen)**self.N_r,dim=0).cpu()),c,torch.tensor([[self.rho_new]],device=self.device).repeat_interleave(n_batch*len(self.c_scen)**self.N_r,dim=0),self.lam,u_tilde])
        else:
            if self.multistage:
                x = np.hstack((np.array(x0.cpu()), c,Q_dot,np.array(u0.cpu()).reshape((-1,1))))
                x=x.reshape((-1,1))
            else:
                x = np.hstack((np.array(x0.cpu()), c, np.array(
                    torch.tensor([[self.rho_new]], device=self.device).repeat_interleave(
                        n_batch * len(self.c_scen) ** self.N_r, dim=0).cpu()), self.lam, u_tilde, np.array(u0.cpu())))

        return x
    def update_x_batch(self,n_batch,zk_batch,xk_batch,u0_old,Q_dot,Tk_batch):
        c=self.get_scenarios(n_batch)
        mask = (Tk_batch < 1e0)
        u=zk_batch[:,self.n_x*self.N:self.n_x*self.N+self.N_r*self.n_u].clone()
        u_update = zk_batch[mask, self.n_x * self.N:self.n_x * self.N + self.N_r * self.n_u].clone()
        self.lam=self.lam+self.rho_new*(u-self.spread_to_scenarios(self.u_tilde,n_batch))
        self.u_tilde = self.update_u_tilde(u_update,n_batch)
        u_tilde=self.spread_to_scenarios(self.u_tilde,n_batch)
        self.rho_new=min(self.rho_new*2.2,0.7)
        x = torch.hstack([xk_batch[:,:self.n_x], c,Q_dot,torch.tensor([[self.rho_new]],device=self.device).repeat_interleave(n_batch*len(self.c_scen)**self.N_r,dim=0), self.lam, u_tilde,u0_old])
        return x

    def update_x_batch_cas(self, n_batch, u, x0,u0):
        c = self.get_scenarios(n_batch)
        c=np.array(c.cpu())

        self.lam = self.lam + self.rho_new * (u - np.array(self.spread_to_scenarios(torch.tensor(self.u_tilde), n_batch).cpu()))
        self.u_tilde = np.array(self.update_u_tilde(torch.tensor(u), n_batch).cpu())
        u_tilde =  np.array(self.spread_to_scenarios(torch.tensor(self.u_tilde),n_batch).cpu())

        self.rho_new = min(self.rho_new*1.2,1)

        x = np.hstack((np.array(x0.cpu()), c, np.array(torch.tensor([[self.rho_new]], device=self.device).repeat_interleave(
            n_batch * len(self.c_scen) ** self.N_r, dim=0).cpu()), self.lam, u_tilde,np.array(u0.cpu())))
        return x
    def get_random_scenario(self,n_batch):
        ind=np.random.randint(0,len(self.c_scen),(n_batch,self.N_r))

        c= [[self.c_scen[ind[m][k]] for k in range(self.N_r)] for m in range(n_batch)]
        c=[c[m] for m in range(n_batch)]
        return torch.tensor(c).reshape(n_batch,-1)

    def update_u_tilde(self,u,n_batch):
        u_first = (torch.sum(u, axis=0) / len(u)).reshape(-1, self.n_u)
        self.u_tilde = self.spread_to_scenarios(u_first, n_batch)
        t = torch.abs(u - self.u_tilde)
        w = t ** 1.5

        if self.N_r == 1 and n_batch == 1:
            # u_tilde=torch.zeros(1,self.n_u)
            u_tilde = (torch.sum(w * u, axis=0) / torch.sum(w, axis=0)).reshape(-1, self.n_u)
        else:
            u_tilde=torch.zeros(self.num_cons*n_batch,self.n_u)

            for m in range(n_batch):
                ind_2=0
                for k in range(self.N_r):
                    num_adds = len(self.c_scen)**(self.N_r-k)
                    num_scenarios=len(self.c_scen)**k
                    for ind_1 in range(num_scenarios):
                        u_tilde[ind_2+self.num_cons*m,:]=torch.sum(u[len(self.c_scen)**self.N_r*m+ind_1*num_adds:len(self.c_scen)**self.N_r*m+(ind_1+1)*num_adds,2*k:2*k+2],dim=0)/num_adds
                        ind_2+=1
        return u_tilde
    def spread_to_scenarios(self,value,n_batch):
        new_value=torch.zeros(len(self.c_scen)**self.N_r*n_batch,self.N_r*self.n_u)
        for s in range(n_batch):
            ind=0
            for r in range(self.N_r):
                num_copies = len(self.c_scen) ** (self.N_r - r )
                for k in range(len(self.c_scen)**r):
                    new_value[s*len(self.c_scen)**self.N_r+k*num_copies:s*len(self.c_scen)**self.N_r+k*num_copies+num_copies,r*self.n_u:(r+1)*self.n_u]=value[ind].reshape((self.n_u,1)).repeat_interleave(num_copies,dim=1).T
                    ind=ind+1
        return new_value

    def get_scenarios(self,n_batch):
        c=[]
        for k in range(n_batch):
            num_scenarios=len(self.c_scen)**self.N_r
            c=c+[self.get_scenario(m) for m in range(num_scenarios)]
        return torch.tensor(c,device=self.device).reshape(-1,self.n_p)#*self.N
    def get_scenario(self,number):
        num_base=self.numberToBase(number,len(self.c_scen),self.N_r)
        c_k=[self.c_scen[num_base[k]] for k in range(self.N_r)]
        c_add=[c_k[-1] for r in range(self.N-self.N_r)]
        return c_k#+c_add
    @staticmethod
    def numberToBase(n, b,a):
        if n == 0:
            return [0 for k in range(a)]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        while len(digits)<a:
            digits.append(0)
        return digits[::-1]
    
    def batch_gen_z(self,n_batch):
        return torch.rand(n_batch,self.n_z,device=self.device)
    
    def L_func(self,y,x,nu,lam):
        """Lagrange function.
        f(y) + nu^T h(y) + lam^T g(y)"""
        L_val = self.f_func(y) + nu @ self.h_func(y,x) + lam @ self.g_func(y)
        return L_val

    def KKT_func(self,z,x):
        self.p = x
        self.rho_curr = x[self.n_x + self.n_x * self.N]
        self.c = torch.tensor([0.4,0.2],device=self.device).repeat_interleave(self.N)
        self.lam_curr = torch.zeros(self.N_r * self.n_u)
        self.u_tilde_curr = torch.zeros(self.N_r * self.n_u)
        y,nu,lam = torch.split(z,[self.n_vars,self.n_eq,self.n_ineq])
        L_grad_y_val = self.L_grad_y_func(y,x,nu,lam)
        h_val = self.h_func(y,x)
        g_val = self.g_func(y)
        g_val_plus = torch.relu(g_val)
        dual_feas_minus = torch.relu(-lam)
        comp_slack = lam*g_val
        KKT_val = torch.cat((L_grad_y_val, h_val, g_val_plus, dual_feas_minus, comp_slack))
        return KKT_val
    
    def Fk_func(self,z,x):
        self.p = x
        self.c = x[self.n_x:self.n_x + self.n_p]
        y,nu,lam = torch.split(z,[self.n_vars,self.n_eq,self.n_ineq])
        L_grad_y_val = self.L_grad_y_func(y,x,nu,lam)
        h_val = self.h_func(y,x)
        g_val = self.g_func(y)
        fb_condition = lam - g_val - torch.sqrt(g_val**2 + lam**2 + self.eps_fb**2)
        Fk = torch.hstack([L_grad_y_val, h_val, fb_condition])
        return Fk
    
    def Tk_func(self,z,x):
        #z=20*z
        Fk = self.Fk_func(z,x)
        Tk = 0.5*torch.dot(Fk,Fk)
        return Tk
    def Tk_Fk_func(self,z,x):
        #z=20*z
        Fk = self.Fk_func(z,x)
        Tk = 0.5*torch.dot(Fk,Fk)
        return Tk,Fk
    
    def Fk_lin_func(self,z,x):
        self.p = x
        y,nu,lam = torch.split(z,[self.n_vars,self.n_eq,self.n_ineq])
        L_grad_y_val = self.L_grad_y_func(y.detach(),x,nu,lam) + self.rho*(y-y.detach().clone())
        h_val = self.h_func(y,x)
        g_val = self.g_func(y)
        fb_condition = lam - g_val - torch.sqrt(g_val**2 + lam**2 + self.eps_fb**2)
        Fk_lin = torch.hstack([L_grad_y_val, h_val, fb_condition])
        return Fk_lin

    def Tk_conv_func(self,z,x):
        Fk_lin = self.Fk_lin_func(z,x)
        Tk_conv = 0.5*torch.dot(Fk_lin,Fk_lin)
        return Tk_conv
    def Tk_Fk_conv_func(self,z,x):
        Fk_lin = self.Fk_lin_func(z,x)
        Tk_conv = 0.5*torch.dot(Fk_lin,Fk_lin)
        return Tk_conv,Fk_lin
    #####
    
    def gamma_func(self,z,x,dz):
        """Function to compute the step size."""
        Fk, jvpFk = torch.func.jvp(self.Fk_func, (z,x), (dz,torch.zeros_like(x)))
        gamma = - Fk@jvpFk/(jvpFk@jvpFk)
        return gamma
    
    def Vk_func(self,z,x,dz):
        """Newton type loss as in Lüken L. and Lucia S. (2024)"""
        Fk, jvpFk = torch.func.jvp(self.Fk_func, (z,x), (dz,torch.zeros_like(x)))
        dFk = Fk+jvpFk
        Vk = 0.5*torch.dot(dFk,dFk)
        return Vk

    # KKT Matrix
    def DFk_Dz_func(self,z,x):
        DFk_Dz_val = torch.func.jacfwd(self.Fk_func, argnums=0)(z,x)
        return DFk_Dz_val
    
    def cond_num_func(self,z,x):
        # calculate KKT matrix
        DFk_Dz = self.DFk_Dz_func(z,x)

        # calculate condition number
        cond_num = torch.linalg.cond(DFk_Dz)
        return cond_num
      
    # Casadi implementation for usage of IPOPT for NLPs
    def _casadi_setup(self):
        y_sym = ca.MX.sym('y',self.n_vars)
        x_sym = ca.MX.sym('x',self.n_eq)
        if self.obj_type in  ["quad", "nonconvex","nonlinear"]:
            Q = ca.MX(self.Q.cpu().numpy())
            p = ca.MX(self.p.cpu().numpy())
            R = ca.MX(self.R.cpu().numpy())
        if self.obj_type == "quad":
            f_sym = 0.5 * y_sym.T @ Q @ y_sym + p.T @ y_sym
        elif self.obj_type == "nonconvex":
            f_sym = 0.5 * y_sym.T @ Q @ y_sym + p.T @ ca.sin(y_sym)
        elif self.obj_type == "rosenbrock":
            f_sym = ca.sum1(100*(y_sym[1:] - y_sym[:-1]**2)**2 + (1-y_sym[:-1])**2)/100
            # f_sym = ca.sum1(100*(y_sym[1:] - y_sym[:-1]**2)**2 + (1-y_sym[:-1])**2)
        elif self.obj_type == "rosenbrock_modified":
            p = ca.MX(self.p.cpu().numpy())
            f_sym = ca.sum1(100*(y_sym[1:] - y_sym[:-1]**2)**2 + (1-y_sym[:-1])**2)/100 + 5*p.T @ y_sym
        elif self.obj_type == "rastrigin":
            f_sym = ca.sum1(y_sym**2 - 10*ca.cos(2*ca.pi*y_sym) + 10)
        elif self.obj_type == "nonlinear":
            f_sym = ca.sum1((y_sym - 0.6).T @ Q @ (y_sym - 0.6))
        else:
            raise ValueError("Objective Type not supported.")
        

        if self.obj_type == "nonlinear":
            if self.multistage:
                x = ca.MX.sym('x', self.num_scen**self.n_p*(self.n_x + self.n_p  + self.n_u+self.N ))
                y = ca.MX.sym('y',self.num_scen**self.n_p*self.n_vars)
                y_sym=y[0:self.n_vars]
                x_sym=x[0:(self.n_x + self.n_p + self.n_u +self.N)]
                ineq_sym = self.g_func_cas(y_sym, x[0:self.n_x])
                eq_sym = self.h_func_cas(y_sym, x[0:self.n_x], x[self.n_x:self.n_x + self.n_p+self.N])

                past_u = x_sym[
                         self.n_x + self.n_p +self.N :self.n_x + self.n_p +self.n_u+self.N]
                u_past = ca.vertcat(past_u.T,
                                    y_sym[self.n_x * self.N:self.N * (self.n_x + self.n_u) - self.n_u].reshape(
                                        (-1, self.n_u))).reshape((-1, 1))
                u = y_sym[self.n_x * self.N:self.N * (self.n_x + self.n_u)]

                c_a = y_sym[0:self.N]
                c_b = y_sym[self.N:self.N * 2]
                T_R = y_sym[self.N*2:self.N * 3]
                Q1 = np.array(self.Q1.detach().cpu())
                Q2 = np.array(self.Q2.detach().cpu())
                R = np.zeros((self.n_u * self.N, self.n_u * self.N))
                for k in range(self.N):
                    R[k, k] = 0.1

                f_sym = ((c_a - 0.35).T @ Q1 @ (c_a - 0.35) + (c_b - 0.3).T @ Q2 @ (c_b - 0.3) + (u - u_past).T @ R @ (
                            u - u_past))
                for k in range(1,self.num_scen**self.n_p):
                    x_sym = x[k * (
                                self.n_x +  self.n_p + self.n_u+self.N ):(k+1) * (
                                self.n_x +  self.n_p  + self.n_u+self.N)]
                    y_sym=y[k*self.n_vars:(k+1)*self.n_vars]

                    ineq_sym =ca.vertcat(ineq_sym,self.g_func_cas(y_sym, x_sym[0:self.n_x]))
                    eq_sym = ca.vertcat(eq_sym,self.h_func_cas(y_sym, x_sym[0:self.n_x], x_sym[self.n_x:self.n_x + self.n_p+self.N]))


                    past_u = x_sym[
                             self.n_x + self.n_p +self.N :self.n_x + self.n_p  + self.n_u+self.N]
                    u_past = ca.vertcat(past_u.T,
                                        y_sym[self.n_x * self.N:self.N * (self.n_x + self.n_u) - self.n_u].reshape(
                                            (-1, self.n_u))).reshape((-1, 1))
                    u = y_sym[self.n_x * self.N:self.N * (self.n_x + self.n_u)]
                    u0_anti=y[(k-1)*self.n_vars+self.n_x*self.N:k*self.n_vars:self.N]
                    eq_non_anti =u[0::self.N]-u0_anti
                    eq_sym=ca.vertcat(eq_sym,eq_non_anti)
                    c_a = y_sym[0:self.N]
                    c_b = y_sym[self.N:self.N * 2]
                    T_R = y_sym[self.N*2:self.N * 3]
                    Q1 = np.array(self.Q1.detach().cpu())
                    Q2 = np.array(self.Q2.detach().cpu())
                    R = np.zeros((self.n_u * self.N, self.n_u * self.N))
                    for k in range(self.N):
                        R[k, k] = 0.1
                        #R[2 * k, 2 * k] = 0
                    f_sym =f_sym+ ((c_a - 0.35).T @ Q1 @ (c_a - 0.35) + (c_b - 0.3).T @ Q2 @ (c_b - 0.3) + (
                                u - u_past).T @ R @ (
                                    u - u_past))
            else:
                x_sym = ca.MX.sym('x', self.n_x+self.N*self.n_p+1+self.n_u*(self.N_r*2+1))
                ineq_sym = self.g_func_cas(y_sym, x_sym[0:self.n_x])
                eq_sym = self.h_func_cas(y_sym, x_sym[0:self.n_x],x_sym[self.n_x:self.n_x+self.N*self.n_p])
                rho_curr=x_sym[self.n_x+self.n_p*self.N]*20
                lam_curr=x_sym[self.n_x+self.n_p*self.N+1:self.n_x+self.n_p*self.N+1+self.N_r*self.n_u]
                u_tilde_curr=x_sym[self.n_x+self.n_p*self.N+1+self.N_r*self.n_u:self.n_x+self.n_p*self.N+1+self.N_r*self.n_u*2]
                past_u = x_sym[
                         self.n_x + self.n_p * self.N + self.N_r * self.n_u * 2 + 1:self.n_x + self.n_p * self.N + self.N_r * self.n_u * 2 + 1 + self.n_u]
                u_past = ca.vertcat(past_u.T, y_sym[self.n_x * self.N:self.N * (self.n_x + self.n_u) - self.n_u].reshape((-1,self.n_u))).reshape((-1,1))
                u = y_sym[self.n_x * self.N:self.N * (self.n_x + self.n_u)]

                c_a=y_sym[0:self.N]
                c_b=y_sym[self.N:self.N*2]

                Q1=np.array(self.Q1.detach().cpu())
                Q2=np.array(self.Q2.detach().cpu())
                R=np.zeros((self.n_u*self.N,self.n_u*self.N))
                for k in range(self.N):
                    R[k,k]=0.1
                    #R[2*k,2*k]=1e-3
                f_sym = (c_a - 0.35).T @ Q1 @ (c_a - 0.35)+(c_b - 0.3).T @ Q2 @ (c_b - 0.3)+(u-u_past).T @ R @ (u-u_past)#+0.5*rho_curr*(sum([(y_sym[self.n_x*self.N+k*self.N]-u_tilde_curr[k])**2 for k in range(self.n_u)]))+sum([lam_curr[k]*(y_sym[self.n_x*self.N+k*self.N]-u_tilde_curr[k]) for k in range(self.n_u)])#+ca.sum1((u-u_past).T @ R @ (u-u_past))#+lam_curr[k+1]*(y_sym[self.n_x*self.N+k*self.N]-u_tilde_curr[k+1])+0.5*rho_curr*(ca.sqrt([(y_sym[self.n_x*self.N+k*self.N]-u_tilde_curr[k])**2 for k in range(self.n_u)])#+lam_curr.T@(y_sym[self.n_x*self.N:self.N*(self.n_x)+self.N*(self.N_r+1):self.N]-u_tilde_curr)+0.5*rho_curr*sum([ca.sumsqr(y_sym[self.n_x*self.N:self.N*(self.n_x)+self.N*(self.N_r+1):self.N].reshape((-1,self.n_u))[k,:]-u_tilde_curr.reshape((-1,self.n_u))[k,:])**2 for k in range(self.N_r)])#+lam_curr[0]*(y_sym[48]-u_tilde_curr[0])+lam_curr[1]*(y_sym[72]-u_tilde_curr[1])+0.5*rho_curr*(ca.sqrt((y_sym[48]-u_tilde_curr[0])**2+(y_sym[72]-u_tilde_curr[1])**2))**2#
                self.n_eq = eq_sym.shape[0]
                self.n_ineq = ineq_sym.shape[0]
        nlp = {'x': y_sym,'p': x_sym, 'f': f_sym, 'g': ca.vertcat(eq_sym,ineq_sym)}
        lb_eq = np.zeros(self.n_eq)
        ub_eq = np.zeros(self.n_eq)
        lb_ineq = -np.inf * np.ones(self.n_ineq)
        ub_ineq = np.zeros(self.n_ineq)

        lbg = ca.vertcat(lb_eq, lb_ineq)
        ubg = ca.vertcat(ub_eq, ub_ineq)
        if self.multistage:
            nlp = {'x': y, 'p': x, 'f': f_sym, 'g': ca.vertcat(eq_sym, ineq_sym)}
            lb_eq = np.zeros((self.n_eq+1)*self.num_scen**self.n_p-1)
            ub_eq = np.zeros((self.n_eq+1)*self.num_scen**self.n_p-1)
            lb_ineq = -np.inf * np.ones(self.n_ineq*self.num_scen**self.n_p)
            ub_ineq = np.zeros(self.n_ineq*self.num_scen**self.n_p)

            lbg = ca.vertcat(lb_eq, lb_ineq)
            ubg = ca.vertcat(ub_eq, ub_ineq)

        self.nlp = nlp
        self.lbg = lbg
        self.ubg = ubg

        # s_opts = {'ipopt.fixed_variable_treatment': 'make_constraint','ipopt.tol': 1e-10, 'ipopt.acceptable_tol': 1e-10}
        s_opts = {'ipopt.tol': 1e-10, 'ipopt.acceptable_tol': 1e-10,'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
        self.ca_solver = ca.nlpsol('solver', 'ipopt', nlp, s_opts)

    def setup_casadi_single(self):
        x = ca.MX.sym('x', (self.n_x + self.n_p + self.n_u*self.N+ self.n_u*3+1))
        y = ca.MX.sym('y',  self.n_vars)
        y_sym = y[0:self.n_vars]
        x_sym = x[0:(self.n_x + self.n_p + self.n_u*self.N+3*self.n_u)+1]
        ineq_sym = self.g_func_cas(y_sym, x[0:self.n_x])
        eq_sym = self.h_func_cas(y_sym, x[0:self.n_x], x[self.n_x:self.n_x + self.n_p+self.n_u*self.N])
        rho=20*x[self.n_x+self.n_p+self.N]
        lam_F=x_sym[
                 self.n_x + self.n_p+1+self.N:self.n_x + self.n_p +self.n_u+1+self.N]
        F_tilde=x_sym[
                 self.n_x + self.n_p+self.n_u+1+self.N:self.n_x + self.n_p + 2*self.n_u+1+self.N]

        past_u = x_sym[
                 self.n_x + self.n_p+2*self.n_u+1+self.N:self.n_x + self.n_p + 3*self.n_u+1+self.N]


        c_a = y_sym[0:self.N]
        c_b = y_sym[self.N:self.N * 2]
        F = y_sym[self.n_x * self.N:self.N * (self.n_x + 1)]
        F_old = ca.vertcat(past_u[0], F[:-1])
        T_R = y_sym[2 * self.N:3 * self.N]
        Q1 = np.array(self.Q1.detach().cpu())
        Q2 = np.array(self.Q2.detach().cpu())
        R1 = np.zeros((self.N, self.N))
        R2 = np.zeros((self.N, self.N))
        # +(T_R - 0.88).T @ Q2 @ (T_R - 0.88)
        for k in range(self.N):
            R1[k, k] = 0.1
            R2[k, k] = 1e-3
        f_sym = (c_a - 0.35).T @ Q1 @ (c_a - 0.35) + (c_b - 0.3).T @ Q2 @ (c_b - 0.3) + (F - F_old).T @ R1 @ (
                    F - F_old)  +lam_F*(F[0]-F_tilde[0])+rho/2*((F[0]-F_tilde[0])**2)
        nlp = {'x': y, 'p': x, 'f': f_sym, 'g': ca.vertcat(eq_sym, ineq_sym)}
        lb_eq = np.zeros(self.n_eq)
        ub_eq = np.zeros(self.n_eq)
        lb_ineq = -np.inf * np.ones(self.n_ineq)
        ub_ineq = np.zeros(self.n_ineq)

        lbg = ca.vertcat(lb_eq, lb_ineq)
        ubg = ca.vertcat(ub_eq, ub_ineq)

        self.nlp_small = nlp
        self.lbg_small = lbg
        self.ubg_small = ubg
        s_opts = {'ipopt.tol': 1e-10, 'ipopt.acceptable_tol': 1e-10}
        self.ca_solver_small = ca.nlpsol('solver', 'ipopt', nlp, s_opts)
    def init_small(self,approx_mpc):
        self.approx_mpc = approx_mpc
        # logging
        self.history = {"epoch": []}
        self.dir = None
    def sample_closed_loop_data_small(self,num_samples=1000,data_dir="./sampling"):
        self.setup_casadi_single()
        n_batch = num_samples

        data_dir = Path(data_dir)
        for n in range(n_batch):
            t1=time.time()
            x_opt = []
            u0_opt = []
            c = self.get_scenarios(
                1)
            c=np.array(c.cpu())
            u_tilde = np.array((torch.tensor([[0.95], [1]], device=self.device).T * torch.rand(1, self.n_u) + torch.tensor(
                [[0.05], [0]],
                device=self.device).T).repeat(len(c),1).cpu())
            x0 = np.array((torch.tensor([[0.95], [0.95], [0.625], [0.625]], device=self.device).T * torch.rand(1, self.n_x,
                                                                                                     device=self.device) + torch.tensor(
                [[0.05], [0.05], [0.375], [0.375]],
                device=self.device).T).repeat(len(c),1).cpu())
            u0=u_tilde
            self.num_cons = sum([len(self.c_scen) ** k for k in range(self.N_r)])
            self.rho_new = 0.2 / 20
            self.c = c
            self.u_tilde = u_tilde
            self.lam = np.zeros(((len(self.c_scen) ** self.N_r), self.N_r * self.n_u))
            u_tilde = np.array(self.spread_to_scenarios(torch.tensor(self.u_tilde),
                                               1).cpu().detach())
            x = np.hstack(
                [x0, c, np.array(torch.tensor([[self.rho_new]], device=self.device).repeat_interleave(
                     len(self.c_scen) ** self.N_r, dim=0).cpu()), self.lam, u_tilde, u0])
            cont=True
            y_init_guess = np.random.rand(self.n_vars)
            for s in range(1):
                u=np.zeros((len(x),self.N))
                u_update=np.zeros((len(x),self.n_u))
                for m in range(len(x)):
                    x_curr = x[m].reshape((-1, 1))
                    res = self.ca_solver_small(x0=y_init_guess, p=x_curr, lbx=-np.inf, ubx=np.inf, lbg=self.lbg_small,
                                               ubg=self.ubg_small)
                    u[m] = res['x'][(self.n_x+1)*self.N:self.N*(self.n_x+2)].full().reshape((-1,))
                    u_update[m]=res['x'][self.n_x * self.N:(self.n_x + self.n_u) * self.N:self.N].full().reshape((-1,))
                    if self.ca_solver_small.stats()['success']:
                        x_opt.append(x_curr)
                        u0_opt.append(u[m].reshape((self.N,1)))
                    else:
                        cont=False
                        break
                x=self.update_x_batch_cas(1,u_update,x0,u0)
                if not cont:
                    break
            t2=time.time()
            print("Number of samples: " + str(n))
            print("Time: " + str(t2-t1))
            file_path = data_dir.joinpath('data_n' + str(num_samples)+"_"+str(n) + '_opt.pkl')

            # Save the list using pickle
            with open(file_path, 'wb') as f:
                pickle.dump(x_opt, f)
                pickle.dump(u0_opt,f)

    def generate_dataset(self, num_samples,data_dir='./sampling'):
        x_opt = []
        u0_opt = []
        data_dir = Path(data_dir)
        for k in range(num_samples):
            file_path = data_dir.joinpath('data_n' + str(num_samples)+"_"+str(k) + '_opt.pkl')
            with open(file_path, 'rb') as f:
                x=pickle.load(f)
        x_opt = torch.tensor(x_opt, device=self.device, dtype=torch.float32).squeeze()
        u0_opt = torch.tensor(u0_opt, device=self.device, dtype=torch.float32).squeeze()
        data = TensorDataset(x_opt, u0_opt)

        torch.save(data, data_dir.joinpath('data_n' + str(num_samples) + '_opt.pth'))


    def sample_random_data_small(self,num_samples=1000,data_dir="./sampling"):
        self.setup_casadi_single()
        n_batch=num_samples
        c = self.get_random_scenario(n_batch)
        self.c = c
        # self.u_tilde = torch.rand(n_batch * self.num_cons, self.n_u)
        lam = 1e-3 * torch.rand(n_batch, self.N_r * self.n_u) - 1e-3 # *2
        rho_new = 0.7*torch.rand(n_batch, 1)
        u_tilde = torch.tensor([[0.95]], device=self.device).T * torch.rand(n_batch, self.n_u) + torch.tensor(
            [[0.05]],
            device=self.device).T  # torch.tensor([0.05,1]).repeat(n_batch).reshape(n_batch,-1)##torch.rand(n_batch, self.N_r * self.n_u)#self.spread_to_scenarios(self.u_tilde, n_batch)
        x0 = torch.tensor([[0.95], [0.95], [0.625], [0.625]], device=self.device).T * torch.rand(n_batch, self.n_x,
                                                                                                 device=self.device) + torch.tensor(
            [[0.05], [0.05], [0.375], [0.375]],
            device=self.device).T  # torch.tensor([0.4,0.25,134.14/140,130/140]).repeat(n_batch).reshape(n_batch,-1)##.repeat_interleave(len(self.c_scen)**self.N_r,dim=0)
        past_u = torch.tensor([[0.95]], device=self.device).T * torch.rand(n_batch, 1)+ torch.tensor([[0.05]],device=self.device).T  # torch.tensor([[0.95],[1]],device=self.device).T*torch.rand(n_batch, self.n_u)+torch.tensor([[0.05],[0]],device=self.device).T#torch.tensor([0.05,1]).repeat(n_batch).reshape(n_batch,-1)#
        Q_dot=torch.randint(0, 2, (n_batch, self.N))
        x = np.array(torch.hstack([x0, c,Q_dot, rho_new, lam, u_tilde, past_u]).cpu().detach())
        y_init_guess = np.random.rand(self.n_vars)
        x_opt=[]
        u0_opt=[]
        for k in range(num_samples):
            x_curr=x[k].reshape((-1,1))
            res = self.ca_solver_small(x0=y_init_guess, p=x_curr, lbx=-np.inf, ubx=np.inf, lbg=self.lbg_small, ubg=self.ubg_small)
            u0=res['x'][self.n_x*self.N:(self.n_x+self.n_u)*self.N:self.N].full()
            if self.ca_solver_small.stats()['success']:
                x_opt.append(x_curr)
                u0_opt.append(u0)

        x_opt=torch.tensor(x_opt,device=self.device,dtype=torch.float32).squeeze()
        u0_opt=torch.tensor(u0_opt,device=self.device,dtype=torch.float32).squeeze()
        data = TensorDataset(x_opt, u0_opt)
        data_dir = Path(data_dir)
        torch.save(data, data_dir.joinpath('data_n' + str(num_samples) + '_opt.pth'))

    def load_data(self, data_dir, num_samples, val=0.2, batch_size=1000, shuffle=True, learning_rate=1e-3):
        data_dir=Path(data_dir)
        data = torch.load(data_dir.joinpath('data_n' + str(num_samples) + '_opt.pth'))
        training_data, val_data = random_split(data, [1 - val, val],generator=torch.Generator(device=self.device))
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle,generator=torch.Generator(device=self.device))
        test_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle,generator=torch.Generator(device=self.device))
        lr_scheduler_patience = 100  # train_config["lr_scheduler_patience"]
        lr_scheduler_cooldown = 10  # train_config["lr_scheduler_cooldown"]
        lr_reduce_factor = 0.1  # train_config["lr_reduce_factor"]
        min_lr = 1e-8  # train_config["min_lr"]

        # Early Stopping
        early_stop = True  # train_config["early_stop"]
        optimizer = optim.Adam(self.approx_mpc.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor,
                                                                  patience=lr_scheduler_patience,
                                                                  threshold=1e-5, threshold_mode='rel',
                                                                  cooldown=lr_scheduler_cooldown, min_lr=min_lr,
                                                                  eps=0.0)

        return train_dataloader, test_dataloader, optimizer, lr_scheduler


    def default_training(self,n_samples,n_epochs,data_dir='./sampling'):
        train_dataloader,test_dataloader,optimizer,rl_scheduler=self.load_data(data_dir,n_samples)
        self.train(n_epochs,optimizer,train_dataloader,rl_scheduler,val_loader=test_dataloader)
    def log_value(self, val, key):
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        assert isinstance(val, (int, float)), "Value must be a scalar."
        if not key in self.history.keys():
            self.history[key] = []
        self.history[key].append(val)

    def print_last_entry(self, keys=["epoch,train_loss"]):
        assert isinstance(keys, list), "Keys must be a list."
        for key in keys:
            # check wether keys are in history
            assert key in self.history.keys(), "Key not in history."
            print(key, ": ", self.history[key][-1])

    def visualize_history(self, key, log_scaling=False, save_fig=False):
        fig, ax = plt.subplots()
        ax.plot(self.history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        if log_scaling:
            ax.set_yscale('log')
        ax.legend()
        fig.show()
        if save_fig:
            assert self.dir is not None, "exp_pth must be provided."
            plt.savefig(self.dir.joinpath(key + ".png"))
        return fig, ax

    def train_step(self, optim, x, y):
        optim.zero_grad()
        y_pred = self.approx_mpc(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        optim.step()
        return loss.item()

    def train_epoch(self, optim, train_loader):
        train_loss = 0.0
        # Training Steps
        for idx_train_batch, batch in enumerate(train_loader):
            x, y = batch
            loss = self.train_step(optim, x, y)
            train_loss += loss
        n_train_steps = idx_train_batch + 1
        train_loss = train_loss / n_train_steps
        return train_loss

    def validation_step(self, x, y):
        with torch.no_grad():
            y_pred = self.approx_mpc(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
        return loss.item()

    def validation_epoch(self, val_loader):
        val_loss = 0.0
        for idx_val_batch, batch in enumerate(val_loader):
            x_val, y_val = batch
            loss = self.validation_step(x_val, y_val)
            val_loss += loss
        n_val_steps = idx_val_batch + 1
        val_loss = val_loss / n_val_steps
        return val_loss

    def train(self, N_epochs, optim, train_loader, lr_scheduler, val_loader=None, print_frequency=10):
        for epoch in range(N_epochs):
            # Training
            train_loss = self.train_epoch(optim, train_loader)
            lr_scheduler.step(train_loss)

            # Logging
            self.log_value(epoch, "epochs")
            self.log_value(train_loss, "train_loss")
            self.log_value(optim.param_groups[0]["lr"], "lr")
            print_keys = ["epochs", "train_loss", "lr"]

            # Validation
            if val_loader is not None:
                val_loss = self.validation_epoch(val_loader)
                self.log_value(val_loss, "val_loss")
                print_keys.append("val_loss")

            # Print
            if (epoch + 1) % print_frequency == 0:
                self.print_last_entry(keys=print_keys)
                print("-------------------------------")
            if optim.param_groups[0]["lr"] <= 1e-8:
                break
        return self.history
    def casadi_solve(self,x,y_init=None):
        if y_init is None:
            if self.multistage:
                y_init_guess=np.random.rand(self.n_vars*self.num_scen**self.n_p)
            else:
                y_init_guess = np.random.rand(self.n_vars)
        else:
            y_init_guess = y_init.cpu().numpy()
        res = self.ca_solver(x0=y_init_guess, p=x, lbx=-np.inf, ubx=np.inf, lbg=self.lbg, ubg=self.ubg)
        return res
    def casadi_solve_parallel(self,x_batch):
        y_init_guess = np.random.rand(self.n_vars)
        pool = mp.Pool(processes=mp.cpu_count())
        tasks = [(y_init_guess,x_batch[m],-np.inf,np.inf,self.lbg,self.ubg) for m in range(len(x_batch))]
        # Execute the worker function in parallel
        results = pool.starmap(self.ca_solver, tasks)

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        return results
    def L_grad_y_func(self,y,x,nu,lam):
        """Lagrange function gradient w.r.t y (decision variable).
        f_grad(y) + A^T nu + G^T lam"""
        if self.obj_type== "nonlinear":
            #s=time.time()
            (_,fun_h_grad)=torch.func.vjp(self.h_func,y,x)
            (_,fun_g_grad)=torch.func.vjp(self.g_func,y)
            L_grad_y_val = self.f_grad_func(y,x)+fun_h_grad(nu)[0]+fun_g_grad(lam)[0]#torch.func.jvp(self.h_func,(y,),(y,))[0]+torch.func.jvp(self.g_func,(y,),(y,))[0]#+nu.T@self.h_func_grad(y)  + lam.T @ self.g_func_grad(y)
            #r=time.time()
            #print(r-s)
        else:
            L_grad_y_val = self.f_grad_func(y) + self.A.T @ nu + self.G.T @ lam

        # L_grad_y_val = self.f_grad_func(y) + nu @ self.A + lam @ self.G
        return L_grad_y_val
    def get_z_from_casadi(self,res):
        x=np.asarray(np.array_split(np.array(res['x'])[:self.n_x*self.N],self.n_x)).squeeze().T.reshape((self.N*self.n_x,)).tolist()
        u=np.asarray(np.array_split(np.array(res['x'])[self.n_x*self.N:self.n_x*self.N+self.n_u*self.N],self.n_u)).squeeze().T.reshape((self.N*self.n_u,)).tolist()
        y_opt = torch.tensor(x+u,device=self.device).squeeze()
        nu_opt = torch.tensor(np.array(res['lam_g'][0:self.n_eq]),device=self.device).squeeze()
        lam_opt = torch.tensor(np.array(res['lam_g'][self.n_eq:]),device=self.device).squeeze()
        z_opt = torch.hstack((y_opt,nu_opt,lam_opt))
        return z_opt

    def sample_casadi(self,n_samples):
        print("Sampling...")
        n_total = 0
        n_opt = 0
        z_opt_list = []
        x_list = []
        solve_time_list = []
        iter_count_list = []
        while n_opt < n_samples:
            x_take=self.batch_gen_x_new(1).squeeze()[0]
            x = x_take[0:self.n_x]
            res = self.casadi_solve(x)
            z_opt = self.get_z_from_casadi(res)
            status = self.ca_solver.stats()['return_status']
            solve_time = self.ca_solver.stats()['t_wall_total']
            iter_count = self.ca_solver.stats()['iter_count']
            n_total += 1
            if status == "Solve_Succeeded":
            # if torch.allclose(self.KKT_func(z_opt,x),torch.zeros_like(self.KKT_func(z_opt,x)),atol=1e-8):
                z_opt_list.append(z_opt)
                x_list.append(x_take)
                solve_time_list.append(solve_time)
                iter_count_list.append(iter_count)
                n_opt += 1
        z_opt_batch = torch.stack(z_opt_list)
        x_batch = torch.stack(x_list)

        print("Samples: ", n_total)
        print("Optimal Samples: ", n_opt)
        print("Sampling Done.")

        return z_opt_batch, x_batch, solve_time_list, iter_count_list
    
    # OSQP Solver
    def solve_osqp(self,x):
        """Solve QP using cvxpy.
        Returns solution vector y."""

        Q = self.Q.clone().cpu().numpy()
        p = self.p.clone().cpu().numpy()
        A = self.A.clone().cpu().numpy()
        G = self.G.clone().cpu().numpy()
        h = self.h.clone().cpu().numpy()

        x = x.clone().cpu().numpy()

        y = cp.Variable(self.n_vars)
        objective = cp.Minimize(0.5 * cp.quad_form(y, Q) + p.T @ y)
        constraints = [A @ y == x, G @ y <= h]
        prob = cp.Problem(objective, constraints)
        tic = time.perf_counter()
        prob.solve(solver=cp.OSQP)
        toc = time.perf_counter()
        full_solve_time = toc-tic

        y_opt = torch.as_tensor(y.value,device=self.device)
        nu_opt = torch.as_tensor(constraints[0].dual_value,device=self.device)
        lam_opt = torch.as_tensor(constraints[1].dual_value,device=self.device)

        status = prob.status

        solve_time = prob.solver_stats.solve_time
        num_iters = prob.solver_stats.num_iters

        z_opt = torch.hstack((y_opt,nu_opt,lam_opt))
        
        return z_opt, status, full_solve_time, solve_time, num_iters
    
    def sample_osqp_solutions(self,n_samples):
        print("Sampling...")
        n_total = 0
        n_opt = 0
        z_opt_list = []
        x_list = []
        full_solve_time_list = []
        solve_time_list = []
        iter_count_list = []
        while n_opt < n_samples:
            x = self.batch_gen_x(1).squeeze()
            z_opt, status, full_solve_time, solve_time, num_iters = self.solve_osqp(x)
            n_total += 1
            if status == "optimal":
                z_opt_list.append(z_opt)
                x_list.append(x)
                full_solve_time_list.append(full_solve_time)
                solve_time_list.append(solve_time)
                iter_count_list.append(num_iters)
                n_opt += 1
        z_opt_batch = torch.stack(z_opt_list)
        x_batch = torch.stack(x_list)

        print("Samples: ", n_total)
        print("Optimal Samples: ", n_opt)
        print("Sampling Done.")

        return z_opt_batch, x_batch, full_solve_time_list, solve_time_list, iter_count_list

    def ode(self, xk, uk):
        # states, inputs and parameters
        theta = xk[0]
        phi = xk[1]
        psi = xk[2]
        u_tilde = uk

        E = self.E_0 - self.c_tilde * u_tilde ** 2
        v_a = self.v_0 * E * torch.cos(theta)

        # Differential equations
        dphi = -v_a / (self.L_tether * torch.sin(theta)) * torch.sin(psi)

        d_theta = v_a / self.L_tether * (torch.cos(psi) - torch.tan(theta) / E)
        d_phi = dphi
        d_psi = v_a / self.L_tether * u_tilde + dphi * (torch.cos(theta))

        dxdt = torch.hstack([d_theta, d_phi, d_psi])
        return dxdt

    def ode_parallel(self, X, U,x):
        """Computes the system dynamics for batch of states, inputs and parameters."""
        assert X.shape[0] == U.shape[0]
        # assert P.shape[0] == X.shape[0]
        assert X.shape[1] == self.n_x
        assert U.shape[1] == self.n_u
        # assert P.shape[1] == self.np

        # states, inputs and parameters

        C_a=X[:, 0].squeeze()*2
        C_b=X[:, 1].squeeze()*2
        T_R=X[:, 2].squeeze() * 140
        T_K=X[:, 3].squeeze()*140
        F=U[:, 0].squeeze()*100



        # Certain parameters
        K0_ab = 1.287e12  # K0 [h^-1]
        K0_bc = 1.287e12  # K0 [h^-1]
        K0_ad = 9.043e9  # K0 [l/mol.h]
        R_gas = 8.3144621e-3  # Universal gas constant
        E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
        H_R_ab = 4.2  # [kj/mol A]
        H_R_bc = -11.0  # [kj/mol B] Exothermic
        H_R_ad = -41.85  # [kj/mol A] Exothermic
        Rou = 0.9342  # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
        A_R = 0.215  # Area of reactor wall [m^2]
        V_R = 10.01  # 0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0  # Temp of inflow [Celsius]
        K_w = 4032.0  # [kj/h.m^2.K]
        C_A0 = (5.7 + 4.5) / 2.0 * 1.0  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
        alpha = 1
        beta = 1
        T_dif=T_R - T_K
        T_in=x[self.n_x+4].repeat_interleave(self.N)*140#:self.n_x+self.N*self.n_p:self.n_p]*140
        C_A0=x[self.n_x+2].repeat_interleave(self.N)*5.7#:self.n_x+self.N*self.n_p:self.n_p]*5.7
        m_k=x[self.n_x+3].repeat_interleave(self.N)*6#:self.n_x+self.N*self.n_p:self.n_p]*6
        alpha=x[self.n_x].repeat_interleave(self.N)*1.05#:self.n_x+self.N*self.n_p:self.n_p]*1.05
        beta=x[self.n_x+1].repeat_interleave(self.N)*1.1#:self.n_x+self.N*self.n_p:self.n_p]*1.1
        H_R_ab=x[self.n_x+5].repeat_interleave(self.N)*4.6#:self.n_x+self.N*self.n_p:self.n_p]*4.6
        H_R_bc=-x[self.n_x+6].repeat_interleave(self.N)*12#:self.n_x+self.N*self.n_p:self.n_p]*12
        Q_dot=-8500*x[self.n_x+self.n_p:self.n_x+self.n_p+self.N]

        K_1 = beta * K0_ab * torch.exp((-E_A_ab) / ((T_R + 273.15)))#T_R
        K_2 = K0_bc * torch.exp((-E_A_bc) / ((T_R + 273.15)))
        K_3 = K0_ad * torch.exp((-alpha * E_A_ad) / ((T_R + 273.15)))
        # Fixed parameters:EQNWT
        dx1= F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2)
        dx2= -F * C_b + K_1 * C_a - K_2 * C_b
        dx3= ((K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a ** 2) * H_R_ad) / (-Rou * Cp)) + F * (
                                  T_in - T_R) + (((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R))
        dx4=(Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k)

        dxdt = (torch.stack([dx1, dx2,dx3,dx4]).T/torch.tensor(self.x_scaling))
        return dxdt

    def ode_cas(self, X, U,P):
        C_a = X[:, 0]*2#.squeeze()
        C_b = X[:, 1]*2#.squeeze()
        T_R = X[:, 2]*140#.squeeze()
        T_K = X[:, 3]*140#.squeeze()
        F = U[:, 0]*100#.squeeze()


        # Certain parameters
        K0_ab = 1.287e12  # K0 [h^-1]
        K0_bc = 1.287e12  # K0 [h^-1]
        K0_ad = 9.043e9  # K0 [l/mol.h]
        R_gas = 8.3144621e-3  # Universal gas constant
        E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
        H_R_ab = 4.2  # [kj/mol A]
        H_R_bc = -11.0  # [kj/mol B] Exothermic
        H_R_ad = -41.85  # [kj/mol A] Exothermic
        Rou = 0.9342  # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
        A_R = 0.215  # Area of reactor wall [m^2]
        V_R = 10.01  # 0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0  # Temp of inflow [Celsius]
        K_w = 4032.0  # [kj/h.m^2.K]
        C_A0 = (5.7 + 4.5) / 2.0 * 1.0  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
        alpha = 1
        beta = 1
        T_dif = T_R - T_K
        T_in = ca.repmat(P[4],self.N)*140
        C_A0 = ca.repmat(P[2],self.N)*5.7
        m_k = ca.repmat(P[3],self.N)*6
        alpha = ca.repmat(P[0],self.N)*1.05
        beta = ca.repmat(P[1],self.N)*1.1
        H_R_ab =ca.repmat(P[5],self.N)*4.6
        H_R_bc = -ca.repmat(P[6],self.N) * 12
        Q_dot=-8500*P[self.n_p:self.n_p+self.N]
        K_1 = beta * K0_ab * ca.exp((-E_A_ab) / ((T_R + 273.15)))
        K_2 = K0_bc * ca.exp((-E_A_bc) / ((T_R + 273.15)))
        K_3 = K0_ad * ca.exp((-alpha * E_A_ad) / ((T_R + 273.15)))
        # Fixed parameters:EQNWT
        dx1 = F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2)
        dx2 = -F * C_b + K_1 * C_a - K_2 * C_b
        dx3 = ((K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a ** 2) * H_R_ad) / (-Rou * Cp)) + F * (
                T_in - T_R) + (((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R))
        dx4 = (Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k)

        dxdt = (ca.horzcat(dx1, dx2,dx3,dx4).T/self.x_scaling).T
        return dxdt

    def sim_rk4(self, xk, uk):
        """Simulate the system using RK4 method."""
        dt = self.dt
        k1 = self.ode(xk, uk)
        k2 = self.ode(xk + dt / 2 * k1, uk)
        k3 = self.ode(xk + dt / 2 * k2, uk)
        k4 = self.ode(xk + dt * k3, uk)
        xk_next = xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk_next

        # rk4 simulation of the system in parallel on multiple starting points and inputs

    def sim_rk4_parallel(self, X, U,x):
        """Simulate the system using RK4 method, in parallel."""
        dt = self.dt
        k1 = self.ode_parallel(X, U,x)
        k2 = self.ode_parallel(X + dt / 2 * k1, U,x)
        k3 = self.ode_parallel(X + dt / 2 * k2, U,x)
        k4 = self.ode_parallel(X + dt * k3, U,x)
        X_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X_next

    def sim_rk4_cas(self, X, U,P):
        dt = self.dt
        k1 = self.ode_cas(X, U,P)
        k2 = self.ode_cas(X + dt / 2 * k1, U,P)
        k3 = self.ode_cas(X + dt / 2 * k2, U,P)
        k4 = self.ode_cas(X + dt * k3, U,P)
        X_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X_next

    def sim_trajectory(self, x0, U_traj):
        """Simulate the system using RK4 method."""
        xk = x0
        X_traj = [x0]
        N = U_traj.shape[0]
        for i in range(N):
            uk = U_traj[i]
            # for uk in U_traj:
            xk = self.sim_rk4(xk, uk)
            X_traj.append(xk)
        return torch.stack(X_traj)

    def eq_system(self, X_traj, U_traj,x):
        """Compute the equality constraint related to the continuity of the system dynamics in a multiple shooting approach."""
        assert X_traj.shape[0] == self.N + 1  # x0, x1, x2, ..., xN
        assert U_traj.shape[0] == self.N  # u0, u1, ..., uN-1
        assert X_traj.shape[1] == self.n_x
        assert U_traj.shape[1] == self.n_u

        X_traj_prev = X_traj[:-1, :]  # x0, x1, x2, ..., xN-1
        X_traj_next = self.sim_rk4_parallel(X_traj_prev, U_traj,x)  # x1+, x2+, ..., xN+

        assert X_traj_next.shape == X_traj_prev.shape
        assert X_traj_next.shape[0] == self.N  # x1+, x2+, ..., xN+

        # compute the equality constraint
        eq = X_traj_next - X_traj[1:, :]  # x1+ - x1, x2+ - x2, ..., xN+ - xN
        return eq

    def eq_system_cas(self, X_traj, U_traj,P):
        X_traj_prev = X_traj[:-1, :] # x0, x1, x2, ..., xN-1
        X_traj_next = self.sim_rk4_cas(X_traj_prev, U_traj,P)
        # compute the equality constraint
        eq = ((X_traj_next - X_traj[1:, :]))#.T/self.x_scaling).T  # x1+ - x1, x2+ - x2, ..., xN+ - xN
        return eq

    def ineq_states(self, xk):
        """Compute the inequality constraint related to the states."""
        # lb - xk <= 0
        # xk - ub <= 0
        assert xk.shape[0] == self.n_x
        lb = self.state_bounds[:, 0]
        ub = self.state_bounds[:, 1]
        ineq = torch.vstack([lb - xk, xk - ub])
        return ineq

    def ineq_states_traj(self, X_traj):
        """Compute the inequality constraint related to the states."""
        # lb - xk <= 0
        # xk - ub <= 0
        assert X_traj.shape[1] == self.n_x
        lb = self.state_bounds[:, 0]
        ub = self.state_bounds[:, 1]
        ineq = torch.vstack([lb - X_traj, X_traj - ub])
        # ordering: x0_0_lb, x0_1_lb, x0_2_lb, x1_0_lb, x1_1_lb, x1_2_lb, ..., xN_0_lb, xN_1_lb, xN_2_lb, x0_0_ub, x0_1_ub, x0_2_ub, x1_0_ub, x1_1_ub, x1_2_ub, ..., xN_0_ub, xN_1_ub, xN_2_ub
        # ordering: lower bounds for variables at each time step followed by upper bounds for variables at each time step
        return ineq.reshape(-1)

    def ineq_states_traj_cas(self, X_traj):
        """Compute the inequality constraint related to the states."""
        # lb - xk <= 0
        # xk - ub <= 0
        assert X_traj.shape[1] == self.n_x
        lb = self.state_bounds[:, 0]
        ub = self.state_bounds[:, 1]
        lb = lb.detach().cpu().numpy()
        ub = ub.detach().cpu().numpy()
        lb = np.array(
            np.split(np.repeat(lb, self.N), self.n_x)).T  # np.split(np.repeat(lb, self.N),2)#.reshape((-1, self.n_u))
        ub = np.array(np.split(np.repeat(ub, self.N), self.n_x)).T
        #X_traj=(X_traj.T/self.x_scaling).T
        ineq = ca.horzcat(lb - X_traj, X_traj - ub)
        # ordering: x0_0_lb, x0_1_lb, x0_2_lb, x1_0_lb, x1_1_lb, x1_2_lb, ..., xN_0_lb, xN_1_lb, xN_2_lb, x0_0_ub, x0_1_ub, x0_2_ub, x1_0_ub, x1_1_ub, x1_2_ub, ..., xN_0_ub, xN_1_ub, xN_2_ub
        # ordering: lower bounds for variables at each time step followed by upper bounds for variables at each time step
        return ineq.reshape((-1, 1))

    def ineq_inputs(self, uk):
        """Compute the inequality constraint related to the inputs."""
        # lb - uk <= 0
        # uk - ub <= 0
        assert uk.shape[0] == self.n_u
        lb = self.input_bounds[:, 0]
        ub = self.input_bounds[:, 1]
        ineq = torch.vstack([lb - uk, uk - ub])
        return ineq

    def ineq_inputs_traj(self, U_traj):
        """Compute the inequality constraint related to the inputs."""
        # lb - uk <= 0
        # uk - ub <= 0
        assert U_traj.shape[1] == self.n_u
        lb = self.input_bounds[:, 0]
        ub = self.input_bounds[:, 1]
        ineq = torch.vstack([lb - U_traj, U_traj - ub])
        # ordering: u0_lb, u1_lb, u2_lb, ..., uN-1_lb, u0_ub, u1_ub, u2_ub, ..., uN-1_ub
        return ineq.reshape(-1)

    def ineq_inputs_traj_cas(self, U_traj):
        """Compute the inequality constraint related to the inputs."""
        # lb - uk <= 0
        # uk - ub <= 0
        assert U_traj.shape[1] == self.n_u
        lb = self.input_bounds[:, 0]
        ub = self.input_bounds[:, 1]
        lb = lb.detach().cpu().numpy()
        ub = ub.detach().cpu().numpy()
        #U_traj = (U_traj.T / self.u_scaling).T
        lb = np.array(np.split(np.repeat(lb, self.N),self.n_u)).T#np.split(np.repeat(lb, self.N),2)#.reshape((-1, self.n_u))
        ub = np.array(np.split(np.repeat(ub, self.N),self.n_u)).T#np.repeat(ub, self.N).reshape((-1, self.n_u))
        ineq = ca.horzcat(lb - U_traj, U_traj - ub)
        # ordering: u0_lb, u1_lb, u2_lb, ..., uN-1_lb, u0_ub, u1_ub, u2_ub, ..., uN-1_ub
        return ineq.reshape((-1, 1))

    def ineq_full_traj(self, X_traj, U_traj):
        """Compute the inequality constraint related to the full trajectory."""
        assert X_traj.shape[0] == self.N + 1  # x0, x1, x2, ..., xN
        # assume that x0 is feasible and not a decision variable
        X_traj = X_traj[1:, :]
        ineq_x = self.ineq_states_traj(X_traj)
        ineq_u = self.ineq_inputs_traj(U_traj)
        ineq = torch.hstack([ineq_x, ineq_u])#ineq_x,
        return ineq

    def ineq_full_traj_cas(self, X_traj, U_traj):
        """Compute the inequality constraint related to the full trajectory."""
        # assert X_traj.shape[0] == self.N + 1  # x0, x1, x2, ..., xN
        # assume that x0 is feasible and not a decision variable
        X_traj = X_traj[1:, :]
        ineq_x = self.ineq_states_traj_cas(X_traj)
        ineq_u = self.ineq_inputs_traj_cas(U_traj)
        ineq = ca.vertcat(ineq_x,ineq_u)
        return ineq

    def traj_to_vars(self, X_traj, U_traj):
        """Convert the trajectory into individual variables."""
        x0 = X_traj[0, :]
        w = torch.hstack([X_traj[1:, :].reshape(-1), U_traj.reshape(-1)])
        return x0, w

    def vars_to_traj(self, w, x0):
        """Convert the individual variables into the trajectory."""
        X_traj = w[:self.n_x_traj - self.n_x].reshape(-1, self.n_x)
        U_traj = w[self.n_x_traj - self.n_x:].reshape(-1, self.n_u)
        X_traj = torch.vstack([x0, X_traj])
        return X_traj, U_traj

    def vars_to_traj_cas(self, w, x0):
        """Convert the individual variables into the trajectory."""
        X_traj = w[:self.n_x_traj - self.n_x].reshape((-1, self.n_x))
        U_traj = w[self.n_x_traj - self.n_x:].reshape((-1, self.n_u))
        X_traj = ca.vertcat(x0.T, X_traj)
        return X_traj, U_traj

    def stack_p(self, x0, u_prev):
        """Stack the parameters."""
        p = torch.hstack([x0, u_prev])
        return p

    def unstack_p(self, p):
        """Unstack the parameters."""
        x0 = p[:self.n_x]
        u_prev = p[self.n_x:]
        return x0, u_prev

    def stack_z(self, w, nu, lam):
        """Function to stack primal and dual variables to be used in the NLP."""
        # z = [w,nu,lam]
        assert w.shape[0] == self.nw
        assert nu.shape[0] == self.n_eq
        assert lam.shape[0] == self.n_ineq

        z = torch.hstack([w, nu, lam])
        return z

    def unstack_z(self, z):
        """Function to unstack primal and dual variables from the NLP solution."""
        w = z[:self.nw]
        nu = z[self.nw:self.nw + self.n_eq]
        lam = z[self.nw + self.n_eq:]
        return w, nu, lam

        #### NLP Functions

    def h_func(self, w,x):
        """Function to compute the equality constraints."""
        # x0, u_prev = self.unstack_p(p)
        # p_u=w[self.n_x_traj - self.n_x:].reshape(-1, self.n_u)
        X_traj, U_traj = self.vars_to_traj(w, x[0:4])

        # U_traj_1=torch.zeros_like(w[self.n_x_traj - self.n_x:].reshape(-1, self.n_u).clone())
        # U_traj_2=torch.ones_like(w[self.n_x_traj - self.n_x:].reshape(-1, self.n_u))
        # eq_1 = (self.eq_system(X_traj, U_traj_1)*(1-p_u)).reshape(-1)
        # eq_2= (self.eq_system(X_traj, U_traj_2)*p_u).reshape(-1)
        eq = self.eq_system(X_traj, U_traj,x).reshape(-1)
        return eq  # eq_1+eq_2

    def h_func_cas(self, y, x0,P):
        X_traj, U_traj = self.vars_to_traj_cas(y, x0)
        eq = self.eq_system_cas(X_traj, U_traj,P).reshape((-1, 1))
        return eq

    def g_func(self, w):
        """Function to compute the inequality constraints."""
        # x0, u_prev = self.unstack_p(p)
        X_traj, U_traj = self.vars_to_traj(w, self.p[0:self.n_x])
        ineq = self.ineq_full_traj(X_traj, U_traj)
        return ineq

    def g_func_cas(self, w, x0):
        X_traj, U_traj = self.vars_to_traj_cas(w, x0)
        ineq = self.ineq_full_traj_cas(X_traj, U_traj)
        return ineq