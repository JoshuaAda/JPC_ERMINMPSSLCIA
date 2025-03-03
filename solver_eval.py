# %%
# IMPORTS
import time
import torch
import matplotlib
from torch.utils.data import DataLoader,random_split,TensorDataset
import os
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from parametric_OP import parametricOP
from models import FeedforwardNN, Predictor, Solver
#from parametric_OP_with_z import parametricOPz
import numpy as np
#from nn_models import ExplicitSolverFramework, HistoryLogger, FeedforwardNN, generate_experiment_dir, weight_norms, count_params
from simulator_CSTR import template_simulator
from model_CSTR import template_model
import do_mpc
from parametric_OPz import parametricOPz
from models import FeedforwardNN
# %%
import json
# CONFIG
seed = 1#42
matplotlib.use('TkAgg')
op_name = "nonlinear_100x50x50_0" # "rosenbrock_1000x500x500_0"

# pths
folder_name = "parametric_OP_data"
file_pth = Path(__file__).parent.resolve()
op_pth = file_pth.joinpath(folder_name)




# %%
# PREPARE
np.random.seed(seed)
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.manual_seed(seed)
batched_dot = torch.func.vmap(torch.dot,chunk_size=None)
def get_all_configs(pth):
    # configs
    config_list = []
    op_names = [f for f in os.listdir(pth) if os.path.isdir(pth.joinpath(f))]
    for op_name in op_names:
        op_pth = pth.joinpath(op_name)
        for i in range(100):
            local_pth = Path(op_pth,f"exp_{i}")
            if local_pth.exists():
                config_pth = local_pth.joinpath("config.json")
                if config_pth.exists():
                    with open(config_pth, 'r') as f:
                        config = json.load(f)
                        config_list.append(config)
                else:
                    continue
    return config_list

def load_OP(pth):
    return parametricOP.from_json(pth,file_name="op_cfg")

def load_predictor(predictor_config):
    # load config
    n_in = predictor_config["n_in"]
    n_out = predictor_config["n_out"]
    n_layers = predictor_config["n_layers"]
    n_neurons = predictor_config["n_neurons"]
    act_fn = predictor_config["act_fn"]
    output_act_fn = predictor_config["output_act_fn"]
    # build model
    model = FeedforwardNN(n_in,n_out,n_layers,n_neurons,act_fn,output_act_fn)
    predictor = Predictor(model)
    # load weights
    predictor.load_weights(predictor_config["exp_pth"],file_name="predictor_weights.pt")
    predictor.model = predictor.model.to(dtype=dtype,device=device)
    return predictor

def load_predictor_from_pth(pth):
    with open(pth.joinpath("config.json"), 'r') as f:
        predictor_config = json.load(f)
    return load_predictor(predictor_config)

def load_solver(OP,solver_config):
    # NN
    n_in = solver_config["n_in"]
    n_out = solver_config["n_out"]
    n_neurons = solver_config["n_neurons"]
    n_layers = solver_config["n_layers"]
    act_fn = solver_config["act_fn"]
    output_act_fn = solver_config["output_act_fn"]
    model = FeedforwardNN(n_in=n_in,n_out=n_out,n_hidden_layers=n_layers,n_neurons=n_neurons,act_fn=act_fn,output_act_fn=output_act_fn)
    solver = Solver(model,OP)
    # load weights
    solver.load_weights(solver_config["exp_pth"],file_name="solver_weights.pt")
    solver.model = solver.model.to(dtype=dtype,device=device)
    return solver
def load_solver_from_pth(pth,OP):
    with open(pth.joinpath("config.json"), 'r') as f:
        solver_config = json.load(f)
    return load_solver(OP,solver_config)
# pths


OP=load_OP("parametric_OP_data_robust/nonlinear_96x48x192_0")
OPz=parametricOPz(obj_type = "nonlinear")

# NN
n_layers = 1 #(L = n+1)
n_neurons = 5000 # 512 #4096 #512 #512 #512
act_fn = "relu" # "relu", "tanh", "leaky_relu", "linear", "sigmoid"
output_act_fn = "linear" # "linear", "sigmoid"

n_in = 50 + OP.n_z + 1# 63
n_out = OP.n_z

# NN Training
N_epochs = 25000
batch_size = 1
weight_decay = 0.0 # 1e-3
lr = 1e-3

# Solver
solver_steps = 100
alpha =1e-2 #torch.Tensor([1e-2]).cuda()
Tk_lim = 1e-6
resampling_delay = 100
# gamma = 1.0 # 0.95
# gamma_sum = sum([gamma**(solver_steps-i) for i in range(solver_steps)])

# Verbosity
print_frequency = 10
# Numerics
offset = 1e-16

log_loss = True

# Postprocessing
save_run = False
log_dir = op_name
save_dir = file_pth.joinpath("implicit_solver_runs")
data_dir="./sampling"
# %%
# NN
model = FeedforwardNN(n_in=n_in,n_out=n_out,n_hidden_layers=n_layers,n_neurons=n_neurons,act_fn=act_fn,output_act_fn=output_act_fn)
#solver_nn = ExplicitSolverFramework(model,OP,mode="full")
path=Path(__file__).parent.resolve()
predictor=load_predictor_from_pth(path.joinpath("robust_case_study/predictor/nonlinear_96x48x192_0/exp_opt_1"))#324#312#313#310#305#244#255#244#245#240#194#168#194#191#157#143#123#96#105
pth=path.joinpath("robust_case_study/solver_with_pred/nonlinear_96x48x192_0/exp_opt_1")#482#464#470#461#451#418#376#385#376#377#360#311#356#311#300#260#292#255#225#219#184#203
solver=load_solver_from_pth(pth,OP)
predictor_z=load_predictor_from_pth(path.joinpath("robust_case_study/predictor/nonlinear_96x48x192_0/exp_opt_2"))#236#181#156
pth=path.joinpath("robust_case_study/solver_with_pred/nonlinear_96x48x192_0/exp_opt_2")#500#484#492#479#352#264#247#219#184#203
solver_z=load_solver_from_pth(pth,OPz)





# %%



N=20
model=template_model()
simulator=template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)
fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data, figsize=(8,5))
plt.ion()
show_animation=False
num_iter=10
num_time=1
num_run=1000
x_max=np.array([[2],[2],[140],[140]])
x_min=np.array([[0.1],[0.1],[50],[50]])
x_sam_max=np.array([[1.7],[1.7],[135],[130]])
x_sam_min=np.array([[0.3],[0.3],[100],[100]])
p_list=[]
x_ges_list=[]
u_ges_list=[]
cost_ges_list=[]
num_full_list=[]
num_last_list=[]
time_ges_list=[]
value_ges_list=[]
num_full_list_z=[]
num_last_list_z=[]
value_ges_list_z=[]
num_ges_cons_list=[]
Tk_hist=[]
runs=0
for k in range(num_run):
    x0=np.random.uniform(x_sam_min,x_sam_max)#np.array([[0.8],[0.5],[134],[130]])#=np.array([[0.79918283], [0.86885595], [0.9441837] , [0.83149743]])*np.array([[2],[2],[140],[140]])#np.random.uniform(x_min,x_max)##np.random.uniform(x_min,x_max)#np.array([[0.8],[0.5],[134.14],[130.0]])#np.random.uniform(np.array([[0.1],[0.1],[50],[50]]),np.array([[2],[2],[140],[140]]))#np.array([[0.8],[0.5],[134.14],[130.0]])
    u0=np.array([[5],[-8500]])#np.concatenate((np.random.uniform(np.array([[5]]),np.array([[100]])),-8500*np.random.randint(np.array([[0]]),np.array([[2]]))))#np.array([[5],[-8500]])#-8500#np.array([[np.random.uniform(5,100)],[np.random.randint(0,2)*-8500]])#5,-8500
    u_app=u0
    simulator.x0=x0
    num_scen=len(OP.c_scen)**OP.N_r
    max_iter = 500#500
    max_iter_z = 100#200
    x_list=[]
    Tk_50_list=[]
    OP.setup_casadi_single()
    if OP.learning_big:
        approx_3 = FeedforwardNN(OP.n_x + OP.n_u, OP.n_u)
        approx_3.load_state_dict(torch.load("app_mpc.pth"))
        '''
        OP.init_small(approx_3)
        x_opt=[]
        u0_opt=[]
        for r in range(10):
            x0=np.random.uniform(np.array([[0.1], [0.1], [50], [50]]),
                              np.array([[2], [2], [140], [140]]))  # np.array([[0.8],[0.5],[134.14],[130.0]])
            u0 = np.array([[np.random.uniform(5,100)],[np.random.randint(0,2)*-8500]])#5,-8500

            for k in range(num_time):

                x_all = time.time()
                zk_batch = torch.rand(batch_size*num_scen,OP.n_z, device=device)#
                x0=torch.hstack([torch.repeat_interleave(torch.Tensor([x0[0][0]/2, x0[1][0]/2,x0[2][0]/140, x0[3][0]/140]).reshape(1, 4), batch_size*num_scen,
                                             dim=0)]).to(device=device)
                uapp_old = torch.hstack(
                    [torch.repeat_interleave(torch.Tensor([u_app[0][0]/100, -u_app[1][0]/8500]).reshape(1, 2), batch_size*num_scen,
                                             dim=0)]).to(device=device)

                u0_old = torch.hstack(
                    [torch.repeat_interleave(torch.Tensor([u0[0][0] / 100, -u0[1][0] / 8500]).reshape(1, 2),
                                             batch_size * num_scen,
                                             dim=0)]).to(device=device)
                #uapp_old=u0_old
                u_tilde = u0_old#torch.hstack(
                    #[torch.repeat_interleave(torch.Tensor([0.05, 0]).reshape(1, 2),
                    #                         batch_size * num_scen,
                    #                         dim=0)]).to(device=device)




                xk_batch = OP.batch_gen_x_new_cas(batch_size, x0, uapp_old)
                num_scenarios = len(OP.c_scen) ** OP.N_r
                #for s in range(1):
                u = np.zeros((len(OP.c_scen) ** OP.N_r, OP.N_r * OP.n_u))
                u_ges = np.zeros((len(OP.c_scen) ** OP.N_r, OP.N * OP.n_u))

                # Gather the results
                # for m,result in enumerate(results):
                # res = result

                # u_traj=np.array(np.split(np.array([res['x'][OP.n_x * OP.N:OP.n_x * OP.N + OP.n_u*OP.N].full()]).squeeze(),OP.n_u)).T
                # u0 = u_traj[0].reshape((OP.n_u, 1))
                # u[m] = u_traj[0:OP.N_r].reshape((OP.n_u*OP.N_r,))
                # u_ges[m] = u_traj.reshape((OP.N*OP.n_u,))
                # for m in range(num_scenarios):
                # x=xk_batch[m]
                x = time.time()
                res = OP.casadi_solve(xk_batch)
                if not OP.ca_solver.stats()['success']:
                    break
                y = time.time()
                print(y - x)
                # for m in range(num_scenarios):
                #    res=results[m]
                u_traj = np.array(
                    np.split(np.array([res['x'][OP.n_x * OP.N:OP.n_x * OP.N + OP.n_u * OP.N].full()]).squeeze(), OP.n_u)).T
                print()
                u0 = u_traj[0].reshape((OP.n_u, 1))
                u = u_traj[0:OP.N_r].reshape((OP.n_u * OP.N_r,))
                u_ges = u_traj.reshape((OP.N * OP.n_u,))
                if not OP.multistage:
                    xk_batch = OP.update_x_batch_cas(batch_size, u, x0, u0_old)

                u_int = np.zeros((OP.num_scen ** OP.n_p, OP.N))
                u = np.array(
                    [res['x'][OP.n_x * OP.N + OP.N + k * OP.n_vars:OP.n_x * OP.N + 2 * OP.N + k * OP.n_vars].full() for k in
                     range(OP.num_scen ** OP.n_p)])

                for m in range(OP.num_scen ** OP.n_p):

                    summe = 0
                    for i in range(len(u[0])):  # -1,-1,-1):
                        #    # print("k:"+str(k)+" i:"+str(i))
                        summe = summe + u[m, i]
                        if summe > 0.5:
                            summe -= 1
                            u_int[m, i] = 1
                ### Testen
                xk_batch2 = OPz.batch_gen_x_new_cas(batch_size, x0, uapp_old[:, 0], u_int)
                res2 = OPz.casadi_solve(xk_batch2)
                if not OPz.ca_solver.stats()['success']:
                    break
                u0 = np.array([res['x'][80].full()[0], res['x'][100].full()[0]]) * np.array([[100], [-8500]])
                uapp_old = u_app
                u_app = np.array([[res2['x'][80].full()[0][0]], [u_int[0][0]]]) * np.array([[100], [-8500]])
                x0_curr=np.array(x0[0,:].detach().cpu().T)
                x_curr=np.concatenate((x0_curr.reshape((-1,1)),uapp_old))
                if OP.ca_solver.stats()['success'] and OPz.ca_solver.stats()['success']:
                    x_opt.append(x_curr)
                    u0_opt.append(u_app)
                p_num = simulator.get_p_template()
                p_num['alpha'] = np.random.uniform(0.95, 1.05)
                p_num['beta'] = np.random.uniform(0.9, 1.1)
                p_num['C_A0'] = np.random.uniform(4.5, 5.7)
                p_num['T_in'] = np.random.uniform(120, 140)
                p_num['m_k'] = np.random.uniform(4, 6)
                p_num['H_R_ab'] = np.random.uniform(3.8, 4.6)
                p_num['H_R_bc'] = np.random.uniform(-12, -10)


                def p_fun(t_now):
                    return p_num


                simulator.set_p_fun(p_fun)
                y_next = simulator.make_step(u_app)
                x0 = estimator.make_step(y_next)

        x_opt = torch.tensor(x_opt, device=OP.device, dtype=torch.float32).squeeze()
        u0_opt = torch.tensor(u0_opt, device=OP.device, dtype=torch.float32).squeeze()
        data = TensorDataset(x_opt, u0_opt)
        data_dir = Path(data_dir)
        torch.save(data, data_dir.joinpath('data_n' + str(250) + '_opt.pth'))
        
        OP.default_training(n_samples=250,n_epochs=10)
        torch.save(approx_3.state_dict(), "model_4.pth")
        '''

    u=np.array([[-8.12521285e-09 ,-7.73606113e-09 ,-6.65586438e-09 ,-2.52201699e-09,  8.87578832e-02 , 9.99999972e-01 , 9.99999952e-01,  9.99443734e-01,  4.97773547e-01 , 3.15338343e-01 , 1.96426109e-01 , 1.20031194e-01,  7.24789108e-02 , 4.34530739e-02 , 2.59412196e-02 , 1.54494593e-02,  9.19048179e-03,  5.46748878e-03,  3.25877616e-03 , 1.95676924e-03,  1.14130145e-03 , 6.81675679e-04  ,5.03706138e-04,  5.17466214e-04]]).reshape((24,1))
    x1=np.array([[0.60350562 ,0.77014999, 1.01003724 ,1.32216325 ,1.63453204, 1.56819683, 1.40707013, 1.20893782, 1.12759772, 1.07732449, 1.0464358, 1.02774117, 1.01652375, 1.00982534 ,1.0058364,  1.00346467, 1.00205569, 1.00121895 ,1.00072198 ,1.00042599 ,1.00025809 ,1.00015982,1.00009,1.00001 ]]).reshape((24,1))
    x2=np.array([[0.55809,0.47583,0.44896,0.48661,0.61340,0.75317,0.87224,0.92073,0.95343,0.97272,0.98398,0.99056,0.99442,0.99670,0.99804,0.99884,0.99931,0.99959,0.99975,0.99984,0.99990,0.99994,0.99995,0.99992]]).reshape((24,1))
    time_list=[]
    cost_list=[]
    u_list=[]
    num_full_z=0
    num_full = 0
    num_last=0
    num_last_z=0
    cons=0
    conv=False


    with torch.no_grad():
        value_list=[]
        value_list_z=[]

        for k in range(num_time):

            x_all = time.time()
            zk_batch = torch.rand(batch_size*num_scen,OP.n_z, device=device)#
            x0=torch.hstack([torch.repeat_interleave(torch.Tensor([x0[0][0]/2, x0[1][0]/2,x0[2][0]/140, x0[3][0]/140]).reshape(1, 4), batch_size*num_scen,
                                         dim=0)]).to(device=device)
            uapp_old = torch.hstack(
                [torch.repeat_interleave(torch.Tensor([u_app[0][0]/100, -u_app[1][0]/8500]).reshape(1, 2), batch_size*num_scen,
                                         dim=0)]).to(device=device)

            u0_old = torch.hstack(
                [torch.repeat_interleave(torch.Tensor([u0[0][0] / 100, -u0[1][0] / 8500]).reshape(1, 2),
                                         batch_size * num_scen,
                                         dim=0)]).to(device=device)
            #uapp_old=u0_old
            u_tilde = u0_old#torch.hstack(
                #[torch.repeat_interleave(torch.Tensor([0.05, 0]).reshape(1, 2),
                #                         batch_size * num_scen,
                #                         dim=0)]).to(device=device)




            if OP.solver_learning:
                t=0
                xk_batch = OP.batch_gen_x_new(batch_size,x0,uapp_old,u_tilde)
                if conv:
                    Tk = OP.Tk_conv_func(zk_batch[0], xk_batch[0])
                    # print(Tk)
                    Tk_batch, Fk_batch = OP.Tk_Fk_conv_batch_func(zk_batch, xk_batch)
                else:
                    Tk_batch,Fk_batch = OP.Tk_Fk_batch_func(zk_batch, xk_batch)

                zk_batch = predictor.forward(xk_batch)
                for s in range(num_iter):

                    mask3=((Tk_batch>1e-2) & (s>0))

                    zk_batch[mask3]=predictor.forward(xk_batch[mask3])

                    if conv:
                        OP.Tk_Fk_conv_func(zk_batch[0], xk_batch[0])
                        Tk_batch, Fk_batch = OP.Tk_Fk_conv_batch_func(zk_batch, xk_batch)
                    else:

                        Tk_batch,Fk_batch = OP.Tk_Fk_batch_func(zk_batch, xk_batch)
                    count=0

                    x= time.time()
                    count=torch.zeros(len(Tk_batch))
                    alpha=torch.ones(len(Tk_batch),1)

                    Tk_hist.append(np.array(Tk_batch.cpu().detach()).tolist())
                    for m in range(max_iter):

                        if m==max_iter-1:
                            value_list.append(sum(Tk_batch < 1e-6) / 2187)
                        if all(Tk_batch<1e-6):
                            print('yes')
                            num_full += 1
                            value_list.append(sum(Tk_batch < 1e-6) / 2187)
                            break

                        dzk_batch = solver.forward_fk(zk_batch,xk_batch,Fk_batch)*alpha
                        zk1_batch = zk_batch + dzk_batch
                        mask4 = (zk1_batch[:, 80:120] > 1)
                        zk1_batch[:, 80:120][mask4] = 1
                        mask5 = (zk1_batch[:, 81:120:2] < 0)
                        zk1_batch[:, 81:120:2][mask5] = 0



                        if conv:
                            Tk_new, Fk_new = OP.Tk_Fk_batch_func(zk1_batch, xk_batch)
                        else:
                            Tk_new,Fk_new= OP.Tk_Fk_batch_func(zk1_batch, xk_batch)

                        mask=(Tk_new<Tk_batch)
                        count[mask]=0
                        alpha[mask]=1
                        alpha[~mask]=alpha[~mask]*0.9
                        count[~mask]=count[~mask]+1
                        mask2=(count<4) & (Tk_new<0.5)
                        zk_batch[mask2] = zk1_batch[mask2]
                        Tk_batch[mask2]=Tk_new[mask2]
                        Fk_batch[mask2]=Fk_new[mask2]

                    y = time.time()
                    print(y-x)
                    t+=(y-x)
                    Tk = torch.median(Tk_batch)
                    print(Tk_batch)
                    if conv:
                        OP.Tk_conv_func(zk_batch[0], xk_batch[0])
                    else:
                        OP.Tk_func(zk_batch[0], xk_batch[0])
                    xk_batch = OP.update_x_batch(batch_size, zk_batch.detach().clone(), xk_batch.detach().clone(),Tk_batch)
                    if s == num_iter - 1:
                        if all(Tk_batch<1e-6):
                            num_last+=1
                        else:
                            p_list.append(np.array(xk_batch[0,:].cpu().detach()))

                u=zk_batch[:,OP.n_x*OP.N+1:(OP.n_x+OP.n_u)*OP.N:2]
                u_int = torch.zeros_like(u)

                # Iterate along the second axis (columns) for all rows
                summe = torch.zeros(u.shape[0], device=u.device)  # Keeps track of cumulative sums per row

                for i in range(u.shape[1]):
                    summe += u[:, i]  # Add current column values to cumulative sum
                    exceeds_threshold = summe > 0.5  # Check where cumulative sum exceeds threshold
                    u_int[exceeds_threshold, i] = 1  # Set u_int where condition is met
                    summe[exceeds_threshold] -= 1


                xk_batch_z = OPz.batch_gen_x_new(batch_size, x0, uapp_old[:,0:1],u_int)
                zk_batch_z = torch.rand(batch_size*num_scen,OPz.n_z, device=device)

                Tk = OPz.Tk_func(zk_batch_z[0], xk_batch_z[0])
                cond = OPz.cond_num_func(zk_batch_z[0], xk_batch_z[0])
                x = time.time()
                zk_batch_z=predictor_z.forward(xk_batch_z)
                Tk_batch_z, Fk_batch_z = OPz.Tk_Fk_batch_func(zk_batch_z, xk_batch_z)
                for s in range(num_iter):

                    mask3 = ((Tk_batch_z > 1e-2) & (s > 0))
                    zk_batch_z[mask3] = predictor_z.forward(xk_batch_z[mask3])

                    OPz.Tk_Fk_func(zk_batch_z[0], xk_batch_z[0])
                    Tk_batch_z, Fk_batch_z = OPz.Tk_Fk_batch_func(zk_batch_z, xk_batch_z)

                    x = time.time()

                    count = torch.zeros(len(Tk_batch_z))
                    alpha = torch.ones(len(Tk_batch_z), 1)
                    for m in range(max_iter_z):

                        if m == max_iter_z - 1:
                            value_list_z.append(sum(Tk_batch_z < 1e-5) / 2187)
                        if all(Tk_batch_z < 1e-5):
                            num_full_z += 1
                            value_list_z.append(sum(Tk_batch < 1e-5) / 2187)
                            print('yes')
                            break
                        dzk_batch_z = solver_z.forward_fk(zk_batch_z, xk_batch_z, Fk_batch_z)
                        zk1_batch_z = zk_batch_z + alpha*dzk_batch_z

                        Tk_new_z, Fk_new_z = OPz.Tk_Fk_batch_func(zk1_batch_z, xk_batch_z)

                        mask_z = (Tk_new_z < Tk_batch_z)


                        alpha[mask_z] = 1
                        alpha[~mask_z] = alpha[~mask_z] * 0.9
                        count[~mask_z] = count[~mask_z] + 1
                        mask2 = (count < 4) & (Tk_new_z < 0.5)
                        zk_batch_z[mask2] = zk1_batch_z[mask2]
                        Tk_batch_z[mask2] = Tk_new_z[mask2]
                        Fk_batch_z[mask2] = Fk_new_z[mask2]

                    y = time.time()
                    print(y - x)
                    t += (y - x)
                    Tk = OPz.Tk_func(zk_batch_z[0], xk_batch_z[0])
                    print(torch.median(Tk_batch_z))
                    xk_batch_z = OPz.update_x_batch(batch_size, zk_batch_z.detach().clone(), xk_batch_z.detach().clone(), u0_old[:,0:1],u_int,Tk_batch_z.detach().clone())
                    if s == num_iter - 1:
                        if all(Tk_batch_z<1e-5):
                            num_last_z+=1
                        else:
                            print('Hi')
                            p_list.append(np.array(xk_batch_z[0,:].cpu().detach()))

                y_all = time.time()
                u0 = np.array(OP.u_tilde[0].cpu().detach()).reshape((OP.n_u, 1)) * np.array([[100], [-8500]])
                u2=np.array(OPz.u_tilde[0].cpu().detach()).reshape((OPz.n_u, 1)) * np.array([[100]])
                #uapp_old=u_app
                u_app = np.array([[u2[0][0]],[(u0[1][0]<-4250)*-8500]])

            if OP.multistage:

                xk_batch = OP.batch_gen_x_new_cas(batch_size, x0,uapp_old)
                num_scenarios=len(OP.c_scen)**OP.N_r
                for s in range(1):
                    u=np.zeros((len(OP.c_scen)**OP.N_r,OP.N_r*OP.n_u))
                    u_ges=np.zeros((len(OP.c_scen)**OP.N_r,OP.N*OP.n_u))
                    x=time.time()
                    res = OP.casadi_solve(xk_batch)
                    y=time.time()
                    print(y-x)

                    u_traj=np.array(np.split(np.array([res['x'][OP.n_x * OP.N:OP.n_x * OP.N + OP.n_u*OP.N].full()]).squeeze(),OP.n_u)).T

                    u0 = u_traj[0].reshape((OP.n_u, 1))
                    u = u_traj[0:OP.N_r].reshape((OP.n_u*OP.N_r,))
                    u_ges = u_traj.reshape((OP.N*OP.n_u,))
                    if not OP.multistage:
                        xk_batch=OP.update_x_batch_cas(batch_size,u,x0,u0_old)

                u_int = np.zeros((OP.num_scen ** OP.n_p, OP.N))
                u = np.array(
                    [res['x'][OP.n_x * OP.N + OP.N + k * OP.n_vars:OP.n_x * OP.N + 2 * OP.N + k * OP.n_vars].full() for k in
                     range(OP.num_scen ** OP.n_p)])

                for m in range(OP.num_scen ** OP.n_p):

                    summe = 0
                    for i in range(len(u[0])):  # -1,-1,-1):
                        #    # print("k:"+str(k)+" i:"+str(i))
                        summe = summe + u[m, i]
                        if summe > 0.5:
                            summe -= 1
                            u_int[m, i] = 1
                ### Testen
                xk_batch2 = OPz.batch_gen_x_new_cas(batch_size, x0, uapp_old[:, 0], u_int)
                res2 = OPz.casadi_solve(xk_batch2)
                u0 = np.array([res['x'][80].full()[0], res['x'][100].full()[0]]) * np.array([[100], [-8500]])
                uapp_old=u_app
                u_app = np.array([[res2['x'][80].full()[0][0]], [u_int[0][0]]]) * np.array([[100], [-8500]])


            if OP.learning_big:
                x_s=x0[0,:]
                u_s=uapp_old[0,:]
                x_curr=torch.concatenate((x_s,u_s))
                u_app=approx_3(x_curr)

                u_app=np.array(u_app.detach().cpu()).reshape((-1,1))
                u_app[1]=u_app[1]>0.5
                u_app= u_app*np.array([[100],[-8500]])
            y_all=time.time()

            time_list.append(y_all-x_all)
            print(u0)
            print(u_app)

            p_num = simulator.get_p_template()
            p_num['alpha'] = np.random.uniform(0.95,1.05)
            p_num['beta'] = np.random.uniform(0.9,1.1)
            p_num['C_A0'] = np.random.uniform(4.5,5.7)
            p_num['T_in'] = np.random.uniform(120,140)
            p_num['m_k'] = np.random.uniform(4,6)
            p_num['H_R_ab'] = np.random.uniform(3.8,4.6)
            p_num['H_R_bc'] = np.random.uniform(-12, -10)


            def p_fun(t_now):
                return p_num


            simulator.set_p_fun(p_fun)
            y_next=simulator.make_step(u_app)
            x0=estimator.make_step(y_next)
            u_list.append(u0[0][0])
            u_list.append(u0[1][0])
            u_list.append(u_app[0][0])
            u_list.append(u_app[1][0])
            x_list.append(x0[0][0])
            x_list.append(x0[1][0])
            x_list.append(x0[2][0])
            x_list.append(x0[3][0])
            if np.any(x0>x_max) or np.any(x0<x_min):
                cons+=1
            cost_list.append((x0[1][0]/2-0.3)**2+(x0[0][0]/2-0.35)**2+0.1*(u0[0][0]/100-u0_old[0][0]/100)**2)
            if show_animation:
                graphics.plot_results(t_ind=k)

                graphics.reset_axes()
                plt.show()
                plt.pause(0.01)
        value_ges_list_z.append(sum(value_list_z)/len(value_list_z))
        num_ges_cons_list.append(cons)
        value_ges_list.append(sum(value_list)/len(value_list))
        time_ges_list.append(sum(time_list)/len(time_list))
        cost_ges_list.append(sum(cost_list))
        num_full_list.append(num_full/(num_iter*num_time))
        num_full_list_z.append(num_full_z/(num_iter*num_time))
        num_last_list.append(num_last/(num_time))
        num_last_list_z.append(num_last_z/num_time)

print('Mean Time: '+str((sum(time_ges_list))/(len(time_ges_list))))
print('Stage Cost: '+str(sum(cost_ges_list)/len(cost_ges_list)))
print('Percentage of overall complete convergence first problem: '+str(sum(num_full_list)/len(num_full_list)))
print('Percentage of overall complete convergence second problem: '+str(sum(num_full_list_z)/len(num_full_list_z)))
print('Percentage of last complete convergence first problem: '+str(sum(num_last_list)/len(num_last_list)))
print('Percentage of last complete convergence second problem: '+str(sum(num_last_list_z)/len(num_last_list_z)))
print('Percentage of total convergence first problem: '+str(sum(value_ges_list)/len(value_ges_list)))
print('Percentage of total convergence second problem: '+str(sum(value_ges_list_z)/len(value_ges_list_z)))
print('Number of constraint violations: '+str(sum(num_ges_cons_list)))
