run_folder = "robust_case_study"
op_folder_name = "parametric_OP_data_robust"

mode = "predictor"
# mode = "solver_no_pred" 
# mode = "solver_with_pred" 

speed_eval = True
overwrite = False

MAX_ITER = 500
NUM_ITERATIONS = 100 # for inference time evaluation (how often to repeat)
Tk_lim=[1e-6,1e-8,1e-10] # for solver evaluation
KKT_lim=[1e-3,1e-4,1e-5,1e-6] # for solver evaluation

# Map update modes from training to evaluation modes ("line_search" is basically an extension of "eta_total")
# In this paper, only "eta_total" for training and correspondingly "line_search" for evaluation is used.
UPDATE_MODES = {"eta_total": "line_search",
                "eta": "line_search",
                "full": "line_search",
                "gamma": "gamma"}


specific_op = ["nonlinear_96x48x192_0"]
specific_exp = None


# Imports
import json
import os
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from parametric_OP import parametricOP
from models import FeedforwardNN, Predictor, Solver

seed = 42
dtype = torch.float32
# dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(seed)
file_pth = Path(__file__).parent.resolve()

# Functions
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

def get_metrics(torch_tensor,mode="default"):
    # assert torch_tensor.dim() == 1
    if mode == "default":
        # min, 1%, 50%, 99%, max, mean, std
        return torch.min(torch_tensor).item(), torch.quantile(torch_tensor,0.01).item(), torch.median(torch_tensor).item(), torch.quantile(torch_tensor,0.99).item(), torch.max(torch_tensor).item(), torch.mean(torch_tensor).item(), torch.std(torch_tensor).item()
    elif mode == "large":
        # min, 1%, 50%, 90%, 95%, 99%, max, mean, std
        return torch.min(torch_tensor).item(), torch.quantile(torch_tensor,0.01).item(), torch.median(torch_tensor).item(), torch.quantile(torch_tensor,0.9).item(), torch.quantile(torch_tensor,0.95).item(), torch.quantile(torch_tensor,0.99).item(), torch.max(torch_tensor).item(), torch.mean(torch_tensor).item(), torch.std(torch_tensor).item()
    elif mode == "small":
        # min, 50%, max, mean, std
        return torch.min(torch_tensor).item(), torch.median(torch_tensor).item(), torch.max(torch_tensor).item(), torch.mean(torch_tensor).item(), torch.std(torch_tensor).item()
    else:
        raise ValueError("Invalid mode.")
    
# inference time
def measure_inference_time_predictor(model, input_data, use_cuda=True):
    # Set the model to evaluation mode
    model.eval()
    
    # Move model and input data to GPU if necessary
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        input_data = input_data.cuda()
    else:
        model = model.cpu()
        input_data = input_data.cpu()
    
    # Tensor to store inference times
    inference_times = torch.zeros(NUM_ITERATIONS)

    # Measure inference time
    with torch.no_grad():
        # Perform warm-up iterations (not measured)
        for _ in range(100):
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        for i in range(NUM_ITERATIONS):
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            start_time = time.perf_counter()
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            end_time = time.perf_counter()            
            inference_times[i] = end_time - start_time
    
    # Calculate statistics
    min_time, perc1_time, med_time, perc99_time, max_time, mean_time, std_dev_time = get_metrics(inference_times)
    
    # Store results in a dictionary
    results = {
        "max_time": max_time,
        "perc99_time": perc99_time,
        "med_time": med_time,
        "min_time": min_time,
        "mean_time": mean_time,
        "std_dev_time": std_dev_time
    }
    
    return results

def measure_step_time_solver(solver_step_func, zk, x, use_cuda=True):    
    # Move model and input data to GPU if necessary
    if use_cuda and torch.cuda.is_available():
        zk = zk.cuda()
        x = x.cuda()
    else:
        zk = zk.cpu()
        x = x.cpu()
    
    # Tensor to store inference times
    inference_times = torch.zeros(NUM_ITERATIONS)

    # Measure inference time
    with torch.no_grad():
        # Perform warm-up iterations (not measured)
        for _ in range(100):
            _ = solver_step_func(zk,x)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        for i in range(NUM_ITERATIONS):
            start_time = time.perf_counter()
            _ = solver_step_func(zk,x)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            end_time = time.perf_counter()            
            inference_times[i] = end_time - start_time
    
    # Calculate statistics
    # min, 1%, 50%, 99%, max, mean, std
    min_time, perc1_time, med_time, perc99_time, max_time, mean_time, std_dev_time = get_metrics(inference_times)
    
    # Store results in a dictionary
    results = {
        "max_step_time": max_time,
        "perc99_time": perc99_time,
        "med_step_time": med_time,
        "min_step_time": min_time,
        "mean_step_time": mean_time,
        "std_dev_step_time": std_dev_time
    }
    
    return results

def measure_total_time_solver_cpu_single(solver,solver_step_func_cpu, x_batch, predictor_cpu=None,repeat=1,max_iter=200,tol=1e-16,update_mode="line_search"):
    x_batch = x_batch.cpu()
    n_data = x_batch.shape[0]

    iter_count = []
    solve_time = []
    rel_step_time = []
    success_list = []
    for i in range(n_data):
        x_i_cpu = x_batch[i,:]
        for r in range(repeat):
            start_time = time.perf_counter()
            _,n_iter,success = solver.solve_fast_single(solver_step_func_cpu,x_i_cpu,predictor=predictor_cpu,max_iter=max_iter,tol=tol,use_cuda=False,update_mode=update_mode)
            end_time = time.perf_counter()
            if success:
                iter_count.append(n_iter)
                solve_time.append(end_time-start_time)
                rel_step_time.append((end_time-start_time)/n_iter)
            success_list.append(success)
    success_rate = torch.sum(torch.tensor(success_list)).item()/len(success_list)
    if success_rate == 0.0:
        iter_count = [max_iter]
        solve_time = [100.0]
        rel_step_time = [100.0]
    iter_count = torch.tensor(iter_count).to(torch.float)
    solve_time = torch.tensor(solve_time)
    rel_step_time = torch.tensor(rel_step_time)

    # metrics
    ic_min, ic_50, ic_max, ic_mean, ic_std = get_metrics(iter_count,mode="small")
    st_min, st_50, st_max, st_mean, st_std = get_metrics(solve_time,mode="small")
    rst_min, rst_50, rst_max, rst_mean, rst_std = get_metrics(rel_step_time,mode="small")

    results = {"iter_count_min": ic_min, "iter_count_50": ic_50, "iter_count_max": ic_max, "iter_count_mean": ic_mean, "iter_count_std": ic_std,
                "solve_time_min": st_min, "solve_time_50": st_50, "solve_time_max": st_max, "solve_time_mean": st_mean, "solve_time_std": st_std,
                "rel_step_time_min": rst_min, "rel_step_time_50": rst_50, "rel_step_time_max": rst_max, "rel_step_time_mean": rst_mean, "rel_step_time_std": rst_std,
                "success_rate": success_rate}
    return results
    
# evaluation
def evaluate_predictor(predictor,OP,test_data):
    # 1. load data and data metrics
    x_batch = test_data["x"].to(dtype)
    n_data = x_batch.shape[0]
    z_opt_batch = test_data["z_opt"].to(dtype)
    y_opt_batch = z_opt_batch[:,:OP.n_vars]
    nu_opt_batch = z_opt_batch[:,OP.n_vars:OP.n_vars+OP.n_eq]
    lam_opt_batch = z_opt_batch[:,OP.n_vars+OP.n_eq:]
    test_data_dict = {"n_test_data": n_data}
    if OP.obj_type == "quad":
        full_solve_time = torch.tensor(test_data["full_solve_time"])
        fst_min, fst_50, fst_max, fst_mean, fst_std = get_metrics(full_solve_time,mode="small")
        test_data_dict["opt_full_solve_time_min"] = fst_min
        test_data_dict["opt_full_solve_time_50"] = fst_50
        test_data_dict["opt_full_solve_time_max"] = fst_max
        test_data_dict["opt_full_solve_time_mean"] = fst_mean
        test_data_dict["opt_full_solve_time_std"] = fst_std
        test_data_dict["test_solver"] = "OSQP"
    else:
        test_data_dict["test_solver"] = "IPOPT"
    solve_time = torch.tensor(test_data["solve_time"])
    st_min, st_50, st_max, st_mean, st_std = get_metrics(solve_time,mode="small")
    test_data_dict["opt_solve_time_min"] = st_min
    test_data_dict["opt_solve_time_50"] = st_50
    test_data_dict["opt_solve_time_max"] = st_max
    test_data_dict["opt_solve_time_mean"] = st_mean
    test_data_dict["opt_solve_time_std"] = st_std
    iter_count = torch.tensor(test_data["iter_count"]).to(torch.float)
    ic_min, ic_50, ic_max, ic_mean, ic_std = get_metrics(iter_count,mode="small")
    test_data_dict["opt_iter_count_min"] = ic_min
    test_data_dict["opt_iter_count_50"] = ic_50
    test_data_dict["opt_iter_count_max"] = ic_max
    test_data_dict["opt_obj_val"] = OP.f_batch_func(y_opt_batch,x_batch).mean().item()

    # 2. predict
    with torch.no_grad():
        z_hat_batch = predictor(x_batch)
        # split
        y_hat_batch = z_hat_batch[:,:OP.n_vars]
        nu_hat_batch = z_hat_batch[:,OP.n_vars:OP.n_vars+OP.n_eq]
        lam_hat_batch = z_hat_batch[:,OP.n_vars+OP.n_eq:]

    # 3. evaluate
    # 3.1 optimality conditions
    KKT_batch = OP.KKT_batch_func(z_hat_batch,x_batch)
    KKT_inf_norm = torch.norm(KKT_batch,p=float("inf"),dim=1)
    KKT_inf_min, KKT_inf_1, KKT_inf_50, KKT_inf_90, KKT_inf_95, KKT_inf_99, KKT_inf_max, KKT_inf_mean, KKT_inf_std = get_metrics(KKT_inf_norm,mode="large")
    Tk_batch = OP.Tk_batch_func(z_hat_batch,x_batch)
    Tk_min, Tk_1, Tk_50, Tk_90, Tk_95, Tk_99, Tk_max, Tk_mean, Tk_std = get_metrics(Tk_batch,mode="large")
    optimality_dict = {"KKT_inf_min": KKT_inf_min, "KKT_inf_1": KKT_inf_1, "KKT_inf_50": KKT_inf_50,"KKT_inf_90": KKT_inf_90,
                       "KKT_inf_95": KKT_inf_95,"KKT_inf_99": KKT_inf_99, "KKT_inf_max": KKT_inf_max, "KKT_inf_mean": KKT_inf_mean, "KKT_inf_std": KKT_inf_std,
                       "Tk_min": Tk_min, "Tk_1": Tk_1, "Tk_50": Tk_50, "Tk_90": Tk_90, "Tk_95": Tk_95, "Tk_99": Tk_99, "Tk_max": Tk_max, "Tk_mean": Tk_mean, "Tk_std": Tk_std}

    # 3.2 distance to data
    y_diff_batch = torch.norm(y_hat_batch - y_opt_batch,dim=1)
    nu_diff_batch = torch.norm(nu_hat_batch - nu_opt_batch,dim=1)
    lam_diff_batch = torch.norm(lam_hat_batch - lam_opt_batch,dim=1)
    y_diff_batch_scaled = y_diff_batch/(torch.norm(y_opt_batch,dim=1)+1e-16)
    nu_diff_batch_scaled = nu_diff_batch/(torch.norm(nu_opt_batch,dim=1)+1e-16)
    lam_diff_batch_scaled = lam_diff_batch/(torch.norm(lam_opt_batch,dim=1)+1e-16)

    y_diff_min, y_diff_1, y_diff_50, y_diff_99, y_diff_max, y_diff_mean, y_diff_std = get_metrics(y_diff_batch,mode="default")
    nu_diff_min, nu_diff_1, nu_diff_50, nu_diff_99, nu_diff_max, nu_diff_mean, nu_diff_std = get_metrics(nu_diff_batch,mode="default")
    lam_diff_min, lam_diff_1, lam_diff_50, lam_diff_99, lam_diff_max, lam_diff_mean, lam_diff_std = get_metrics(lam_diff_batch,mode="default")

    y_diff_scaled_min, y_diff_scaled_1, y_diff_scaled_50, y_diff_scaled_99, y_diff_scaled_max, y_diff_scaled_mean, y_diff_scaled_std = get_metrics(y_diff_batch_scaled,mode="default")
    nu_diff_scaled_min, nu_diff_scaled_1, nu_diff_scaled_50, nu_diff_scaled_99, nu_diff_scaled_max, nu_diff_scaled_mean, nu_diff_scaled_std = get_metrics(nu_diff_batch_scaled,mode="default")
    lam_diff_scaled_min, lam_diff_scaled_1, lam_diff_scaled_50, lam_diff_scaled_99, lam_diff_scaled_max, lam_diff_scaled_mean, lam_diff_scaled_std = get_metrics(lam_diff_batch_scaled,mode="default")

    distance_dict = {"y_diff_max": y_diff_max, "y_diff_99": y_diff_99, "y_diff_50": y_diff_50, "y_diff_min": y_diff_min,
                    "nu_diff_max": nu_diff_max, "nu_diff_99": nu_diff_99, "nu_diff_50": nu_diff_50, "nu_diff_min": nu_diff_min,
                    "lam_diff_max": lam_diff_max, "lam_diff_99": lam_diff_99, "lam_diff_50": lam_diff_50, "lam_diff_min": lam_diff_min,
                    "y_diff_scaled_max": y_diff_scaled_max, "y_diff_scaled_99": y_diff_scaled_99, "y_diff_scaled_50": y_diff_scaled_50, "y_diff_scaled_min": y_diff_scaled_min,
                    "nu_diff_scaled_max": nu_diff_scaled_max, "nu_diff_scaled_99": nu_diff_scaled_99, "nu_diff_scaled_50": nu_diff_scaled_50, "nu_diff_scaled_min": nu_diff_scaled_min,
                    "lam_diff_scaled_max": lam_diff_scaled_max, "lam_diff_scaled_99": lam_diff_scaled_99, "lam_diff_scaled_50": lam_diff_scaled_50, "lam_diff_scaled_min": lam_diff_scaled_min
    }

    # 3.3 constraint violations
    h_val_batch = OP.h_batch_func(y_hat_batch,x_batch)
    g_val_batch = OP.g_batch_func(y_hat_batch)        
    viol_h_batch = torch.abs(h_val_batch)
    viol_g_batch = torch.relu(g_val_batch)
    viol_lam_batch = torch.relu(-lam_hat_batch)

    viol_h_min, viol_h_1, viol_h_50, viol_h_99, viol_h_max, viol_h_mean, viol_h_std = get_metrics(viol_h_batch,mode="default")
    viol_g_min, viol_g_1, viol_g_50, viol_g_99, viol_g_max, viol_g_mean, viol_g_std = get_metrics(viol_g_batch,mode="default")
    viol_lam_min, viol_lam_1, viol_lam_50, viol_lam_99, viol_lam_max, viol_lam_mean, viol_lam_std = get_metrics(viol_lam_batch,mode="default")

    constraints_dict = {"viol_h_min": viol_h_min, "viol_h_50": viol_h_50, "viol_h_99": viol_h_99, "viol_h_max": viol_h_max, "viol_h_mean": viol_h_mean, "viol_h_std": viol_h_std,
                        "viol_g_min": viol_g_min, "viol_g_50": viol_g_50, "viol_g_99": viol_g_99, "viol_g_max": viol_g_max, "viol_g_mean": viol_g_mean, "viol_g_std": viol_g_std,
                        "viol_lam_min": viol_lam_min, "viol_lam_50": viol_lam_50, "viol_lam_99": viol_lam_99, "viol_lam_max": viol_lam_max, "viol_lam_mean": viol_lam_mean, "viol_lam_std": viol_lam_std}

    # 3.4 optimality gap
    f_hat_batch = OP.f_batch_func(y_hat_batch,x_batch)
    f_opt_batch = OP.f_batch_func(y_opt_batch,x_batch)
    opt_gap_abs_batch = torch.abs(f_hat_batch - f_opt_batch)
    opt_gap_rel_batch = opt_gap_abs_batch/(torch.abs(f_opt_batch)+1e-16)
    opt_gap_abs_min, opt_gap_abs_1, opt_gap_abs_50, opt_gap_abs_99, opt_gap_abs_max, opt_gap_abs_mean, opt_gap_abs_std = get_metrics(opt_gap_abs_batch,mode="default")
    opt_gap_rel_min, opt_gap_rel_1, opt_gap_rel_50, opt_gap_rel_99, opt_gap_rel_max, opt_gap_rel_mean, opt_gap_rel_std = get_metrics(opt_gap_rel_batch,mode="default")
    opt_gap_dict = {"obj_val":f_hat_batch.mean().item(),"opt_gap_abs_50": opt_gap_abs_50, "opt_gap_abs_99": opt_gap_abs_99, "opt_gap_abs_max": opt_gap_abs_max, "opt_gap_abs_mean": opt_gap_abs_mean,
                    "opt_gap_rel_50": opt_gap_rel_50, "opt_gap_rel_99": opt_gap_rel_99, "opt_gap_rel_max": opt_gap_rel_max, "opt_gap_rel_mean": opt_gap_rel_mean}
    
    # 4. optimizer solution comparison
    # 4.1 number of active constraints
    g_opt = OP.g_batch_func(y_opt_batch)
    n_active_g_batch = torch.sum(g_opt >= 0,dim=1).to(torch.float)
    n_active_g_min, n_active_g_50, n_active_g_max, n_active_g_mean, n_active_g_std = get_metrics(n_active_g_batch,mode="small")
    # 4.2 optimality of optimizer data
    KKT_opt = OP.KKT_batch_func(z_opt_batch,x_batch)
    KKT_inf_opt = torch.norm(KKT_opt,p=float("inf"),dim=1)
    KKT_inf_opt_min, KKT_inf_opt_50, KKT_inf_opt_max, KKT_inf_opt_mean, KKT_inf_opt_std = get_metrics(KKT_inf_opt,mode="small")
    optimal_solution_dict = {"n_active_g_min": n_active_g_min, "n_active_g_50": n_active_g_50, "n_active_g_max": n_active_g_max, 
                             "KKT_inf_opt_min": KKT_inf_opt_min, "KKT_inf_opt_50": KKT_inf_opt_50, "KKT_inf_opt_max": KKT_inf_opt_max}

    # 5. inference speed
    if speed_eval:
        model_jit_gpu = predictor.export_jit_gpu()
        model_jit_cpu = predictor.export_jit_cpu()
        # 4.1 batch prediction GPU
        inference_times_batch_gpu = measure_inference_time_predictor(model_jit_gpu, x_batch, use_cuda=True)
        # 4.2 single datapoint jit prediction GPU
        x_i_gpu = x_batch[0,:]
        inference_times_single_gpu = measure_inference_time_predictor(model_jit_gpu, x_i_gpu, use_cuda=True)
        # 4.3 single datapoint jit prediction CPU
        x_i_cpu = x_batch[0,:]
        x_i_cpu = x_i_cpu.cpu()
        inference_times_single_cpu = measure_inference_time_predictor(model_jit_cpu, x_i_cpu, use_cuda=False)    

        # 4.4 stack results
        # add info to keys corresponding to 4.1 to 4.3
        inference_times_batch_gpu = {f"gpu_batch_{k}": v for k,v in inference_times_batch_gpu.items()}
        inference_times_single_gpu = {f"gpu_single_{k}": v for k,v in inference_times_single_gpu.items()}
        inference_times_single_cpu = {f"cpu_single_{k}": v for k,v in inference_times_single_cpu.items()}
    else:
        inference_times_batch_gpu = {}
        inference_times_single_gpu = {}
        inference_times_single_cpu = {}

    # 6. stack results
    results_dict = {**test_data_dict,**optimal_solution_dict, **optimality_dict, **distance_dict, **constraints_dict, **opt_gap_dict,
                    **inference_times_batch_gpu, **inference_times_single_gpu, **inference_times_single_cpu}
    return results_dict

def evaluate_trajectory(OP,zk_traj,x_batch,save_pth=None):
    n_data = x_batch.shape[0]
    traj_eval_list = []
    Tk_list = []
    for idx, zk_i in enumerate(zk_traj):
        Tk_i = OP.Tk_batch_func(zk_i,x_batch)
        Tk_list.append(Tk_i)
        Tk_i_min, Tk_i_1, Tk_i_50, Tk_i_99, Tk_i_max, Tk_i_mean, Tk_i_std = get_metrics(Tk_i)

        KKT_i = OP.KKT_batch_func(zk_i,x_batch)
        KKT_inf_norm_i = torch.norm(KKT_i,p=float("inf"),dim=1)
        KKT_inf_min, KKT_inf_1, KKT_inf_50, KKT_inf_99, KKT_inf_max, KKT_inf_mean, KKT_inf_std = get_metrics(KKT_inf_norm_i)

        # percentage of trajectories that have reached tolerances of Tk_lim = 1e-6, 1e-8, 1e-10, 1e-16
        Tk_lim_frac = [torch.sum(Tk_i < lim).item()/n_data for lim in Tk_lim]

        KKT_lim_frac = [torch.sum(KKT_inf_norm_i < lim).item()/n_data for lim in KKT_lim]

        traj_eval = {}
        traj_eval["Tk_i_1"] = Tk_i_1
        traj_eval["Tk_i_99"] = Tk_i_99
        traj_eval["Tk_i_50"] = Tk_i_50
        traj_eval["Tk_i_min"] = Tk_i_min
        traj_eval["Tk_i_mean"] = Tk_i_mean
        traj_eval["Tk_i_max"] = Tk_i_max
        traj_eval["Tk_lim_frac"] = Tk_lim_frac
        traj_eval["KKT_inf_min"] = KKT_inf_min
        traj_eval["KKT_inf_1"] = KKT_inf_1
        traj_eval["KKT_inf_50"] = KKT_inf_50
        traj_eval["KKT_inf_99"] = KKT_inf_99
        traj_eval["KKT_inf_max"] = KKT_inf_max
        traj_eval["KKT_inf_mean"] = KKT_inf_mean
        traj_eval["KKT_lim_frac"] = KKT_lim_frac
        traj_eval_list.append(traj_eval)

    if save_pth is not None:
        # visualize Tk trajectories
        # 1. quantiles
        fig0, ax0 = plt.subplots()
        ax0.plot([traj_eval["Tk_i_1"] for traj_eval in traj_eval_list],label="Tk_1")
        ax0.plot([traj_eval["Tk_i_99"] for traj_eval in traj_eval_list],label="Tk_99")
        ax0.plot([traj_eval["Tk_i_50"] for traj_eval in traj_eval_list],label="Tk_50")
        ax0.plot([traj_eval["Tk_i_min"] for traj_eval in traj_eval_list],label="Tk_min")
        ax0.plot([traj_eval["Tk_i_mean"] for traj_eval in traj_eval_list],label="Tk_mean")
        ax0.plot([traj_eval["Tk_i_max"] for traj_eval in traj_eval_list],label="Tk_max")
        ax0.legend()
        ax0.set_yscale("log")
        ax0.set_ylabel("Tk")
        ax0.set_xlabel("iter")
        fig0.savefig(save_pth.joinpath("solver_traj_eval_Tk.png"))

        fig1, ax1 = plt.subplots()
        ax1.plot([traj_eval["KKT_inf_1"] for traj_eval in traj_eval_list],label="KKT_inf_1")
        ax1.plot([traj_eval["KKT_inf_99"] for traj_eval in traj_eval_list],label="KKT_inf_99")
        ax1.plot([traj_eval["KKT_inf_50"] for traj_eval in traj_eval_list],label="KKT_inf_50")
        ax1.plot([traj_eval["KKT_inf_min"] for traj_eval in traj_eval_list],label="KKT_inf_min")
        ax1.plot([traj_eval["KKT_inf_mean"] for traj_eval in traj_eval_list],label="KKT_inf_mean")
        ax1.plot([traj_eval["KKT_inf_max"] for traj_eval in traj_eval_list],label="KKT_inf_max")
        ax1.legend()
        ax1.set_yscale("log")
        ax1.set_ylabel("KKT_inf")
        ax1.set_xlabel("iter")
        fig1.savefig(save_pth.joinpath("solver_traj_eval_KKT_inf.png"))

        # 2. fractions
        fig2, ax2 = plt.subplots()
        for i, lim in enumerate(Tk_lim):
            ax2.plot([traj_eval["Tk_lim_frac"][i] for traj_eval in traj_eval_list],label=f"Tolerance = {lim}")
        ax2.legend()
        ax2.set_ylabel("frac Tk < lim")
        ax2.set_xlabel("iter")
        fig2.savefig(save_pth.joinpath("solver_traj_eval_frac_Tk.png"))

        fig2, ax2 = plt.subplots()
        for i, lim in enumerate(KKT_lim):
            ax2.plot([traj_eval["KKT_lim_frac"][i] for traj_eval in traj_eval_list],label=f"Tolerance = {lim}")
        ax2.legend()
        ax2.set_ylabel("frac KKT_inf < lim")
        ax2.set_xlabel("iter")
        fig2.savefig(save_pth.joinpath("solver_traj_eval_frac_KKT_inf.png"))

        # 3. 50 random trajectories
        idx_list = torch.randperm(n_data)[:50]
        fig4, ax4 = plt.subplots()
        for idx in idx_list:
            plt_list = [Tk_i[idx].item() for Tk_i in Tk_list]
            ax4.plot(plt_list)
        ax4.set_yscale("log")
        ax4.set_ylabel("Tk")
        ax4.set_xlabel("iter")
        fig4.savefig(save_pth.joinpath("solver_traj_eval_50.png"))

        # 4. 50 worst trajectories
        # at Tk_list[-1], get the 50 worst
        Tk_last = Tk_list[-1]
        idx_list = torch.argsort(Tk_last,descending=True)[:50]
        fig5, ax5 = plt.subplots()
        for idx in idx_list:
            plt_list = [Tk_i[idx].item() for Tk_i in Tk_list]
            ax5.plot(plt_list)
        ax5.set_yscale("log")
        ax5.set_ylabel("Tk")
        ax5.set_xlabel("iter")
        fig5.savefig(save_pth.joinpath("solver_traj_eval_50_worst.png"))        
    
    return traj_eval_list

def evaluate_solver(solver,OP,test_data,predictor=None,alpha=1.0,max_iter=200,tol=1e-16,delta=1000.0,update_mode="line_search",save_pth=None):
    # 1. load data and data metrics
    x_batch = test_data["x"].to(dtype)
    n_data = x_batch.shape[0]
    z_opt_batch = test_data["z_opt"].to(dtype)
    y_opt_batch = z_opt_batch[:,:OP.n_vars]
    nu_opt_batch = z_opt_batch[:,OP.n_vars:OP.n_vars+OP.n_eq]
    lam_opt_batch = z_opt_batch[:,OP.n_vars+OP.n_eq:]
    test_data_dict = {"n_test_data": n_data}
    if OP.obj_type == "quad":
        full_solve_time = torch.tensor(test_data["full_solve_time"])
        fst_min, fst_50, fst_max, fst_mean, fst_std = get_metrics(full_solve_time,mode="small")
        test_data_dict["opt_full_solve_time_min"] = fst_min
        test_data_dict["opt_full_solve_time_50"] = fst_50
        test_data_dict["opt_full_solve_time_max"] = fst_max
        test_data_dict["opt_full_solve_time_mean"] = fst_mean
        test_data_dict["opt_full_solve_time_std"] = fst_std
        test_data_dict["test_solver"] = "OSQP"
    else:
        test_data_dict["test_solver"] = "IPOPT"
    solve_time = torch.tensor(test_data["solve_time"])
    st_min, st_50, st_max, st_mean, st_std = get_metrics(solve_time,mode="small")
    test_data_dict["opt_solve_time_min"] = st_min
    test_data_dict["opt_solve_time_50"] = st_50
    test_data_dict["opt_solve_time_max"] = st_max
    test_data_dict["opt_solve_time_mean"] = st_mean
    test_data_dict["opt_solve_time_std"] = st_std
    iter_count = torch.tensor(test_data["iter_count"]).to(torch.float)
    ic_min, ic_50, ic_max, ic_mean, ic_std = get_metrics(iter_count,mode="small")
    test_data_dict["opt_iter_count_min"] = ic_min
    test_data_dict["opt_iter_count_50"] = ic_50
    test_data_dict["opt_iter_count_max"] = ic_max
    test_data_dict["opt_obj_val"] = OP.f_batch_func(y_opt_batch).mean().item()

    # 2. solve
    z_hat_batch, zk_traj, step_time_list = solver.solve_batch(x_batch,max_iter=max_iter,alpha=alpha,predictor=predictor,update_mode=update_mode,return_trajectory=True,delta=delta)

    # 2.1 trajectory evaluation
    traj_eval_list = evaluate_trajectory(OP,zk_traj,x_batch,save_pth=save_pth)

    # for each tolerance in KKT_lim, calculate the number of iterations, until 50%, 95%, 99% and 100% of the trajectories have reached the tolerance
    n_iter_lim_50 = []
    n_iter_lim_95 = []
    n_iter_lim_99 = []
    n_iter_lim_100 = []
    for idx_lim, lim in enumerate(KKT_lim):
        n_val_50 = None
        n_val_95 = None
        n_val_99 = None
        n_val_100 = None
        for idx, traj_eval in enumerate(traj_eval_list):            
            # 50%
            if n_val_50 is None:
                if traj_eval["KKT_lim_frac"][idx_lim] >= 0.5:
                    n_val_50 = idx+1
            # 95%
            if n_val_95 is None:
                if traj_eval["KKT_lim_frac"][idx_lim] >= 0.95:
                    n_val_95 = idx+1
            # 99%
            if n_val_99 is None:
                if traj_eval["KKT_lim_frac"][idx_lim] >= 0.99:
                    n_val_99 = idx+1
            # 100%
            if n_val_100 is None:
                if traj_eval["KKT_lim_frac"][idx_lim] >= 1.0:
                    n_val_100 = idx+1
        n_iter_lim_50.append(n_val_50)
        n_iter_lim_95.append(n_val_95)
        n_iter_lim_99.append(n_val_99)
        n_iter_lim_100.append(n_val_100)               

    # split
    y_hat_batch = z_hat_batch[:,:OP.n_vars]
    nu_hat_batch = z_hat_batch[:,OP.n_vars:OP.n_vars+OP.n_eq]
    lam_hat_batch = z_hat_batch[:,OP.n_vars+OP.n_eq:]

    # 3. evaluate
    # 3.1 optimality conditions
    KKT_batch = OP.KKT_batch_func(z_hat_batch,x_batch)
    KKT_inf_norm = torch.norm(KKT_batch,p=float("inf"),dim=1)
    KKT_inf_min, KKT_inf_1, KKT_inf_50, KKT_inf_90, KKT_inf_95, KKT_inf_99, KKT_inf_max, KKT_inf_mean, KKT_inf_std = get_metrics(KKT_inf_norm,mode="large")
    Tk_batch = OP.Tk_batch_func(z_hat_batch,x_batch)
    Tk_min, Tk_1, Tk_50, Tk_90, Tk_95, Tk_99, Tk_max, Tk_mean, Tk_std = get_metrics(Tk_batch,mode="large")
    optimality_dict = {"KKT_inf_min": KKT_inf_min, "KKT_inf_1": KKT_inf_1, "KKT_inf_50": KKT_inf_50,"KKT_inf_90": KKT_inf_90,
                       "KKT_inf_95": KKT_inf_95,"KKT_inf_99": KKT_inf_99, "KKT_inf_max": KKT_inf_max, "KKT_inf_mean": KKT_inf_mean, "KKT_inf_std": KKT_inf_std,
                       "Tk_min": Tk_min, "Tk_1": Tk_1, "Tk_50": Tk_50, "Tk_90": Tk_90, "Tk_95": Tk_95, "Tk_99": Tk_99, "Tk_max": Tk_max, "Tk_mean": Tk_mean, "Tk_std": Tk_std}

    # 3.2 distance to data
    y_diff_batch = torch.norm(y_hat_batch - y_opt_batch,dim=1)
    nu_diff_batch = torch.norm(nu_hat_batch - nu_opt_batch,dim=1)
    lam_diff_batch = torch.norm(lam_hat_batch - lam_opt_batch,dim=1)
    y_diff_batch_scaled = y_diff_batch/(torch.norm(y_opt_batch,dim=1)+1e-16)
    nu_diff_batch_scaled = nu_diff_batch/(torch.norm(nu_opt_batch,dim=1)+1e-16)
    lam_diff_batch_scaled = lam_diff_batch/(torch.norm(lam_opt_batch,dim=1)+1e-16)

    y_diff_min, y_diff_1, y_diff_50, y_diff_99, y_diff_max, y_diff_mean, y_diff_std = get_metrics(y_diff_batch,mode="default")
    nu_diff_min, nu_diff_1, nu_diff_50, nu_diff_99, nu_diff_max, nu_diff_mean, nu_diff_std = get_metrics(nu_diff_batch,mode="default")
    lam_diff_min, lam_diff_1, lam_diff_50, lam_diff_99, lam_diff_max, lam_diff_mean, lam_diff_std = get_metrics(lam_diff_batch,mode="default")

    y_diff_scaled_min, y_diff_scaled_1, y_diff_scaled_50, y_diff_scaled_99, y_diff_scaled_max, y_diff_scaled_mean, y_diff_scaled_std = get_metrics(y_diff_batch_scaled,mode="default")
    nu_diff_scaled_min, nu_diff_scaled_1, nu_diff_scaled_50, nu_diff_scaled_99, nu_diff_scaled_max, nu_diff_scaled_mean, nu_diff_scaled_std = get_metrics(nu_diff_batch_scaled,mode="default")
    lam_diff_scaled_min, lam_diff_scaled_1, lam_diff_scaled_50, lam_diff_scaled_99, lam_diff_scaled_max, lam_diff_scaled_mean, lam_diff_scaled_std = get_metrics(lam_diff_batch_scaled,mode="default")

    distance_dict = {"y_diff_max": y_diff_max, "y_diff_99": y_diff_99, "y_diff_50": y_diff_50, "y_diff_min": y_diff_min,
                    "nu_diff_max": nu_diff_max, "nu_diff_99": nu_diff_99, "nu_diff_50": nu_diff_50, "nu_diff_min": nu_diff_min,
                    "lam_diff_max": lam_diff_max, "lam_diff_99": lam_diff_99, "lam_diff_50": lam_diff_50, "lam_diff_min": lam_diff_min,
                    "y_diff_scaled_max": y_diff_scaled_max, "y_diff_scaled_99": y_diff_scaled_99, "y_diff_scaled_50": y_diff_scaled_50, "y_diff_scaled_min": y_diff_scaled_min,
                    "nu_diff_scaled_max": nu_diff_scaled_max, "nu_diff_scaled_99": nu_diff_scaled_99, "nu_diff_scaled_50": nu_diff_scaled_50, "nu_diff_scaled_min": nu_diff_scaled_min,
                    "lam_diff_scaled_max": lam_diff_scaled_max, "lam_diff_scaled_99": lam_diff_scaled_99, "lam_diff_scaled_50": lam_diff_scaled_50, "lam_diff_scaled_min": lam_diff_scaled_min
    }

    # 3.3 constraint violations
    h_val_batch = OP.h_batch_func(y_hat_batch,x_batch)
    g_val_batch = OP.g_batch_func(y_hat_batch)        
    viol_h_batch = torch.abs(h_val_batch)
    viol_g_batch = torch.relu(g_val_batch)
    viol_lam_batch = torch.relu(-lam_hat_batch)

    viol_h_min, viol_h_1, viol_h_50, viol_h_99, viol_h_max, viol_h_mean, viol_h_std = get_metrics(viol_h_batch,mode="default")
    viol_g_min, viol_g_1, viol_g_50, viol_g_99, viol_g_max, viol_g_mean, viol_g_std = get_metrics(viol_g_batch,mode="default")
    viol_lam_min, viol_lam_1, viol_lam_50, viol_lam_99, viol_lam_max, viol_lam_mean, viol_lam_std = get_metrics(viol_lam_batch,mode="default")

    constraints_dict = {"viol_h_min": viol_h_min, "viol_h_50": viol_h_50, "viol_h_99": viol_h_99, "viol_h_max": viol_h_max, "viol_h_mean": viol_h_mean, "viol_h_std": viol_h_std,
                        "viol_g_min": viol_g_min, "viol_g_50": viol_g_50, "viol_g_99": viol_g_99, "viol_g_max": viol_g_max, "viol_g_mean": viol_g_mean, "viol_g_std": viol_g_std,
                        "viol_lam_min": viol_lam_min, "viol_lam_50": viol_lam_50, "viol_lam_99": viol_lam_99, "viol_lam_max": viol_lam_max, "viol_lam_mean": viol_lam_mean, "viol_lam_std": viol_lam_std}

    # 3.4 optimality gap
    f_hat_batch = OP.f_batch_func(y_hat_batch)
    f_opt_batch = OP.f_batch_func(y_opt_batch)
    opt_gap_abs_batch = torch.abs(f_hat_batch - f_opt_batch)
    opt_gap_rel_batch = opt_gap_abs_batch/(torch.abs(f_opt_batch)+1e-16)
    opt_gap_abs_min, opt_gap_abs_1, opt_gap_abs_50, opt_gap_abs_99, opt_gap_abs_max, opt_gap_abs_mean, opt_gap_abs_std = get_metrics(opt_gap_abs_batch,mode="default")
    opt_gap_rel_min, opt_gap_rel_1, opt_gap_rel_50, opt_gap_rel_99, opt_gap_rel_max, opt_gap_rel_mean, opt_gap_rel_std = get_metrics(opt_gap_rel_batch,mode="default")
    opt_gap_dict = {"obj_val":f_hat_batch.mean().item(),"opt_gap_abs_50": opt_gap_abs_50, "opt_gap_abs_99": opt_gap_abs_99, "opt_gap_abs_max": opt_gap_abs_max, "opt_gap_abs_mean": opt_gap_abs_mean,
                    "opt_gap_rel_50": opt_gap_rel_50, "opt_gap_rel_99": opt_gap_rel_99, "opt_gap_rel_max": opt_gap_rel_max, "opt_gap_rel_mean": opt_gap_rel_mean}

    # 4. optimizer solution comparison
    # 4.1 number of active constraints
    g_opt = OP.g_batch_func(y_opt_batch)
    n_active_g_batch = torch.sum(g_opt >= 0,dim=1).to(torch.float)
    n_active_g_min, n_active_g_50, n_active_g_max, n_active_g_mean, n_active_g_std = get_metrics(n_active_g_batch,mode="small")
    # 4.2 optimality of optimizer data
    KKT_opt = OP.KKT_batch_func(z_opt_batch,x_batch)
    KKT_inf_opt = torch.norm(KKT_opt,p=float("inf"),dim=1)
    KKT_inf_opt_min, KKT_inf_opt_50, KKT_inf_opt_max, KKT_inf_opt_mean, KKT_inf_opt_std = get_metrics(KKT_inf_opt,mode="small")
    optimal_solution_dict = {"n_active_g_min": n_active_g_min, "n_active_g_50": n_active_g_50, "n_active_g_max": n_active_g_max, 
                             "KKT_inf_opt_min": KKT_inf_opt_min, "KKT_inf_opt_50": KKT_inf_opt_50, "KKT_inf_opt_max": KKT_inf_opt_max}
    
    # 5. inference speed
    if speed_eval:
        if predictor is not None:
            # model_jit_gpu = predictor.export_jit_gpu()
            model_jit_cpu = predictor.export_jit_cpu()
        else:
            # model_jit_gpu = None
            model_jit_cpu = None        
        
        solve_step_batch_gpu = solver.get_solver_step_funcs(alpha=alpha,batch_size=n_data,jit=True,use_cuda=True)
        solve_step_single_cpu= solver.get_solver_step_funcs(alpha=alpha,batch_size=None,jit=True,use_cuda=False)

        # 5.1 batch step time jit GPU
        step_times_batch_gpu = measure_step_time_solver(solve_step_batch_gpu, z_hat_batch, x_batch, use_cuda=True)
        # 5.2 single datapoint step time jit CPU
        x_i_cpu = x_batch[0,:]
        x_i_cpu = x_i_cpu.cpu()
        z_i_cpu = z_hat_batch[0,:]
        z_i_cpu = z_i_cpu.cpu()
        step_times_single_cpu = measure_step_time_solver(solve_step_single_cpu, z_i_cpu, x_i_cpu, use_cuda=False)

        # 5.3 total time solver CPU single
        if update_mode == "gamma":
            total_solve_time_cpu_single = {}
        else:
            total_solve_time_cpu_single = measure_total_time_solver_cpu_single(solver,solve_step_single_cpu, x_batch, predictor_cpu=model_jit_cpu,repeat=1,max_iter=max_iter,update_mode=update_mode,tol=tol)
            total_solve_time_cpu_single = {f"cpu_single_{k}": v for k,v in total_solve_time_cpu_single.items()}

        # 5.4 stack results
        # add info to keys corresponding to 5.1 to 5.3
        step_times_batch_gpu = {f"gpu_batch_{k}": v for k,v in step_times_batch_gpu.items()}
        step_times_single_cpu = {f"cpu_single_{k}": v for k,v in step_times_single_cpu.items()}

    else:
        step_times_batch_gpu = {}
        step_times_single_cpu = {}
        total_solve_time_cpu_single = {}
    
    # 6. stack results
    results_dict = {**test_data_dict,**optimal_solution_dict, **optimality_dict, **distance_dict, **constraints_dict, **opt_gap_dict,
                    **step_times_batch_gpu, **step_times_single_cpu, **total_solve_time_cpu_single}
    results_dict["Tk_lim"] = Tk_lim
    results_dict["KKT_lim"] = KKT_lim
    results_dict["n_iter_lim_50"] = n_iter_lim_50
    results_dict["n_iter_lim_95"] = n_iter_lim_95
    results_dict["n_iter_lim_99"] = n_iter_lim_99
    results_dict["n_iter_lim_100"] = n_iter_lim_100
    results_dict["trajectories"] = traj_eval_list
    return results_dict


# %%
# MAIN
if __name__ == "__main__":
    results_pth = file_pth.joinpath(run_folder,mode)
    config_list = get_all_configs(results_pth)

    for config in config_list:
        # close all figures
        plt.close("all")
        if not config["evaluated"] or overwrite:
            if specific_op is not None:
                if config["op_name"] not in specific_op:
                    continue
            if specific_exp is not None:
                if Path(config["exp_pth"]).name not in specific_exp:
                    continue
            # 1. load OP
            op_pth = file_pth.joinpath(op_folder_name,config["op_name"])
            OP = load_OP(op_pth)
            test_data = torch.load(op_pth.joinpath("sample_data.pth"))
            #xk_batch=OP.batch_gen_x_new(1)
            #zk_batch=OP.batch_gen_z(1)
            #test_data['z_opt']=zk_batch
            #test_data['x']=xk_batch
            if mode != "predictor":
                # get update mode
                update_mode = UPDATE_MODES[config["update_mode"]]
                
            if mode == "predictor":
                # 2. load predictor
                predictor = load_predictor(config)
                # 3. evaluate
                results_dict = evaluate_predictor(predictor,OP,test_data)

            elif mode == "solver_no_pred":
                # 2. load solver
                solver = load_solver(OP,config)
                # 3. evaluate
                results_dict = evaluate_solver(solver,OP,test_data,predictor=None,alpha=config["alpha"],max_iter=MAX_ITER,tol=config["Tk_lim"],update_mode=update_mode,save_pth=Path(config["exp_pth"]))

            elif mode == "solver_with_pred":
                # 2. load solver and predictor
                solver = load_solver(OP,config)
                predictor = load_predictor_from_pth(Path(config["predictor"]["exp_pth"]))
                # 3. evaluate
                results_dict = evaluate_solver(solver,OP,test_data,predictor=predictor,alpha=config["alpha"],max_iter=MAX_ITER,tol=config["Tk_lim"],update_mode=update_mode,save_pth=Path(config["exp_pth"]))

            else:
                raise ValueError("mode not recognized.")
            
            # 4. save
            with open(Path(config["exp_pth"]).joinpath("eval.json"), 'w') as f:
                json.dump(results_dict,f,indent=4)

            # 5. update config
            config["evaluated"] = True
            with open(Path(config["exp_pth"]).joinpath("config.json"), 'w') as f:
                json.dump(config,f,indent=4)
            print(f"{config['op_name']} evaluated.\n")
        else:
            print(f"{config['op_name']} already evaluated. ")
            print("To overwrite set overwrite=True.\n")
