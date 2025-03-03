run_folder = "robust_case_study"
op_folder_name = "parametric_OP_data_robust"


#sweep_config_name = "predictor_configs_robust.json"
# sweep_config_name = "solver_no_pred_configs_trial.json"
sweep_config_name = "solver_with_pred_configs_robust.json"

# Imports
from parametric_OP import parametricOP
from parametric_OPz import parametricOPz
from models import generate_experiment_dir, FeedforwardNN, Predictor, Solver, count_params
from pathlib import Path
import matplotlib.pyplot as plt
import json
import torch
import matplotlib
matplotlib.use('TkAgg')
seed = 42
dtype = torch.float32
#dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(seed)
file_pth = Path(__file__).parent.resolve()

# Functions
def train_predictor(OP,config,exp_pth):
    # NN
    n_in = config["n_in"]#151#75#63#OP.n_x
    n_out = OP.n_z    
    n_layers = config["n_layers"]
    n_neurons = config["n_neurons"]
    act_fn = config["act_fn"]
    output_act_fn = config["output_act_fn"]

    model = FeedforwardNN(n_in,n_out,n_layers,n_neurons,act_fn,output_act_fn)
    predictor = Predictor(model)

    # Train Predictor
    train_logger = predictor.train(OP,config)

    ### Postprocessing
    # Visualize
    _ = train_logger.visualize_history("loss",log_scaling=(not config["log_loss"]),exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("Tk",metrics=["mean","50","1","99"],log_scaling=True,exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("weight_norm_sum",log_scaling=False,exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("lr",log_scaling=True,exp_pth=exp_pth,save_fig=True)
    plt.close("all")

    # Save Predictor
    train_logger.save_history(exp_pth=exp_pth,file_name="history")
    predictor.save_model(exp_pth,file_name="predictor_model.pt")
    predictor.save_weights(exp_pth,file_name="predictor_weights.pt")
    # Extend config
    prd_cfg = config.copy()
    prd_cfg["n_in"] = n_in
    prd_cfg["n_out"] = n_out
    prd_cfg["trained"] = True
    prd_cfg["train_time"] = train_logger.history["train_time"]
    prd_cfg["exp_pth"] = str(exp_pth)
    prd_cfg["N_epochs_trained"] = train_logger.history["epoch"][-1]
    prd_cfg["dtype"] = str(dtype)
    prd_cfg["n_params"] = count_params(model)
    with open(exp_pth.joinpath("config.json"),"w") as fp:
        json.dump(prd_cfg,fp,indent=4)
    return train_logger, predictor, prd_cfg

def train_solver(OP,config,exp_pth,predictor=None):
    # NN
    n_in = OP.n_z + 36#132#64#OP.n_x + 1
    n_out = OP.n_z
    n_neurons = config["n_neurons"]
    n_layers = config["n_layers"]
    act_fn = config["act_fn"]
    output_act_fn = config["output_act_fn"]

    model = FeedforwardNN(n_in=n_in,n_out=n_out,n_hidden_layers=n_layers,n_neurons=n_neurons,act_fn=act_fn,output_act_fn=output_act_fn)
    solver = Solver(model,OP)

    # Train Solver
    if predictor is not None:
        train_logger = solver.train(config,exp_pth,predictor=predictor)
    else:
        train_logger = solver.train(config)

    # Visualize
    _ = train_logger.visualize_history("loss",log_scaling=(not config["log_loss"]),exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("Tk",metrics=["mean","min","max","99","50"],log_scaling=True,exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("eta",metrics=["mean","min","max"],log_scaling=True,exp_pth=exp_pth,save_fig=True)
    # _ = train_logger.visualize_history("lr",log_scaling=True,exp_pth=exp_pth,save_fig=True)

    _ = train_logger.visualize_history("n_steps",metrics=["50","min","max"],log_scaling=False,exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("n_resample",log_scaling=False,exp_pth=exp_pth,save_fig=True)

    _ = train_logger.visualize_history("sum_nn_weight_norms",log_scaling=False,exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_running_avg("loss",log_scaling=(not config["log_loss"]),exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_running_avg("Tk_50",log_scaling=True,exp_pth=exp_pth,save_fig=True)
    plt.close("all")

    # Save Solver
    train_logger.save_history(exp_pth=exp_pth,file_name="history",as_json=True)
    solver.save_model(exp_pth,file_name="solver_model.pt")
    solver.save_weights(exp_pth,file_name="solver_weights.pt")
    # Extend config
    svl_cfg = config.copy()
    svl_cfg.update(config)
    svl_cfg["n_in"] = n_in
    svl_cfg["n_out"] = n_out
    svl_cfg["trained"] = True
    svl_cfg["train_time"] = train_logger.history["train_time"]
    svl_cfg["exp_pth"] = str(exp_pth)
    svl_cfg["dtype"] = str(dtype)
    svl_cfg["n_params"] = count_params(model)
    with open(exp_pth.joinpath("config.json"),"w") as fp:
        json.dump(svl_cfg,fp,indent=4)

    return train_logger, solver, svl_cfg

# loading
def load_OP(pth):
    return parametricOP.from_json(pth,file_name="op_cfg")

def load_predictor(pth):
    with open(pth.joinpath("config.json"), 'r') as f:
        predictor_config = json.load(f)
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
    predictor.load_weights(pth,file_name="predictor_weights.pt")
    return predictor, predictor_config

# TRAINING
if __name__ == "__main__":
    # 1. Load configs
    with open(file_pth.joinpath(run_folder,sweep_config_name), 'r') as f:
        configs = json.load(f)
    
    # 2. Loop
    for i,cfg in enumerate(configs):
        if cfg["trained"]:
            print(f"Skipping {cfg['op_name']} as it is already trained.")
            continue
        
        mode = cfg["mode"]

        # 2.1 load OP
        op_pth = file_pth.joinpath(op_folder_name,cfg["op_name"])
        OP =parametricOPz(obj_type = "nonlinear")#load_OP(op_pth)#parametricOPz(obj_type = "nonlinear")# #load_OP(op_pth)#parametricOPz(obj_type = "nonlinear")#
        # 2.2 setup experiment pth
        # exp_pth = generate_experiment_dir(log_dir=cfg["op_name"],save_dir=file_pth.joinpath(run_folder,mode))            
        exp_pth = generate_experiment_dir(log_dir=cfg["op_name"],save_dir=Path(run_folder,mode)) 
        
        if mode == "predictor":
            # 2.3 train predictor
            train_logger, predictor, cfg_update = train_predictor(OP,cfg,exp_pth)
            print(f"Predictor for {cfg['op_name']} trained and saved.")

        elif mode == "solver_no_pred":
            # 2.3 train solver
            train_logger, solver, cfg_update = train_solver(OP,cfg,exp_pth,predictor=None)
            print(f"Solver for {cfg['op_name']} trained and saved.")

        elif mode == "solver_with_pred":
            # 2.3 load predictor
            predictor_pth = Path(cfg["predictor"]["exp_pth"])
            predictor, predictor_config = load_predictor(predictor_pth)
            print(f"Predictor for {cfg['op_name']} loaded.")
            # 2.4 train solver
            train_logger, solver, cfg_update = train_solver(OP,cfg,exp_pth,predictor=predictor)        
        else:
            raise ValueError(f"mode {mode} not recognized.")

        # 2.4 update config
        configs[i] = cfg_update
    
        # 3. Save updated configs after every iteration
        with open(file_pth.joinpath(run_folder,sweep_config_name), 'w') as f:
            json.dump(configs,f,indent=4)
    print("All done.")