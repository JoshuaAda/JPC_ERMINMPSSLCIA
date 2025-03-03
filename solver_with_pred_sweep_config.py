# IMPORTS
import uuid
from pathlib import Path
import copy
import json
import os
file_pth = Path(__file__).parent.resolve()

# GENERAL CONFIG
op_folder_name = "parametric_OP_data_robust"
results_dir = "robust_case_study"
sweep_config_name = "solver_with_pred_configs_robust"
append_cfg = False

# SWEEP CONFIG
op_names = ["nonlinear_96x48x192_0"]

neuron_numbers = [2048]


# Predictor Choice
n_neurons_predictor = 2048
n_layers_predictor = 1
act_fn_predictor = "leaky_relu"
convexified_predictor = True

# Predictor Info
def get_all_configs(pth):
    # configs
    config_list = []
    op_names = [f for f in os.listdir(pth) if os.path.isdir(pth.joinpath(f))]
    for op_name in op_names:
        op_pth = pth.joinpath(op_name)
        for i in range(1000):
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
predictor_configs_pth = file_pth.joinpath(results_dir,"predictor")
predictor_configs = get_all_configs(predictor_configs_pth)

def get_predictor(op_name,n_neurons,n_layers,act_fn,n_epochs=150000):
    for cfg in predictor_configs:
        if (cfg["op_name"] == op_name) and (cfg["n_neurons"] == n_neurons) and (cfg["n_layers"] == n_layers) and (cfg["act_fn"] == act_fn) and (cfg["N_epochs"] == n_epochs) and (cfg["convexified"] == convexified_predictor):
            assert cfg["trained"], f"Predictor for {op_name} with {n_neurons} neurons, {n_layers} layers and {act_fn} activation not trained yet."
            pred_cfg = {}
            pred_cfg["id"] = cfg["id"]
            pred_cfg["n_neurons"] = cfg["n_neurons"]
            pred_cfg["n_layers"] = cfg["n_layers"]
            pred_cfg["act_fn"] = cfg["act_fn"]
            pred_cfg["exp_pth"] = cfg["exp_pth"]
            return pred_cfg
    raise ValueError(f"Predictor for {op_name} with {n_neurons} neurons, {n_layers} layers and {act_fn} activation not found.")

# Solver Config
solver_config = {}
solver_config["use_predictor"] = True
solver_config["n_steps_max"] = 2000
solver_config["alpha"] = 1.0
solver_config["Tk_lim"] = 1e-8
solver_config["n_layers"] = 1
solver_config["act_fn"] = "leaky_relu"
solver_config["output_act_fn"] = "linear"

solver_config["N_epochs"] = 100000 
solver_config["append_delay"] = 100
solver_config["batch_size"] = 4096
solver_config["weight_decay"] = 1e-3
solver_config["lr"]= 1e-4
solver_config["log_loss"] = True
solver_config["convexified"] = True
solver_config["delta"] = 1000.0
solver_config["update_mode"] = "eta_total"
solver_config["loss_mode"] = "explicit"


solver_configs = []
for op_name in op_names:
    for n_neurons in neuron_numbers:
        cfg = {}
        cfg["op_name"] = op_name      
        cfg["mode"] = "solver_with_pred"  
        cfg["n_neurons"] = n_neurons
        cfg.update(copy.deepcopy(solver_config))
        cfg["id"] = str(uuid.uuid4())
        cfg["trained"] = False
        cfg["evaluated"] = False
        cfg["exp_pth"] = None

        # get predictor config
        pred_cfg = get_predictor(op_name,n_neurons_predictor,n_layers_predictor,act_fn_predictor)
        cfg["predictor"] = pred_cfg

        solver_configs.append(cfg)


# Save configs
# check whether sweep config alreay exists
pth_exists = Path(file_pth).joinpath(results_dir,sweep_config_name+".json").exists()

if not pth_exists:
    with open(Path(file_pth).joinpath(results_dir,sweep_config_name+".json"), 'w') as f:
        json.dump(solver_configs, f, indent=4)
    print(f"Saved {sweep_config_name}.")

else:
    if append_cfg:
        try:
            with open(Path(file_pth).joinpath(results_dir,sweep_config_name+".json"), 'r') as f:
                prev_configs = json.load(f)
            solver_configs = prev_configs + solver_configs
            print(f"Appended to {sweep_config_name}.")
            # save sweep config
            with open(Path(file_pth).joinpath(results_dir,sweep_config_name+".json"), 'w') as f:
                json.dump(solver_configs, f, indent=4)                
        except:
            pass
    else:
        raise FileExistsError(f"{sweep_config_name} already exists. Set append_cfg to True to append to the existing file.")
