# IMPORTS
import uuid
from pathlib import Path
import copy
import json
file_pth = Path(__file__).parent.resolve()

# GENERAL CONFIG
op_folder_name = "parametric_OP_data_trial"
results_dir = "case_study_trial"
sweep_config_name = "solver_no_pred_configs_trial"
append_cfg = False

# SWEEP CONFIG
op_names = ["nonconvex_10x5x5_0"]

neuron_numbers = [2048]

# Solver Config
solver_config = {}
solver_config["use_predictor"] = False
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
        cfg["mode"] = "solver_no_pred"
        cfg["n_neurons"] = n_neurons
        cfg.update(copy.deepcopy(solver_config))
        cfg["id"] = str(uuid.uuid4())
        cfg["trained"] = False
        cfg["evaluated"] = False
        cfg["exp_pth"] = None
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
        raise FileExistsError(f"{sweep_config_name} already exists. Set append_cfg to True to append configs.")
