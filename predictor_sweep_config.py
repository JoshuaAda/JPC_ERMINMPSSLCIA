# IMPORTS
import uuid
from pathlib import Path
import copy
import json
file_pth = Path(__file__).parent.resolve()

# GENERAL CONFIG
op_folder_name = "parametric_OP_data_robust"
results_dir = "robust_case_study"
sweep_config_name = "predictor_configs_robust"
append_cfg = False

# SWEEP CONFIG
op_names = ["nonlinear_96x48x192_0"]

neuron_numbers = [2048]
n_hidden_layers = [1]


# Predictor Config
predictor_config = {}
# predictor_config["n_neurons"] = 500
# predictor_config["n_layers"] = 1
predictor_config["act_fn"] = "leaky_relu"
predictor_config["output_act_fn"] = "linear"

predictor_config["N_epochs"] = 150000
predictor_config["batch_size"] = 4096
predictor_config["weight_decay"] = 1e-3 # 0.0
predictor_config["lr"] = 1e-3 # 1e-4 if no LR scheduler is used
predictor_config["log_loss"] = False
predictor_config["convexified"] = True

predictor_config["use_lr_scheduler"] = True
predictor_config["lr_scheduler_patience"] = 1000
predictor_config["lr_scheduler_cooldown"] = 100
predictor_config["lr_reduce_factor"] = 0.1
predictor_config["min_lr"] = 1e-8
predictor_config["early_stop"] = True


predictor_configs = []
for op_name in op_names:
    for n_neurons in neuron_numbers:
        for n_layers in n_hidden_layers:
            cfg = {}
            cfg["op_name"] = op_name
            cfg["mode"] = "predictor"
            cfg["n_neurons"] = n_neurons
            cfg["n_layers"] = n_layers
            cfg.update(copy.deepcopy(predictor_config))
            cfg["id"] = str(uuid.uuid4())
            cfg["trained"] = False
            cfg["evaluated"] = False
            cfg["exp_pth"] = None
            predictor_configs.append(cfg)

# Save configs
# check whether sweep config alreay exists
pth_exists = Path(file_pth).joinpath(results_dir,sweep_config_name+".json").exists()

if not pth_exists:
    with open(Path(file_pth).joinpath(results_dir,sweep_config_name+".json"), 'w') as f:
        json.dump(predictor_configs, f, indent=4)
    print(f"Saved {sweep_config_name}.")

else:
    if append_cfg:
        try:
            with open(Path(file_pth).joinpath(results_dir,sweep_config_name+".json"), 'r') as f:
                prev_configs = json.load(f)
            predictor_configs = prev_configs + predictor_configs
            print(f"Appended to {sweep_config_name}.")
            # save sweep config
            with open(Path(file_pth).joinpath(results_dir,sweep_config_name+".json"), 'w') as f:
                json.dump(predictor_configs, f, indent=4)                
        except:
            pass
    else:
        raise FileExistsError(f"{sweep_config_name} already exists. Set append_cfg to True to append to existing config.")

