# Imports
import torch
from torch.autograd import Variable
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import copy
import torch.nn as nn
from parametric_OP import parametricOP
op_name = "nonlinear_100x50x50_0" # "rosenbrock_1000x500x500_0"

# pths
folder_name = "parametric_OP_data"
file_pth = Path(__file__).parent.resolve()
op_pth = file_pth.joinpath(folder_name)




# %%
# PREPARE
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# Functions
def get_activation_layer(act_fn):
    if act_fn == 'relu':
        return torch.nn.ReLU()
    elif act_fn == 'tanh':
        return torch.nn.Tanh()

    elif act_fn == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act_fn == 'sigmoid':

        return torch.nn.Sigmoid()
    else:
        raise ValueError("Activation function not implemented.")
    
def generate_experiment_dir(log_dir="runs",save_dir=None):
    # save_dir: where (top-level)
    # log_dir: where to save experiments (sub-level)
    for i in range(1000):
        if save_dir is None:
            local_pth = Path(log_dir,f"exp_{i}")
        else:
            local_pth = Path(save_dir,log_dir,f"exp_{i}")
        if not local_pth.exists():
            exp_dir = Path(log_dir,f"exp_{i}")
            break
    if save_dir is None:
        save_dir = Path(__file__).parent.absolute()
    else:
        save_dir = Path(save_dir)
    # current directory of file
    exp_pth = save_dir.joinpath(exp_dir)
    exp_pth.mkdir(parents=True,exist_ok=True)
    return exp_pth

@torch.no_grad()
def weight_norms(net):
    param_norms = [param.norm() for param in net.parameters()]
    return param_norms

@torch.no_grad()
def count_params(net):
    n_params = sum([param.numel() for param in net.parameters()])
    return n_params

# Feedforward NN
class FeedforwardNN(torch.nn.Module):
    """Feedforward Neural Network model.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in each hidden layer.
        act_fn (str): Activation function.
        output_act_fn (str): Output activation function.
    """
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons=500, act_fn='relu', output_act_fn='linear'):
        super().__init__()
        assert n_hidden_layers >= 0, "Number of hidden layers must be >= 0."
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_hidden_layers+1
        self.n_neurons = n_neurons
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(n_in, n_neurons))
                self.layers.append(get_activation_layer(act_fn))
            elif i == self.n_layers - 1:
                self.layers.append(torch.nn.Linear(n_neurons, n_out))
                if output_act_fn != 'linear':
                    self.layers.append(get_activation_layer(output_act_fn))
            else:
                self.layers.append(torch.nn.Linear(n_neurons, n_neurons))
                self.layers.append(get_activation_layer(act_fn))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
# TrainLogger
class TrainLogger:
    def __init__(self):
        self.history = {"epoch": []}

    def log_value(self,val,key):
        if torch.is_tensor(val):
           val = val.detach().cpu().item()        
        assert isinstance(val,(int,float)), "Value must be a scalar."
        if not key in self.history.keys():
            self.history[key] = []
        self.history[key].append(val)

    def log_data(self,val,key):
        if torch.is_tensor(val):
           val = val.detach().cpu().item()
        if not key in self.history.keys():
            self.history[key] = val
        else:
            self.history[key] = val
    
    def calculate_metrics(self,val):
        assert torch.is_tensor(val), "Value must be a tensor."
        # val = val.detach().clone()
        val_mean = val.mean()
        val_max = val.max()
        val_min = val.min()
        val_99 = torch.quantile(val,0.99)
        val_95 = torch.quantile(val,0.95)
        val_50 = torch.quantile(val,0.5)
        val_1 = torch.quantile(val,0.01)
        return val_mean,val_max,val_min,val_99,val_95,val_50,val_1
    
    def log_metrics(self,val,key):
        val_mean,val_max,val_min,val_99,val_95,val_50,val_1 = self.calculate_metrics(val)
        self.log_value(val_mean,key+"_mean")
        self.log_value(val_max,key+"_max")
        self.log_value(val_min,key+"_min")
        self.log_value(val_99,key+"_99")
        self.log_value(val_95,key+"_95")
        self.log_value(val_50,key+"_50")
        self.log_value(val_1,key+"_1")

    def log_weight_norms(self,net):
        wns = weight_norms(net)
        for i, wn in enumerate(wns):
            self.log_value(wn,f"weight_norm_{i}")

    def print_last_entry(self,keys=["epoch,train_loss"]):
        assert isinstance(keys,list), "Keys must be a list."
        for key in keys:
            # check wether keys are in history
            assert key in self.history.keys(), "Key not in history."
            print(key,": ",self.history[key][-1])
            
    def save_history(self,exp_pth,file_name="history",as_json=True):
        # pt        
        torch.save(self.history,exp_pth.joinpath(file_name+".pt"))
        # json
        if as_json:
            with open(exp_pth.joinpath(file_name+".json"), 'w') as f:
                json.dump(self.history, f, indent=4)
        print("history saved to: ", exp_pth.joinpath(file_name))

    def visualize_history(self,key,metrics=None,log_scaling=False,exp_pth=None,save_fig=False):
        fig, ax = plt.subplots()
        if metrics is None:
            ax.plot(self.history[key],label=key)
        else:
            for metric in metrics:
                ax.plot(self.history[key+"_"+metric],label=key+"_"+metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        if log_scaling:
            ax.set_yscale('log')
        ax.legend()
        fig.show()
        if save_fig:
            assert exp_pth is not None, "exp_pth must be provided."
            plt.savefig(exp_pth.joinpath(key + ".png"))
        return fig, ax
    
    def visualize_running_avg(self,key,log_scaling=False,exp_pth=None,save_fig=False):
        fig, ax = plt.subplots()
        sums = []
        for i in range(len(self.history[key])):
            if i == 0:
                sums.append(self.history[key][i])
            else:
                sums.append(sums[-1]+self.history[key][i])
        running_avg = [sums[i]/(i+1) for i in range(len(sums))]
        ax.plot(running_avg,label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        if log_scaling:
            ax.set_yscale('log')
        ax.legend()
        fig.show()
        if save_fig:
            assert exp_pth is not None, "exp_pth must be provided."
            plt.savefig(exp_pth.joinpath(key + "_running_avg.png"))
        return fig, ax
    
# Predictor
class Predictor(torch.nn.Module):
    """Predictor class for Neural Network model.

    Args:
        model (torch.nn.Module): Neural Network model
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        def custom_weight_init(m):
            if isinstance(m, nn.Linear):
                # Set weights with a uniform distribution between -1/sqrt(n) and 1/sqrt(n),
                # where n is the number of input features. This helps in keeping the
                # output in range.
                n = m.weight.size(1)  # number of input features
                limit = 1 / torch.sqrt(torch.tensor(n, dtype=torch.float32))
                nn.init.uniform_(m.weight, -limit, limit)

                # Optionally, you can set the bias to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(custom_weight_init)
        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def forward(self, x):
        return self.model(x)
    
    # Save and Load    
    def save_weights(self,save_dir, file_name = "weights.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self.state_dict(),save_pth)
        print("Weights (state dict) saved to: ", save_pth)

    def load_weights(self,load_dir, file_name = "weights.pt"):
        load_pth = Path(load_dir,file_name)
        weights = torch.load(load_pth)
        self.load_state_dict(weights)
        print("Weights (state dict) loaded from: ", load_pth)

    def save_model(self,save_dir,file_name = "model.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self,save_pth)
        print("Model saved to: ", save_pth)

    @staticmethod
    def load_model(load_dir,file_name = "model.pt"):
        load_pth = Path(load_dir,file_name)
        model = torch.load(load_pth)
        print("Model loaded from: ", load_pth)
        return model
        
    def get_hparams(self):
        return {"n_in":self.model.n_in,"n_out":self.model.n_out,"n_hidden_layers":self.model.n_layers-1,"n_neurons":self.model.n_neurons,"act_fn":self.model.act_fn,"output_act_fn":self.model.output_act_fn}
    
    def get_hparams_json(self):
        return json.dumps(self.get_hparams(),indent=4)
    
    def save_hparams(self,save_dir,file_name = "hparams.json"):
        save_pth = Path(save_dir,file_name)
        with open(save_pth, 'w') as f:
            json.dump(self.get_hparams(), f, indent=4)
        print("Hyperparameters saved to: ", save_pth)

    # Predict
    # @torch.no_grad()
    @torch.no_grad()
    def predict(self,x_batch):
        return self.model(x_batch)

    # Train
    def train(self,OP,train_config,print_frequency=10,train_logger=None):
        # CONFIG    
        N_epochs = train_config["N_epochs"]
        batch_size = train_config["batch_size"]
        weight_decay = train_config["weight_decay"]
        lr = train_config["lr"]
        log_loss = train_config["log_loss"]

        use_lr_scheduler = train_config["use_lr_scheduler"]

        # backwards compatibility
        if "convexified" in train_config.keys():
            convexified = train_config["convexified"]
        else:
            convexified = False

        # Logger
        if train_logger is None:
            train_logger = TrainLogger()
        else:
            train_logger = copy.deepcopy(train_logger)

        # Optimizer        
        optim = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
        
        # LR Scheduler
        if use_lr_scheduler:            
            # LR Scheduler
            lr_scheduler_patience  = train_config["lr_scheduler_patience"]
            lr_scheduler_cooldown  = train_config["lr_scheduler_cooldown"]
            lr_reduce_factor  = train_config["lr_reduce_factor"]
            min_lr  = train_config["min_lr"]
            
            # Early Stopping
            early_stop = train_config["early_stop"]
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=lr_reduce_factor, patience=lr_scheduler_patience, 
                                                          threshold=1e-5, threshold_mode='rel', cooldown=lr_scheduler_cooldown, min_lr=min_lr, eps=0.0)
            #torch.optim.lr_scheduler.
        # TRAINING
        #optim.param_groups[0]['lr'] = 1e-5
        #optim.param_groups[0]['initial_lr'] = 1e-5
        train_tic = time.perf_counter()
        n_batch=int(batch_size/len(OP.c_scen)**OP.N_r)
        z_hat_batch=0
        for epoch in range(N_epochs):
            tic = time.perf_counter()

            # random batch
            #if epoch % 10==0:
            #    x0 = (torch.tensor([[0.95], [0.95], [0.625], [0.625]], device=OP.device).T * torch.rand(n_batch,4)).repeat_interleave(2187,dim=0) + torch.tensor([[0.05], [0.05], [0.375], [0.375]], device=OP.device).T.repeat((2187*n_batch, 1))
            #    u_tilde = (torch.tensor([[0.95], [1]], device=OP.device).T * torch.rand(n_batch,2)).repeat_interleave(2187,dim=0) + torch.tensor([[0.05], [0]],device=OP.device).T.repeat((2187*n_batch, 1))
            #    past_u= (torch.tensor([[0.95], [1]], device=OP.device).T * torch.cat(
            #        (torch.rand(n_batch, 1), torch.randint(0, 2, (n_batch, 1))), axis=1)).repeat_interleave(2187,
            #                                                                                                dim=0) + torch.tensor(
            #        [[0.05], [0]],
            #        device=OP.device).T.repeat((2187 * n_batch, 1))
            #    p_batch = OP.batch_gen_x_new(n_batch, x0, past_u,u_tilde)
            #else:
            #    p_batch=OP.update_x_batch(n_batch,z_hat_batch.detach(),p_batch.detach())
            p_batch=OP.batch_gen_x_random(batch_size)
            # Zero Grads
            optim.zero_grad(set_to_none=True)
            z_batch=z_hat_batch
            # Prediction
            z_hat_batch = self.forward(p_batch)

            # LOSS
            if convexified:
                Tk_batch = OP.Tk_conv_batch_func(z_hat_batch,p_batch)
            else:
                #x_batch_new = p_batch.repeat(batch_size, 1)
                #zk_batch_new = z_hat_batch
                #zk_batch_new = zk_batch_new.repeat_interleave(batch_size, dim=0)
                #Tk_batch = torch.sum(OP.Tk_batch_func(zk_batch_new, x_batch_new).reshape(batch_size, -1), axis=0)
               # val=OP.Tk_func(z_hat_batch[1],p_batch[1])

                Tk_batch = OP.Tk_batch_func(z_hat_batch,p_batch)

            assert not torch.isnan(Tk_batch).any(), "Tk_batch contains nan"

            # scale and mean loss 
            if log_loss:
                loss = torch.mean(torch.log10(Tk_batch/OP.n_z+1e-16))
            else:
                loss = torch.mean(Tk_batch/OP.n_z)

            # Backprop
            loss.backward()
            p_batch_old=p_batch
            # Step
            optim.step()
            if use_lr_scheduler:
                lr_scheduler.step(loss.item())
                #if epoch < 300:
                #    optim.param_groups[0]['lr'] = 1e-5
                #    optim.param_groups[0]['initial_lr'] = 1e-5

            toc = time.perf_counter()
            step_time = toc-tic

            # end of epoch / history
            train_logger.log_value(epoch,"epoch")
            train_logger.log_value(loss.item(),"loss")
            train_logger.log_value(optim.param_groups[0]["lr"],"lr")
            train_logger.log_value(step_time,"step_time")
            train_logger.log_value(sum(weight_norms(self.model)),"weight_norm_sum")
            
            # Tk stats
            train_logger.log_metrics(Tk_batch,"Tk")
            
            # Print
            if (epoch+1) % print_frequency == 0:
                train_logger.print_last_entry(keys=["epoch","loss","Tk_mean"])
                print("-------------------------------")

            # Early Stopping
            if early_stop & use_lr_scheduler:
                if optim.param_groups[0]["lr"] <= min_lr:
                    break
            #p_batch=OP.update_x_batch(n_batch,z_hat_batch.detach(),p_batch.detach())
        # END OF TRAINING
        train_toc = time.perf_counter()
        train_time = train_toc - train_tic
        train_logger.log_data(train_time,"train_time")

        print("#----------------------------------#")
        print("Training complete.")
        print("Total number of epochs: {0}".format(epoch+1))
        print("Total training time: {0:.2f} sec".format(train_time))
        print("#----------------------------------#")

        return train_logger

    # Export
    def export_jit_gpu(self):
        mod_gpu = copy.deepcopy(self.model).cuda()
        mod_gpu = torch.jit.script(mod_gpu)
        mod_gpu = torch.jit.optimize_for_inference(mod_gpu)
        return mod_gpu
    
    def export_jit_cpu(self):
        mod_cpu = copy.deepcopy(self.model).cpu()
        mod_cpu = torch.jit.script(mod_cpu)
        mod_cpu = torch.jit.optimize_for_inference(mod_cpu)
        return mod_cpu

# Solver
class Solver(torch.nn.Module):
    def __init__(self,model,OP):
        super().__init__()
        self.model = model #step model
        def custom_weight_init(m):
            if isinstance(m, nn.Linear):
                # Set weights with a uniform distribution between -1/sqrt(n) and 1/sqrt(n),
                # where n is the number of input features. This helps in keeping the
                # output in range.
                n = m.weight.size(1)  # number of input features
                limit = 1 / torch.sqrt(torch.tensor(n*10, dtype=torch.float32))
                nn.init.uniform_(m.weight, -limit, limit)

                # Optionally, you can set the bias to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(custom_weight_init)
        self.model.apply(custom_weight_init)
        self.OP = OP
        self.offset = 1e-16
        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def forward(self,zk,x):
        # modified KKT conditions (Fk_batch)
        #batch_size=len(x)

        #x = x.repeat(batch_size, 1)
        #zk_batch_new = zk_batch + dzk_batch
        #zk = zk.repeat_interleave(batch_size, dim=0)
        Fk_batch = self.OP.Fk_batch_func(zk,x)
        #Fk_batch=self.OP.Fk_batch_func(zk,x).reshape(batch_size, -1)
        # 2-norm of modified KKT conditions and log-scaled 2-norm (norm_Fk_batch, norm_Fk_batch_log)
        norm_Fk_batch = (torch.linalg.vector_norm(Fk_batch,ord=2,dim=1)+self.offset).unsqueeze(-1)
        norm_Fk_batch_log = torch.log(norm_Fk_batch)

        # Normalize Fk_batch (Fk_batch_normalized)
        Fk_batch_normalized = torch.divide(Fk_batch,norm_Fk_batch)

        # Stack NN inputs: parameters of OP, normalized Fk_batch, log-scaled 2-norm of Fk_batch
        nn_inputs_batch = torch.hstack((x,Fk_batch_normalized,norm_Fk_batch_log))

        dzk = self.model(nn_inputs_batch)*norm_Fk_batch#*torch.where(norm_Fk_batch < 1,norm_Fk_batch, torch.tensor(1.0))#torch.max(norm_Fk_batch,1)#*torch.tensor()

        return dzk
    def forward_fk(self,zk,x,Fk_batch):
        # modified KKT conditions (Fk_batch)
        #batch_size=len(x)

        #x = x.repeat(batch_size, 1)
        #zk_batch_new = zk_batch + dzk_batch
        #zk = zk.repeat_interleave(batch_size, dim=0)
        #Fk_batch = self.OP.Fk_batch_func(zk,x)
        #Fk_batch=self.OP.Fk_batch_func(zk,x).reshape(batch_size, -1)
        # 2-norm of modified KKT conditions and log-scaled 2-norm (norm_Fk_batch, norm_Fk_batch_log)
        norm_Fk_batch = (torch.linalg.vector_norm(Fk_batch,ord=2,dim=1)+self.offset).unsqueeze(-1)
        norm_Fk_batch_log = torch.log(norm_Fk_batch)

        # Normalize Fk_batch (Fk_batch_normalized)
        Fk_batch_normalized = torch.divide(Fk_batch,norm_Fk_batch)

        # Stack NN inputs: parameters of OP, normalized Fk_batch, log-scaled 2-norm of Fk_batch
        nn_inputs_batch = torch.hstack((x,Fk_batch_normalized,norm_Fk_batch_log))

        dzk = self.model(nn_inputs_batch)*norm_Fk_batch#*torch.where(norm_Fk_batch < 1,norm_Fk_batch, torch.tensor(1.0))#torch.max(norm_Fk_batch,1)#*torch.tensor()

        return dzk
    ### Save/Load Methods
    def save_weights(self,save_dir, file_name = "weights.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self.state_dict(),save_pth)
        print("Weights (state dict) saved to: ", save_pth)

    def load_weights(self,load_dir, file_name = "weights.pt"):
        load_pth = Path(load_dir,file_name)
        weights = torch.load(load_pth)
        self.load_state_dict(weights)
        print("Weights (state dict) loaded from: ", load_pth)

    def save_model(self,save_dir,file_name = "model.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self.model,save_pth)
        print("Model saved to: ", save_pth)

    @staticmethod
    def load_model(load_dir,file_name = "model.pt"):
        load_pth = Path(load_dir,file_name)
        model = torch.load(load_pth)
        print("Model loaded from: ", load_pth)
        return model

    # Train
    def train(self,train_config,exp_pth,predictor=None,print_frequency=10,train_logger=None,loss_mode="explicit"):
        # CONFIG    

        N_epochs = train_config["N_epochs"]
        batch_size = train_config["batch_size"]
        weight_decay = train_config["weight_decay"]
        lr = train_config["lr"]
        log_loss = train_config["log_loss"]
        n_steps_max = train_config["n_steps_max"]
        alpha_0 = train_config["alpha"]
        append_delay = train_config["append_delay"]
        Tk_lim = train_config["Tk_lim"]
        update_mode = train_config["update_mode"]
        training_mode="False"#"Direct"
        alpha=alpha_0
        OP_old = load_OP("parametric_OP_data_robust/nonlinear_96x48x192_0")
        path = Path(__file__).parent.resolve()
        predictor_old = load_predictor_from_pth(
            path.joinpath("robust_case_study/predictor/nonlinear_96x48x192_0/exp_312"))  # 123#96#105
        pth = path.joinpath("robust_case_study/solver_with_pred/nonlinear_96x48x192_0/exp_464")  # 219#184#203
        solver_old = load_solver_from_pth(pth, OP_old)
        # backwards compatibility68
        if "convexified" in train_config.keys():
            convexified = train_config["convexified"]
        else:
            convexified = False
        
        if "delta" in train_config.keys():
            delta = train_config["delta"]
        else:
            delta = 1000.0
        
        # loss modes: "explicit" or "newton"
        # "explicit" --> explicit loss function (Tk_batch)
        # "newton" --> linearization of Fk at zk --> Newton type loss function: (Vk_batch), see Lüken L. and Lucia S. (2024)
        if "loss_mode" in train_config.keys():
            loss_mode = train_config["loss_mode"]

        # Logger
        if train_logger is None:
            train_logger = TrainLogger()
        else:
            train_logger = copy.deepcopy(train_logger)

        # Optimizer        
        optim = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)

        # TRAINING
        train_tic = time.perf_counter()
        if False:
            with torch.no_grad():
                t1=time.time()
                xk_batch_old=OP_old.batch_gen_x_random(batch_size)
                zk_batch_old = predictor_old.forward(xk_batch_old)
                Tk_batch_old, Fk_batch_old = OP_old.Tk_Fk_batch_func(zk_batch_old, xk_batch_old)

                for m in range(300):
                    dzk_batch = solver_old.forward_fk(zk_batch_old, xk_batch_old, Fk_batch_old)
                    zk1_batch = zk_batch_old + dzk_batch
                    # x = time.time()
                    # OP.Tk_func(zk1_batch[0],xk_batch[0])
                    Tk_new, Fk_new = OP_old.Tk_Fk_batch_func(zk1_batch, xk_batch_old)
                    # y = time.time()
                    # print(y - x)
                    # eta=Tk_new/Tk_batch
                    mask = ~(Tk_new > Tk_batch_old)
                    # zk_batch[mask]=zk1_batch[mask]
                    # count[mask]=count[mask]+1
                    # mask2=count<1
                    # else:
                    #    count=0
                    # if count<1:
                    zk_batch_old[mask] = zk1_batch[mask]
                    Tk_batch_old[mask] = Tk_new[mask]
                    Fk_batch_old[mask] = Fk_new[mask]
                t2=time.time()
                print(t2-t1)
            x_batch=self.OP.get_x_from_old(batch_size,zk_batch_old,xk_batch_old)
        else:
            if training_mode=="Direct":
                n_batch=int(batch_size/2187)
                x0 = (torch.tensor([[0.95], [0.95], [0.625], [0.625]], device=self.OP.device).T * torch.rand(n_batch,
                                                                                                        4)).repeat_interleave(
                    2187, dim=0) + torch.tensor([[0.05], [0.05], [0.375], [0.375]], device=self.OP.device).T.repeat(
                    (2187 * n_batch, 1))
                u_tilde = (torch.tensor([[0.95], [1]], device=self.OP.device).T * torch.rand(n_batch, 2)).repeat_interleave(2187,
                                                                                                                  dim=0) + torch.tensor(
                    [[0.05], [0]], device=self.OP.device).T.repeat((2187 * n_batch, 1))
                past_u = (torch.tensor([[0.95], [1]], device=self.OP.device).T * torch.cat(
                    (torch.rand(n_batch, 1), torch.randint(0, 2, (n_batch, 1))), axis=1)).repeat_interleave(2187,
                                                                                                            dim=0) + torch.tensor(
                    [[0.05], [0]],
                    device=self.OP.device).T.repeat((2187 * n_batch, 1))
                x_batch = self.OP.batch_gen_x_new(n_batch, x0, past_u, u_tilde)#self.OP.batch_gen_x_random(batch_size)
            else:
                x_batch=self.OP.batch_gen_x_random(batch_size)
        if predictor is None:
            zk_batch = self.OP.batch_gen_z(batch_size)
        else:
            zk_batch = predictor.predict(x_batch)
        #zk_batch=zk_batch/20
        #t_batch=self.OP.Tk_func(zk_batch[0],x_batch[0])
        T0_batch,Fk_batch_prev = self.OP.Tk_Fk_batch_func(zk_batch,x_batch)
        n_steps_batch = torch.zeros(batch_size)
        Tk_prev=T0_batch
        #num_bad=0
        counter=(T0_batch<torch.inf).int()
        Tk_best = T0_batch.detach().clone()

        no_improvement = torch.zeros(batch_size, dtype=torch.int32)

        # validation metrics
        n_p_seen_total = batch_size
        n_runs_converged = 0
        n_runs_max_iter = 0
        converged_ratio = n_runs_converged / n_p_seen_total
        max_iter_ratio = n_runs_max_iter / n_p_seen_total
        for epoch in range(N_epochs):
            if training_mode=="Direct":
                if epoch%4000==0 or any(Tk_batch>0.2):# or any(Tk_batch<1e-7):
                    x0 = (torch.tensor([[0.95], [0.95], [0.625], [0.625]], device=self.OP.device).T * torch.rand(
                        n_batch,
                        4)).repeat_interleave(
                        2187, dim=0) + torch.tensor([[0.05], [0.05], [0.375], [0.375]], device=self.OP.device).T.repeat(
                        (2187 * n_batch, 1))
                    u_tilde = (torch.tensor([[0.95], [1]], device=self.OP.device).T * torch.rand(n_batch,
                                                                                            2)).repeat_interleave(2187,
                                                                                                                  dim=0) + torch.tensor(
                        [[0.05], [0]], device=self.OP.device).T.repeat((2187 * n_batch, 1))

                    past_u = (torch.tensor([[0.95], [1]], device=self.OP.device).T * torch.cat(
                        (torch.rand(n_batch, 1), torch.randint(0, 2, (n_batch, 1))), axis=1)).repeat_interleave(2187,
                                                                                                                  dim=0) + torch.tensor(
                        [[0.05], [0]],
                        device=self.OP.device).T.repeat((2187 * n_batch, 1))

                    x_batch = self.OP.batch_gen_x_new(n_batch, x0, past_u, u_tilde)# self.OP.batch_gen_x_random(batch_size)
                    if predictor is None:
                        zk_batch = self.OP.batch_gen_z(batch_size)
                    else:
                        zk_batch = predictor.predict(x_batch)
                        # zk_batch=zk_batch/20
                        # t_batch=self.OP.Tk_func(zk_batch[0],x_batch[0])
                    T0_batch, Fk_batch_prev = self.OP.Tk_Fk_batch_func(zk_batch, x_batch)
                    n_steps_batch = torch.zeros(batch_size)
                    Tk_prev = T0_batch
                if epoch%400==399:
                    x_batch=self.OP.update_x_batch(n_batch,zk_batch.detach(),x_batch.detach())
                 #   num_bad+=1
            #    flag=False
            tic = time.perf_counter()            
            # Zero Grads
            optim.zero_grad(set_to_none=True)
            #model=self.model.clone()
            # PREDICT
            dzk_batch = self.forward_fk(zk_batch,x_batch,Fk_batch_prev)*alpha
            if loss_mode == "explicit":
                if convexified:
                    self.OP.Tk_Fk_conv_func(zk_batch[0]+dzk_batch[0],x_batch[0])
                    Tk_batch,Fk_batch = self.OP.Tk_Fk_conv_batch_func(zk_batch + dzk_batch,x_batch)
                else:
                    #Tk=self.OP.Tk_func(zk_batch[0]+dzk_batch[0],x_batch[0])
                    #Tk=self.OP.Tk_func(zk_batch[0]+dzk_batch[0],x_batch[0])
                    Tk_batch,Fk_batch=self.OP.Tk_Fk_batch_func(zk_batch+dzk_batch,x_batch)

                    #Tk_batch=self.OP.Tk_batch_func(zk_batch, x_batch)
                    #x_batch_new = x_batch.repeat(batch_size, 1)
                    #zk_batch_new=zk_batch+dzk_batch
                    #zk_batch_new = zk_batch_new.repeat_interleave(batch_size, dim=0)
                    #Tk_batch = self.OP.Tk_batch_func(zk_batch_new,x_batch_new)

            elif loss_mode == "newton":
                with torch.no_grad():
                    gamma_batch = self.OP.gamma_batch_func(zk_batch,x_batch,dzk_batch)
                    gamma_batch = torch.clamp(gamma_batch,1e-4,10.0)#.clone()
                dzk_batch = gamma_batch[:,None]*dzk_batch
                Tk_batch = self.OP.Vk_batch_func(zk_batch,x_batch,dzk_batch)
                #Tk_batch = Vk_batch
            else:
                raise ValueError("Loss mode not implemented.")

            # choose between two loss functions (mean of log-scaled Tk_batch or mean of Tk_batch)
            #assert not torch.isnan(Tk_batch).any(), "Tk_batch contains nan"
                # Calculate improvement factor over initial iterate (eta_total = Tk/T0)
            with torch.no_grad():
                eta_total_batch = Tk_batch / T0_batch

            if log_loss:
                mask=~torch.isnan(Tk_batch) & ~torch.isinf(Tk_batch) #& (eta_total_batch<3)
                loss = torch.mean(torch.log10(Tk_batch[mask]/self.OP.n_z+self.offset))
                #loss = Variable(loss, requires_grad=True)
            else:
                loss = torch.mean(Tk_batch/self.OP.n_z)
            #grads={}
            #def store(grad, parent):
            #    #print(grad, parent)
            #    grads[parent] = grad.clone()


            #dzk_batch.register_hook(lambda grad: store(grad, dzk_batch))

            ## Backprop
            loss.backward()
            #d=grads[dzk_batch]
            # Step

            optim.step()

            with torch.no_grad():
                # Update iterates using different update modes
                # 1. gamma: step size is determined by exact line search, assuming linearization of Tk_batch --> cannot achieve better steps than Newtons method (see Lüken and Lucia, 2024)
                # 2. full: update every iterate with dzk_batch --> divergence possible, computationally cheaper than other methods
                # 3. eta: update only if Tk+1 is smaller than Tk --> might be too conservative, prevents iterates from getting worse than previous iterate
                # 4. eta_total: update only if Tk+1 is smaller than T0 --> prevents iterates from getting worse than initial iterate (used in paper)
                if epoch >= append_delay:
                    if update_mode == "gamma":
                        if loss_mode == "newton":
                            # zk_batch = zk_batch + dzk_batch
                            mask = eta_total_batch < delta
                            zk1_batch = zk_batch + dzk_batch
                            zk_batch[mask] = zk1_batch[mask]
                        else:
                            with torch.no_grad():
                                gamma_batch = self.OP.gamma_batch_func(zk_batch,x_batch,dzk_batch)
                                gamma_batch = torch.clamp(gamma_batch,1e-4,10.0)
                                zk_batch = zk_batch + gamma_batch[:,None]*dzk_batch
                    elif update_mode == "full":
                        zk_batch = zk_batch + dzk_batch
                    elif update_mode == "eta":
                        #Tk_prev = self.OP.Tk_batch_func(zk_batch, x_batch)
                        #counter+=~(Tk_batch<Tk_prev)
                        #mask=(counter<1)
                        mask_2=(Tk_batch<Tk_prev)
                        counter[mask_2]=0
                        Tk_prev[mask_2]=Tk_batch[mask_2]
                        Fk_batch_prev[mask_2]=Fk_batch[mask_2]
                        #mask = ((Tk_batch < Tk_prev) &  ~torch.isnan(Tk_batch) & ~torch.isinf(Tk_batch))
                        zk_batch[mask_2] = zk_batch[mask_2] + dzk_batch[mask_2]
                    elif update_mode == "eta_total":
                        #mask = ((eta_total_batch < delta)&~torch.isnan(Tk_batch) & ~torch.isinf(Tk_batch))
                        zk1_batch = zk_batch+dzk_batch + self.forward(zk_batch+dzk_batch,x_batch)
                        T_batch=self.OP.Tk_batch_func(zk1_batch,x_batch)
                        mask =  ((eta_total_batch < delta)&~torch.isnan(T_batch) & ~torch.isinf(T_batch))
                        zk_batch[mask] = zk_batch[mask]+dzk_batch[mask]
                    else:
                        raise ValueError("Update mode not implemented.")
                    n_steps_batch += 1

                # If the iterates have reached Tk_lim, the iterates are resampled.
                # The solver is considered to have converged if the iterates have reached Tk_lim.
                # Tk is effectively the squared 2-norm of the modified KKT conditions. Therefore this is equivalent to a tolerance on the modified KKT conditions of sqrt(Tk_lim).
                mask_best = Tk_batch < Tk_best
                Tk_best[mask_best] = Tk_batch[mask_best]
                no_improvement[~mask_best] += 1
                no_improvement[mask_best] = 0
                idx_resample_tol = torch.where(Tk_batch < Tk_lim)[0]
                #idx_resample_safe=torch.where(Tk_batch >0.1)[0]
                idx_resample_max_iter = torch.where(n_steps_batch >= n_steps_max)[0]#torch.where(no_improvement >= n_steps_max)[0]#torch.where(n_steps_batch >= n_steps_max)[0]
                idx_resample = torch.unique(torch.hstack((idx_resample_tol,idx_resample_max_iter)))

                if training_mode!="Direct" and len(idx_resample) > 0:
                    x_batch[idx_resample] = self.OP.batch_gen_x_random(len(idx_resample))
                    counter[idx_resample]=0
                    if predictor is None:
                        zk_batch[idx_resample] = self.OP.batch_gen_z(len(idx_resample))
                    else:
                        zk_batch[idx_resample] = predictor.predict(x_batch[idx_resample])

                    if convexified:
                        T0_batch[idx_resample],Fk_batch_prev[idx_resample] = self.OP.Tk_Fk_conv_batch_func(zk_batch[idx_resample],x_batch[idx_resample])
                    else:
                        T0_batch[idx_resample], Fk_batch_prev[idx_resample] = self.OP.Tk_Fk_batch_func(
                            zk_batch[idx_resample], x_batch[idx_resample])
                    Tk_prev[idx_resample] = T0_batch[idx_resample]
                    n_steps_batch[idx_resample] = 0
                    # update Tk_best
                    Tk_best[idx_resample] = T0_batch[idx_resample]
                    no_improvement[idx_resample] = 0

                # validation metrics
                n_p_seen_total += len(idx_resample)
                n_runs_converged += len(idx_resample_tol)
                n_runs_max_iter += len(idx_resample_max_iter)
                converged_ratio = n_runs_converged / n_p_seen_total
                max_iter_ratio = n_runs_max_iter / n_p_seen_total

                toc = time.perf_counter()
                step_time = toc-tic

                # end of epoch / history
                train_logger.log_value(epoch,"epoch")
                train_logger.log_value(loss,"loss")
                train_logger.log_value(step_time,"step_time")
                train_logger.log_value(len(idx_resample),"n_resample")
                train_logger.log_value(converged_ratio, "converged_ratio")
                train_logger.log_value(max_iter_ratio, "max_iter_ratio")
                # stats
                train_logger.log_metrics(Tk_batch,"Tk")
                train_logger.log_metrics(eta_total_batch,"eta")
                train_logger.log_metrics(n_steps_batch,"n_steps")
                train_logger.log_value(sum(weight_norms(self.model)),"sum_nn_weight_norms")
                if (epoch+1) % 1000 == 0:
                    self.save_model(exp_pth, file_name="solver_model.pt")
                    self.save_weights(exp_pth, file_name="solver_weights.pt")
                # Print
                if (epoch+1) % print_frequency == 0:

                    train_logger.print_last_entry(keys=["epoch","loss","Tk_99","Tk_50","Tk_min","n_steps_max","n_steps_50","n_resample","converged_ratio","max_iter_ratio"])
                    print("-------------------------------")

        train_toc = time.perf_counter()
        train_time = train_toc - train_tic
        train_logger.log_data(train_time,"train_time")

        print("#----------------------------------#")
        print("Training complete.")
        print("Total number of epochs: {0}".format(epoch+1))
        print("Total training time: {0:.2f} sec".format(train_time))
        print("#----------------------------------#")

        return train_logger

    def get_solver_step_funcs(self,alpha=1e-2,jit=False,use_cuda=True,batch_size=100):
        OP = copy.deepcopy(self.OP)
        model = copy.deepcopy(self.model)

        if use_cuda:
            OP.move_to_device("cuda")
            OP._setup_functions()
            model.cuda()
            device = "cuda"
        else:
            OP.move_to_device("cpu")
            OP._setup_functions()
            model.cpu()
            device = "cpu"

        if jit:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
            if batch_size is None:
                dummy_x = torch.randn(OP.n_x).to(device)
                dummy_z = torch.randn(OP.n_z).to(device)
                Fk_func = torch.jit.trace(OP.Fk_func,(dummy_z,dummy_x))
            else:
                dummy_x = torch.randn(batch_size,OP.n_x).to(device)
                dummy_z = torch.randn(batch_size,OP.n_z).to(device)
                Fk_batch_func = torch.jit.trace(OP.Fk_batch_func,(dummy_z,dummy_x))
        else:
            if batch_size is None:
                Fk_func = lambda zk,x: OP.Fk_func(zk,x)
            else:
                Fk_batch_func = lambda zk,x: OP.Fk_batch_func(zk,x)

        if batch_size is None:
            @torch.no_grad()
            def solve_step(zk,x):
                Fk = Fk_func(zk,x)
                norm_Fk = (torch.linalg.vector_norm(Fk,ord=2,dim=0)+1e-16)#.unsqueeze(-1)
                norm_Fk_log = torch.log(norm_Fk)
                Fk_normalized = torch.divide(Fk,norm_Fk)
                nn_inputs_batch = torch.hstack((x,Fk_normalized,norm_Fk_log))
                dzk = model(nn_inputs_batch)*norm_Fk*alpha
                Tk_batch = 0.5*torch.square(norm_Fk)
                return dzk,Tk_batch
        else:
            @torch.no_grad()
            def solve_step(zk,x):
                Fk_batch = Fk_batch_func(zk,x)
                norm_Fk_batch = (torch.linalg.vector_norm(Fk_batch,ord=2,dim=1)+1e-16).unsqueeze(-1)
                norm_Fk_batch_log = torch.log(norm_Fk_batch)
                Fk_batch_normalized = torch.divide(Fk_batch,norm_Fk_batch)
                nn_inputs_batch = torch.hstack((x,Fk_batch_normalized,norm_Fk_batch_log))
                dzk = model(nn_inputs_batch)*norm_Fk_batch*alpha
                Tk_batch = 0.5*torch.square(norm_Fk_batch)
                return dzk,Tk_batch
        return solve_step
    
    @torch.no_grad()
    def solve_fast_single(self,solve_step_func,x,predictor=None,max_iter=100,tol=1e-16,use_cuda=True,update_mode="line_search"):
        success = False
        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"
        if predictor is None:
            zk = self.OP.batch_gen_z(1).squeeze().to(device)
        else:
            zk = predictor(x).squeeze().to(device)
        
        if update_mode == "eta":
            for i in range(max_iter):
                dzk,Tk = solve_step_func(zk,x)
                zk1 = zk + dzk
                # tol check
                if Tk < tol:
                    success = True
                    break
                if i == 0:
                    z_prev = zk
                    Tk_prev = Tk
                else:
                    if Tk > Tk_prev:
                        zk1 = z_prev
                        break
                    z_prev = zk
                    Tk_prev = Tk
                # update
                zk = zk1

        elif update_mode == "full":
            for i in range(max_iter):
                dzk,Tk = solve_step_func(zk,x)
                # tol check
                if Tk < tol:
                    success = True
                    break
                # update
                zk = zk + dzk
        
        elif update_mode == "eta_total":
            dz0,T0 = solve_step_func(zk,x)
            zk = zk + dz0
            for i in range(max_iter-1):
                dzk,Tk = solve_step_func(zk,x)
                if Tk < tol:
                    success = True
                    break
                if Tk > T0:
                    success = False
                    break
                zk = zk + dzk

        elif update_mode == "line_search":
                alpha = 1.0
                dz0,T0 = solve_step_func(zk,x)
                Tbest = T0
                zk_best = zk
                zk = zk + dz0*alpha
                for i in range(max_iter-1):
                    dzk,Tk = solve_step_func(zk,x)                    
                    if Tk < Tbest:
                        Tbest = Tk
                        zk_best = zk
                    if Tk < tol:
                        success = True
                        break
                    # update
                    zk = zk + dzk*alpha
                    if Tk > 10*Tbest:
                        # reset to best & reduce alpha
                        zk = zk_best
                        alpha = alpha*0.95
                    if alpha < 1e-8:
                        break
                zk = zk_best
        else:
            raise ValueError("Update mode not implemented.")
        
        return zk,i+1,success

    @torch.no_grad()
    def solve_batch(self,x_batch,max_iter=100,alpha=1.0,predictor=None,update_mode="full",return_trajectory=False,delta=1000.0):
        alpha_batch = torch.ones(x_batch.shape[0],device=x_batch.device)*alpha
        # initial batch
        n_batch = x_batch.shape[0]
        if predictor is None:
            zk_batch = self.OP.batch_gen_z(n_batch)
        else:
            zk_batch = predictor.predict(x_batch)

        if update_mode in ["eta_total","line_search"]:
            T0_batch = self.OP.Tk_batch_func(zk_batch,x_batch)
            zk_best = zk_batch.detach().clone()
            Tk_best = T0_batch.detach().clone()
        elif update_mode == "eta":
            Tk_prev = self.OP.Tk_batch_func(zk_batch,x_batch)
        else:
            pass

        if return_trajectory:
            zk_traj = [zk_batch.detach().clone()]
        
        step_time_list = []
        # LOOP
        for i in range(max_iter):
            step_tic = time.perf_counter()
            dzk_batch = alpha_batch[:,None,]*self.forward(zk_batch,x_batch)
            if update_mode == "gamma":
                gamma_batch = self.OP.gamma_batch_func(zk_batch,x_batch,dzk_batch)
                gamma_batch = torch.clamp(gamma_batch,1e-4,10.0)
                zk_batch = zk_batch + gamma_batch[:,None]*dzk_batch
            elif update_mode == "full":
                zk_batch = zk_batch + dzk_batch
            elif update_mode == "eta":
                Tk_batch = self.OP.Tk_batch_func(zk_batch + dzk_batch,x_batch)
                mask = Tk_batch < Tk_prev
                zk_batch[mask] = zk_batch[mask] + dzk_batch[mask]
                Tk_prev[mask] = Tk_batch[mask]
            elif update_mode == "eta_total":
                Tk_batch = self.OP.Tk_batch_func(zk_batch + dzk_batch,x_batch)
                eta_total_batch = Tk_batch/T0_batch
                mask = eta_total_batch < delta # Note: this factor is a tuning parameter, which can be adjusted but a sufficiently large value is enough to prevent divergence, has no effect, if other mode is used.
                zk_batch[mask] = zk_batch[mask] + dzk_batch[mask]
                # update best iterates
                mask_best = Tk_batch < Tk_best
                zk_best[mask_best] = zk_batch[mask_best].detach().clone()
                Tk_best[mask_best] = Tk_batch[mask_best].detach().clone()
                # inverse mask: reached limit
                # mask = ~mask
                mask_diverged = Tk_batch > 10*Tk_best
                # reset to best
                zk_batch[mask_diverged] = zk_best[mask_diverged].detach().clone()

            elif update_mode == "line_search":
                # update
                zk_batch = zk_batch + dzk_batch
                Tk_batch = self.OP.Tk_batch_func(zk_batch,x_batch)
                # update best iterates
                mask_best = Tk_batch < Tk_best
                zk_best[mask_best] = zk_batch[mask_best].detach().clone()
                Tk_best[mask_best] = Tk_batch[mask_best].detach().clone()
                # diverging
                mask_diverged = Tk_batch > 10*Tk_best
                # reset to best + reduce alpha
                zk_batch[mask_diverged] = zk_best[mask_diverged].detach().clone()
                alpha_batch[mask_diverged] = alpha_batch[mask_diverged]*0.95

            else:
                raise ValueError("Update mode not implemented.")

            step_toc = time.perf_counter()
            step_time = step_toc - step_tic
            step_time_list.append(step_time)
            if return_trajectory:
                zk_traj.append(zk_batch.detach().clone())

        # END OF SOLVE
        if update_mode in ["eta_total","line_search"]:
            if return_trajectory:
                zk_traj.append(zk_best.detach().clone())
                return zk_best, zk_traj, step_time_list
            else:
                return zk_best, step_time_list
        else:
            if return_trajectory:
                return zk_batch, zk_traj, step_time_list
            else:
                return zk_batch, step_time_list

