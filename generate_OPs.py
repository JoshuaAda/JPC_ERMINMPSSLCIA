# Specifications
n_problems = 1

obj_type = "nonlinear"
op_dims = [{"n_vars":96, "n_eq":48, "n_ineq":192}]

folder_name = "parametric_OP_data_robust"

sample_solutions = True
n_samples = 10

# Imports
from parametric_OP import parametricOP
from pathlib import Path
import torch

# default dtype
dtype = torch.float64
torch.set_default_dtype(dtype)

# default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# pths
file_pth = Path(__file__).parent.resolve()
op_pth = file_pth.joinpath(folder_name)

# OP specifications
op_dicts = []
for idx in range(n_problems):
    for op_dim in op_dims:
        op_dicts.append({"obj_type":obj_type, **op_dim,"idx":idx})

# Generate OPs
for op_dict in op_dicts:
    idx = op_dict["idx"]
    # setup OP to populate parameters Q,p,A,G,h
    op = parametricOP(obj_type=op_dict["obj_type"],n_vars=op_dict["n_vars"],n_eq=op_dict["n_eq"],n_ineq=op_dict["n_ineq"])

    op_name = f'{op.obj_type}_{op.n_vars}x{op.n_eq}x{op.n_ineq}_{idx}'
    folder_pth = op_pth.joinpath(op_name)
    if folder_pth.exists():
        print(f"parametric OP {op_name} already exists.")
        print(f"Skipping...")
    else:
        print(f"Generating {op_name}...")
        folder_pth.mkdir(parents=True,exist_ok=True)

        op.save_config(folder_path=folder_pth,file_name="op_cfg")

        print(f"Saving {op_name}...")
        print(f"parametric OP {op_name} saved.")
    print(f"------------------------------------------")
    print("\n\n")

# %% Sample solutions
if sample_solutions:
    for op_dict in op_dicts:
        op_name = f'{op_dict["obj_type"]}_{op_dict["n_vars"]}x{op_dict["n_eq"]}x{op_dict["n_ineq"]}_{op_dict["idx"]}'
        OP = parametricOP.from_json(folder_pth=op_pth.joinpath(op_name),file_name="op_cfg")
        if op_dict["obj_type"] == "quad":
            z_opt, x, full_solve_time, solve_time, iter_count = OP.sample_osqp_solutions(n_samples)
            data = {"z_opt": z_opt, "x": x, "full_solve_time":full_solve_time, "solve_time": solve_time, "iter_count": iter_count}
        else:
            z_opt, x, solve_time, iter_count = OP.sample_casadi(n_samples)
            data = {"z_opt": z_opt, "x": x, "solve_time": solve_time, "iter_count": iter_count}
        torch.save(data, op_pth.joinpath(op_name).joinpath("sample_data.pth"))
        KKT = OP.KKT_batch_func(z_opt, x)
        assert torch.allclose(KKT, torch.zeros_like(KKT), atol=1e-4), f"KKT not satisfied for {op_name}"

