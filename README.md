# Enabling robust mixed-integer nonlinear model predictive control via self-supervised learning and  combinatorial integral approximation
Under revision for Journal of Process Control
## How to use/Evaluate the results
a) Install the necessary dependencies:

```pip install -r requirements.txt```

b) Change model name in the parametric_OP.py init function to the model you want to evaluate (LISCO, Approx MPC, Robust MPC)

c) Run solver_eval as it currently is (10 trajectories with 40 time steps):

```python solver_eval.py```

d) For the nominal MPC run:

```python closed_loop_nominal.py```

e) Probabilistic validation: Set the initial values to random within the sampling space and calculate the number of trajectories dependent from r
