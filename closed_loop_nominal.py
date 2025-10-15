import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')
# local imports
from model_nominal import template_model_nom
from model_nominal_z import template_model_z
from template_mpc import template_mpc
from template_mpc_z import template_mpc_z
from simulator_CSTR import template_simulator
from model_CSTR import template_model
# user settings
show_animation = False
store_results = False

# setting up the model
model = template_model_nom()
model_z=template_model_z()
model_CSTR=template_model()
# setting up a mpc controller, given the model
mpc = template_mpc(model)
mpc_z = template_mpc_z(model_z)
# setting up a simulator, given the model
simulator = template_simulator(model_CSTR)

# setting up an estimator, given the model
estimator = do_mpc.estimator.StateFeedback(model)

np.random.seed(1)
# Initialize graphic:
graphics = do_mpc.graphics.Graphics(simulator.data)
fig, ax = plt.subplots(5, sharex=True)

# Configure plot:
# adding each lines in th plot
graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
graphics.add_line(var_type='_u', var_name='F', axis=ax[4])

# modifying the labels for the plot
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')

# Update properties for all prediction lines:
for line_i in graphics.pred_lines.full:
    line_i.set_linewidth(1)

label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()
fig.tight_layout()
plt.ion()

timer = Timer()
N_pred=20
num_viol=0
num_sim=10
x_sam_max=np.array([[1.7],[1.7],[135],[130]])
x_sam_min=np.array([[0.3],[0.3],[100],[100]])
cost_list=[]
time_list=[]
u0_min=np.array([[5],[-8500]])
u0_max=np.array([[100],[0]])
for s in range(num_sim):

    mpc.reset_history()
    mpc_z.reset_history()
    simulator.reset_history()
    # Set the initial state of mpc and simulator:
    C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
    C_b_0 = 0.5 # This is the controlled variable [mol/l]
    T_R_0 = 134 #[C]
    T_K_0 = 130.0 #[C]
    u0=np.random.uniform(u0_min,u0_max)#np.array([[5],[0]])
    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
    #u0=np.array([[5],[-8500]])
    # pushing initial condition to mpc and the simulator
    mpc.x0 = x0
    mpc_z.x0=x0
    #mpc_z.u0=np.array([-8500])
    mpc.u0=u0
    mpc_z.u0=np.array(u0[0][0])
    simulator.x0 = x0

    # setting up initial guesses
    mpc.set_initial_guess()
    mpc_z.set_initial_guess()
    simulator.set_initial_guess()
    cost=0
    u_app=u0
    u0_old=u0
    # simulation of the plant
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
    for k in range(40):
        u_app_old = u_app
        u0_old=u0
        mpc.u0=np.array([[u_app_old[0][0]],[u_app_old[1][0]/-8500]])
        # for the current state x0, mpc computes the optimal control action u0
        timer.tic()
        t1=time.time()
        u0 = mpc.make_step(x0)
        t2 = time.time()
        u_int = np.zeros((N_pred,1))
        u = mpc.data.prediction(('_u', 'Q_dot')).squeeze().reshape(-1, 1)/(-8500)
        time_list.append(t2-t1)

        summe = 0
        for i in range(len(u)):  # -1,-1,-1):
            #    # print("k:"+str(k)+" i:"+str(i))
            summe = summe + u[i]
            if summe > 0.5:
                summe -= 1
                u_int[i] = 1

        tvp_template_mpc = mpc_z.get_tvp_template()
        def tvp_fun_mpc(t_now):
            for s in range(mpc.settings.n_horizon):
                tvp_template_mpc['_tvp', s, 'Q_dot'] = u_int[s][0]
            return tvp_template_mpc
        mpc_z.set_tvp_fun(tvp_fun_mpc)
        #F=mpc_z.make_step(x0).squeeze()
        u_app=np.array([[u0[0][0]],[u0[1][0]*-8500]])#np.array([[F],[-8500*u_int[1][0]]])
        timer.toc()

        # for the current state u0, computes the next state y_next
        y_next = simulator.make_step(u_app)

        # for the current state y_next, estimates the next state x0
        x0 = estimator.make_step(y_next)
        cost+=((x0[1] / 2 - 0.3) ** 2 + (x0[0] / 2 - 0.35) ** 2 + 0.1 * (u0[0] / 100 - u0_old[0][
            0] / 100) ** 2)  # +1e-3*(u0[1][0]/8500-u0_old[1][0]/8500)**2)#(x0[2][0]/140-0.88)**2

        if x0[2]>140 or x0[3]>140:
            num_viol+=1
        # update the graphics
        if show_animation:
            graphics.plot_results(t_ind=k)
            #graphics.plot_predictions(t_ind=k)
            graphics.reset_axes()
            plt.show()
            plt.pause(0.01)
    cost_list.append(cost)

print(num_viol)
print(sum(cost_list)/len(cost_list))
print(sum(time_list)/len(time_list))
input("Press any key to exit.")