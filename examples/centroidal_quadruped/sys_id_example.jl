using Revise
using RoboDojo
using JLD2
using Plots
include("sys_id.jl")

# ## Initial conditions
# q1 = nominal_configuration(centroidal_quadruped_param) 
model_path = joinpath(dirname(pathof(RoboDojo)), "robots/centroidal_quadruped_param", "model.jl")
sim_path = joinpath(dirname(pathof(RoboDojo)), "robots/centroidal_quadruped_param", "simulator.jl")

include(model_path)
include(sim_path)

r_model, rz_model, rθ_model = codegen_residual(centroidal_quadruped_param, 
                                               codegen_dynamics(centroidal_quadruped_param)..., 
                                               centroidal_quadruped_param_contact_kinematics, 
                                               centroidal_quadruped_param_contact_kinematics_jacobians)
# residual_expr(centroidal_quadruped_param)
RESIDUAL_EXPR[String(name(centroidal_quadruped_param)) * "_r"] = eval(r_model)
RESIDUAL_EXPR[String(name(centroidal_quadruped_param)) * "_rz"] = eval(rz_model)
RESIDUAL_EXPR[String(name(centroidal_quadruped_param)) * "_rθ"] = eval(rθ_model)

residual_expr(centroidal_quadruped_param)
jacobian_var_expr(centroidal_quadruped_param)
jacobian_data_expr(centroidal_quadruped_param)

# ## 
# Load a trajectory
@load joinpath(@__DIR__, "sample_trajectory.jld2") q_traj v_traj u_traj inertia_body mass_body h_sim
T = length(q_traj)
x_traj = [[q_traj[i]..., v_traj[i]...] for i in 1:T-1]

# ##
vis = Visualizer()
open(vis)
visualize!(vis, centroidal_quadruped_param, q_traj; Δt=h_sim)

# ## 
# Running state param estimation
sim = Simulator(centroidal_quadruped_param, 1; diff_sol=true, h=h_sim)
Q = Diagonal(ones(36)*1e2)
Q[7:18, 7:18] .= 0
# Q[25:end, 25:end] .= 0
R = Diagonal(ones(12))
w_init = [mass_body-5.0,  inertia_body[1,1], inertia_body[2,2], inertia_body[3,3]]
w_gt = [mass_body,  inertia_body[1,1], inertia_body[2,2], inertia_body[3,3]]
w = copy(w_init)

T = 350
w_hist = zeros(size(w,1), T)
horizon = 5
for i in 1:T 
    w = est_param(sim, x_traj[i:i+horizon], u_traj[i:i+horizon]./h_sim, w, Q, R; iterations = 3, α=0.1)
    w_hist[:,i] = w
    println(w)
end 

w_error = w_hist .- w_gt
p = plot(collect(1:T) .* h_sim, w_error', labels=["mass" "Ixx" "Iyy" "Izz"])
title!(p, "parameter estimation error")
xlabel!(p, "time (s)")