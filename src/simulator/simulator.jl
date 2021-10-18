# @with_kw struct SimulatorOptions{T}
#     warmstart::Bool = true
#     z_warmstart::T = 0.001
#     κ_warmstart::T = 0.001
# 	failure_abort::Int = 50
# end

# mutable struct SimulatorStatistics{T}
#     dt::AbstractVector{T}
#     μ_dt::T
#     σ_dt::T
# end

# function SimulatorStatistics()
#     dt = zeros(0)
#     μ_dt = 0.0
#     σ_dt = 0.0
#     return SimulatorStatistics(dt, μ_dt, σ_dt)
# end

# struct Simulator{T}
#     s::Simulation

#     traj::ContactTraj
#     deriv_traj::ContactDerivTraj

#     p::Policy
#     uL::Vector{T}
#     uU::Vector{T}

#     d::Disturbances

#     ip::InteriorPoint

#     opts::SimulatorOptions{T}
#     stats::SimulatorStatistics{T}
# end

# function simulator(s::Simulation, q0::SVector, q1::SVector, h::S, H::Int;
#     p = no_policy(s.model),
#     uL = -Inf * ones(s.model.dim.u),
#     uU = Inf * ones(s.model.dim.u),
#     d = no_disturbances(s.model),
#     r! = s.res.r!,
#     rz! = s.res.rz!,
#     rθ! = s.res.rθ!,
#     rz = s.rz,
#     rθ = s.rθ,
#     space = Euclidean(num_var(s.model, s.env)),
#     ip_opts = InteriorPointOptions(
# 		undercut = Inf,
# 		γ_reg = 0.0,
# 		r_tol = 1e-8,
# 		κ_tol = 1e-8,
# 		),
#     sim_opts = SimulatorOptions{S}()) where S

#     model = s.model
#     env = s.env

#     # initialize trajectories
#     traj = contact_trajectory(model, env, H, h)
#     traj.q[1] = q0
#     traj.q[2] = q1

#     # initialize interior point solver (for pre-factorization)
#     z = zeros(num_var(model, s.env))
#     θ = zeros(num_data(model))
#     z_initialize!(z, model, env, traj.q[2])
#     θ_initialize!(θ, model, traj.q[1], traj.q[2], traj.u[1], traj.w[1], model.μ_world, h)
    
#     ip = interior_point(
# 			 z,
# 			 θ,
# 			 s = space,
# 			 idx = IndicesOptimization(model, env),
# 			 r! = r!,
# 			 rz! = rz!,
# 			 rθ! = rθ!,
# 			 rz = rz,
# 			 rθ = rθ,
# 			 opts = ip_opts)

#     # pre-allocate for gradients
#     traj_deriv = contact_derivative_trajectory(model, env, ip.δz, H)
#     stats = SimulatorStatistics()

#     Simulator(
#         s,
#         traj,
#         traj_deriv,
#         p, uL, uU,
#         d,
#         ip,
#         sim_opts,
#         stats)
# end

# function step!(sim::Simulator, t)
#     # simulation
#     model = sim.s.model
#     env = sim.s.env

#     # unpack
#     q = sim.traj.q
#     u = sim.traj.u
#     w = sim.traj.w
#     h = sim.traj.h
#     ip = sim.ip
#     z = ip.z
#     θ = ip.θ

#     # t = 1 2 3
#     # u1 = traj.u[t]
#     # w1 = traj.w[t]
#     # γ1 = traj.γ[t]
#     # b1 = traj.b[t]
#     # q0 = traj.q[t]
#     # q1 = traj.q[t+1]
#     # q2 = traj.q[t+2]

#     # policy
#     dt = @elapsed u[t] = control_saturation(policy(sim.p, q[t+1], sim.traj, t), sim.uL, sim.uU)
#     push!(sim.stats.dt, dt)

#     # disturbances
#     w[t] = disturbances(sim.d, q[t+1], t)

#     # initialize
#     if sim.opts.warmstart
#         z_warmstart!(z, model, env, q[t+1], sim.opts.z_warmstart)
#         # sim.ip.opts.κ_init = sim.opts.κ_warmstart
#     else
#         z_initialize!(z, model, env, q[t+1])
#     end
#     θ_initialize!(θ, model, q[t], q[t+1], u[t], w[t], model.μ_world, h)

#     # solve
#     status = interior_point_solve!(ip)

#     if status
#         # parse result
#         q2, γ, b, _ = unpack_z(model, env, z)
#         sim.traj.z[t] = copy(z) # we need a copy here otherwize we get a constant trajectory = last computed z element # TODO: maybe not use copy
#         sim.traj.θ[t] = copy(θ) # we need a copy here otherwize we get a constant trajectory = last computed θ element
#         sim.traj.q[t+2] = q2
#         sim.traj.γ[t] = γ
#         sim.traj.b[t] = b
#         sim.traj.κ[1] = ip.κ[1]

#         if sim.ip.opts.diff_sol
#             sim.deriv_traj.dq2dq0[t] = sim.deriv_traj.vqq
#             sim.deriv_traj.dq2dq1[t] = sim.deriv_traj.vqqq
#             sim.deriv_traj.dq2du[t] = sim.deriv_traj.vqu
#             sim.deriv_traj.dγdq0[t] = sim.deriv_traj.vγq
#             sim.deriv_traj.dγdq1[t] = sim.deriv_traj.vγqq
#             sim.deriv_traj.dγdu[t] = sim.deriv_traj.vγu
#             sim.deriv_traj.dbdq0[t] = sim.deriv_traj.vbq
#             sim.deriv_traj.dbdq1[t] = sim.deriv_traj.vbqq
#             sim.deriv_traj.dbdu[t] = sim.deriv_traj.vbu
#         end
#     end

#     return status
# end

# """
#     simulate
#     - solves 1-step feasibility problem for H time steps
#     - initial configurations: q0, q1
#     - time step: h
# """
# function simulate!(sim::Simulator; verbose = false)

#     verbose && println("\nSimulation")

#     # initialize configurations for first step
#     z_initialize!(sim.ip.z, sim.s.model, sim.s.env, sim.traj.q[2])

#     status = true

#     # simulate
#     for t = 1:sim.traj.H
#         verbose && println("t = $t / $(sim.traj.H)")
#         status = step!(sim, t)
#         !status && (@error "failed step (t = $t)")
# 		!status && break
#     end

#     return status
# end

