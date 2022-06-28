function est_param(sim_model, x_traj, u_traj, w, Q, R; iterations=10, α=0.1)
    w̄ = copy(w)
    for i in 1:iterations
        cost_now, grad_now = cost_param(sim_model, x_traj, u_traj, w̄, Q)
        w̄ -= α * grad_now[1,:]
    end
    return w̄
end 

# Parameter estimation
function cost_param(sim, xs_, us_, w_init, Q)
    cost = 0 
    w = copy(w_init)
    J = zeros(1, length(w_init))
    for i in 1:length(xs_)-1
        reset!(sim.traj)
        reset!(sim.grad)
        dynamics(sim, xs_[i], us_[i], w, diff_sol=true, verbose=true)
        x̄ = [sim.traj.q[3] ; sim.traj.v[2]] 
        cost += (xs_[i+1] - x̄)' * Q * (xs_[i+1] - x̄) 

        # calculate the gradient 
        ∂cost∂dyn = -2*xs_[i+1]' * Q + 2*x̄'*Q
        ∂v1∂w1 = (sim.grad.∂q3∂v1[1]' * sim.grad.∂q3∂w1[1])
        # ∂v1∂w1 = sim.grad.∂q3∂w1[1] / hm 
        ∂q3∂w1 = sim.grad.∂q3∂w1[1] 
        ∂dyn∂w1 = [∂q3∂w1 ; ∂v1∂w1]
        J += (∂cost∂dyn) * ∂dyn∂w1 #+ 2 * w_init' * I(4)
    end 
    return cost, J
end 