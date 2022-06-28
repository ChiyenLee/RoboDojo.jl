"""
    centroidal quadruped 
    q = (p, r, f1, f2, f3, f4) 
        p - body position
        r - body orientation (modified Rodriques parameters)
        f1 - foot 1 position 
        f2 - foot 2 position 
        f3 - foot 3 position 
        f4 - foot 4 position 
"""
mutable struct CentroidalQuadrupedParam{T} <: Model{T}
    # dimensions
	nq::Int # generalized coordinates 
    nu::Int # controls 
    nw::Int # parameters
    nc::Int # contact points

    # parameters
    mass_body::Symbolics.Num
    inertia_body::Matrix{Symbolics.Num} 
    mass_foot::T 

    # environment 
    friction_joint::Vector{T}
    friction_body_world::Vector{T} 
    friction_foot_world::Vector{T} 
    gravity::Vector{T}
end

function skew(x)
    return [0.0  -x[3]  x[2];
            x[3]   0.0 -x[1];
           -x[2]  x[1]   0.0]
end

function L_mult(x)
    [x[1] -transpose(x[2:4]); 
     x[2:4] x[1] * I(3) + skew(x[2:4])]
end

# right quaternion multiply as matrix
function R_mult(x)
    [x[1] -transpose(x[2:4]); x[2:4] x[1] * I(3) - skew(x[2:4])]
end

# rotation matrix
function quaternion_rotation_matrix(q) 
    H = [zeros(1, 3); I(3)]
    transpose(H) * L_mult(q) * transpose(R_mult(q)) * H
end

function quaternion_from_mrp(p)
    """Quaternion (scalar first) from MRP"""
    return (1.0 / (1.0 + dot(p, p))) * [(1 - dot(p, p)); 2.0 * p]
end

function mrp_rotation_matrix(x) 
    quaternion_rotation_matrix(quaternion_from_mrp(x))
end

#https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

function euler_rotation_matrix(θ)
	a = θ[1]
	b = θ[2]
	c = θ[3]

	[cos(a) * cos(b) (cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) (cos(a) * sin(b) * cos(c) + sin(a) * sin(c));
	 sin(a) * cos(b) (sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) (sin(a) * sin(b) * cos(c) - cos(a) * sin(c));
	 -sin(b) cos(b) * sin(c) cos(b) * cos(c)]
end

function mass_matrix(model::CentroidalQuadrupedParam, q)
    blockdiag(sparse(centroidal_quadruped_param.mass_body * Diagonal(ones(3))), 
    sparse(centroidal_quadruped_param.inertia_body), 
    sparse(centroidal_quadruped_param.mass_foot * Diagonal(ones(3 * 4))))
end

function dynamics_bias(model::CentroidalQuadrupedParam, q, q̇)
    [
        model.mass_body * model.gravity;            # body position
        skew(q̇[4:6]) * model.inertia_body * q̇[4:6]; # body orienation 
        model.mass_foot * model.gravity;
        model.mass_foot * model.gravity;
        model.mass_foot * model.gravity;
        model.mass_foot * model.gravity;
    ]
end

function parameter_update!(model::CentroidalQuadrupedParam, w) 
    model.mass_body = w[1]
    model.inertia_body[1,1] = w[2]
    model.inertia_body[2,2] = w[3]
    model.inertia_body[3,3] = w[4]
end 

function signed_distance(model::CentroidalQuadrupedParam, q)

    position_body = q[1:3] 
    orientation_body = q[3 .+ (1:3)]

    position_foot1 = q[6 .+ (1:3)]
    position_foot2 = q[9 .+ (1:3)]
    position_foot3 = q[12 .+ (1:3)]
	position_foot4 = q[15 .+ (1:3)]

    return [position_foot1[3]; position_foot2[3]; position_foot3[3]; position_foot4[3]; position_body[3]]
end

function input_jacobian(model::CentroidalQuadrupedParam, q)
    position_body = q[1:3]
    orientation_body = q[3 .+ (1:3)]
    # R = mrp_rotation_matrix(orientation_body)
    R = euler_rotation_matrix(orientation_body)
    
    # kinematics in world frame
	r1 = q[6 .+ (1:3)] - position_body
	r2 = q[9 .+ (1:3)] - position_body
	r3 = q[12 .+ (1:3)] - position_body
	r4 = q[15 .+ (1:3)] - position_body

	z3 = zeros(3, 3)

	transpose([
        I(3) I(3) I(3) I(3);
        transpose(R) * skew(r1) transpose(R) * skew(r2) transpose(R) * skew(r3) transpose(R) * skew(r4);
        -I(3)    z3    z3   z3;
        z3    -I(3)    z3   z3;
        z3       z3 -I(3)   z3;
        z3       z3    z3 -I(3)
    ])
end

function contact_jacobian(model::CentroidalQuadrupedParam, q) 
    z3 = zeros(3, 3)

    [
        z3   z3 I(3)   z3   z3   z3;
        z3   z3   z3 I(3)   z3   z3;
        z3   z3   z3   z3 I(3)   z3;
        z3   z3   z3   z3   z3 I(3);
        I(3) z3   z3   z3   z3   z3;
    ]
end

# nominal configuration 
function nominal_configuration(model::CentroidalQuadrupedParam) 
    [
        0.0; 0.0; 0.15; 
        0.0; 0.0; 0.0;
        0.1; 0.1; 0.0;
        0.1;-0.1; 0.0;
       -0.1; 0.1; 0.0;
       -0.1;-0.1; 0.0;
    ]
end

# friction coefficients 
function friction_coefficients(model::CentroidalQuadrupedParam) 
    return [model.friction_foot_world; model.friction_body_world]
end

function dynamics(model::CentroidalQuadrupedParam, mass_matrix, dynamics_bias, h, q0, q1, u1, w1, λ1, q2)
    # evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

    d = 0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2 # variational integrator (midpoint)
    d .+= transpose(input_jacobian(model, qm2)) * u1             # control inputs
    d .+= λ1                                                # contact impulses
    d .-= model.friction_joint .* vm2 .* h[1] # joint friction

    return d
end

# dimensions
nq = 3 + 3 + 3 * 4       # generalized coordinates
nu = 3 * 4               # controls
nw = 4                   # parameters
nc = 5                   # contact points 

# parameters
gravity = [0.0; 0.0; 9.81]                 # gravity
friction_body_world = [0.5]                # coefficient of friction
friction_foot_world = [0.5; 0.5; 0.5; 0.5] # coefficient of friction
friction_joint = [10ones(3); 30ones(3); 10ones(12)]


# inertial properties
@variables mass_body
@variables centroidal_inertia[1:3]
inertia_body = Array(Diagonal(centroidal_inertia))
mass_foot = 0.2

centroidal_quadruped_param = CentroidalQuadrupedParam(nq, nu, nw, nc,
				mass_body,
                inertia_body, 
                mass_foot,
                friction_joint,
                friction_body_world, 
                friction_foot_world, 
                gravity)

centroidal_quadruped_param_contact_kinematics = [
    q -> q[6  .+ (1:3)],
    q -> q[9  .+ (1:3)],
    q -> q[12 .+ (1:3)],
    q -> q[15 .+ (1:3)],
    q -> q[ 0 .+ (1:3)],
]

centroidal_quadruped_param_contact_kinematics_jacobians = [
    q -> [zeros(3, 6) I(3) zeros(3, 9)],
    q -> [zeros(3, 9) I(3) zeros(3, 6)],
    q -> [zeros(3, 12) I(3) zeros(3, 3)],
    q -> [zeros(3, 15) I(3)],
    q -> [I(3) zeros(3, 15)],
]

name(::CentroidalQuadrupedParam) = :centroidal_quadruped_param
floating_base_dim(::CentroidalQuadrupedParam) = 6