using GLMakie, Lux, Optimization, OptimizationEvolutionary, Optimisers, OptimizationMetaheuristics, Zygote
using Statistics, LinearAlgebra, Random, ForwardDiff

GLMakie.activate!(title="EM-Bottle-Trap")

const μ₀ = 4π * 10^-7
const mₐ = 6.644657230e-27  # Example value for alpha particle mass in kg
const qₐ = 3.2043533e-19  # Example value for alpha particle charge in C
const lim = 15
const dt = 10e-5

struct Dipole
    mag::Float64
    M::Vec3f
    m::Vec3f
    r::Point3f
    Dipole(mag, m, r) = new(mag, mag * m, m, r)
end

function bottleB(r, M1::Dipole, M2::Dipole)
    rA = r - M1.r
    rmagA = sqrt(dot(rA, rA))
    B1bottle = 3 * rA * dot(M1.M, rA) / rmagA^5
    B2bottle = -(M1.M) / rmagA^3

    rB = r - M2.r
    rmagB = sqrt(dot(rB, rB))
    B1bottle2 = 3 * rB * dot(M2.M, rB) / rmagB^5
    B2bottle2 = -(M2.M) / rmagB^3

    B_total = .+(B1bottle, B2bottle, B1bottle2, B2bottle2)

    return (μ₀ / (4 * π) .* B_total)
end

function RK2step(Bfield::Function, M1::Dipole, M2::Dipole, v::Vec3f, r::Point3f) :: Tuple{Point3f, Vec3f}
    dv1 = dt * (qₐ / mₐ) * cross(v, Bfield(r, M1, M2))
    dr1 = dt * v

    dv2 = dt * (qₐ / mₐ) * cross(v + dv1, Bfield(r + dr1, M1, M2))
    dr2 = dt * (v + dv1)

    return (r + (dr1 + dr2) / 2, v + (dv1 + dv2) / 2)
end

# Define the Loss function
function Loss(NN, st, param, input)
    output, st = NN(input, param, st)
    r = Point3f(output[1])
    v = Vec3f(output[2])
    loss = 0
    i = 1
    while (!any(x -> abs(x) >= lim, r))
        r1, v = RK2step(bottleB, M1, M2, v, r)
        r = r1
        i += 1
        loss -= i
        # loss -= dot(r, r)
    end
    return loss, st
end

function Train!(NN, trainIt, params, sts, optimizer)
    magnitude = rand(10e4:10e6, trainIt)
    for epoch in 1:trainIt
        M1 = Dipole(magnitude[epoch], M, [0.0, 10.0, 0.0])
        M2 = Dipole(magnitude[epoch], M, [0.0, -10.0, 0.0])
        inputs = vcat(M1.M..., M2.M...)  # Ensure input is defined
        inputs = Float32.(inputs)
        (loss, sts,), back = pullback(Loss, NN, sts, params, inputs)
        _, _, grad, _ = back((1.0, nothing))
        optimizer, params = Optimisers.update(optimizer, params, grad)
        push!(losshistory, loss)
        if epoch % 10 == 0
            println("Epoch: $epoch, Loss: $loss")
            predictions, _ = NN(input, params, sts)
            println("Predictions: ", predictions)
        end
    end
    
end

M = Vec3f(0, 1, 0) # orientation of the dipole
field = (r -> bottleB(r, M1, M2))

# Create a grid of points
x = y = z = range(-lim, lim, length=100)

space = [Point3f(x, y, z) for x in x, y in y, z in z]

# Initial conditions
rP = Point3f[]  # Adjust the size as needed
vP = Vec3f[]  # Adjust the size as needed
rng = Xoshiro(3242)

# Inputs: M, M. Output: x, y, z, vx, vx, vy, vz
NN = Chain(Dense(6 => 18, swish), Dense(18 => 31, relu), Dense(31 => 15, tanh),
    Parallel(nothing, Dense(15 => 6, x -> 2 * tanh(x)), Dense(15 => 6, x -> 8 * tanh(x))),
    Parallel(nothing, Dense(6 => 3), Dense(6 => 3))
)
params, sts = Lux.setup(rng, NN)
optimizer = Optimisers.setup(RMSProp(1e-3, 3e-2), params)

magnitude = rand(10e4:10e6)
M1 = Dipole(magnitude, M, [0.0, 10.0, 0.0])
M2 = Dipole(magnitude, M, [0.0, -10.0, 0.0])
input = vcat(M1.M..., M2.M...)  # Ensure input is defined
input = Float32.(input)
initpred, _ = NN(input, params, sts)

losshistory = []
Train!(NN, 1000, params, sts, optimizer)

i = 1
finalpred, _ = NN(input, params, sts)
push!(rP, finalpred[1])
push!(vP, finalpred[2])
while (!any(x -> abs(x) >= lim, rP[end]))
    r, v = RK2step(bottleB, M1, M2, vP[i], rP[i])
    push!(rP, r)
    push!(vP, v)
    i += 1
end

trajectory = Observable(Point3f[])

# Plot the field using quiver
fig = Figure()
ax = LScene(fig[1, 1], show_axis=false)
interval = (v -> minimum(v) .. maximum(v))
streamplot!(ax, field, interval(x), interval(y), interval(z), density=0.3, alpha=0.1, transparency=true)
arrows!(ax, [M1.r, M2.r], [M1.M, M2.M], arrowsize=0.7, lengthscale=1 / magnitude[1], linecolor=:red)

# Plot the initial position
plot = @lift(lines!(ax, $trajectory, color=:blue))

# Plot the trajectory
fps = 120
display(fig)
for t in 1:i
    position = rP[t]
    trajectory[] = push!(trajectory[], position)
    plot
    sleep(1 / fps)
end
