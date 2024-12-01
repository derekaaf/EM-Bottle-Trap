using Pkg
Pkg.activate("EMTrapFunctions")
using EMTrapFunctions

Page(listen_url = "localhost", listen_port = 2341)
WGLMakie.activate!()
Makie.inline!(true)

rng = Random.Xoshiro(262397)
Random.seed!(rng)

fig = Figure(size = (1920, 1080))
label = Label(fig[0, 1], justification = :center, fontsize = 40)
ax = Makie.Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Loss", aspect = 1, height = 600, width = 600)

# Inputs: M, M. Output: x, y, z, vx, vx, vy, vz, dipole 1 position, dipole 2 position
NN = Chain(
    Dense(12 => 18, swish),
    Dense(18 => 31, relu),
    Dense(31 => 45, tanh),
    Dense(45 => 22),
    Parallel(nothing, Dense(22 => 6, sigmoid), Dense(22 => 6, softsign), Dense(22 => 13, x -> 10 * sigmoid(x))),
    Parallel(nothing, Dense(6 => 3), Dense(6 => 6), Dense(13 => 8, x -> 10e8 * (sech(x))^2)),
    Parallel(nothing, Dense(3 => 3, x -> lim/3 * tanh(x)), Dense(6 => 3, x -> 2 * lim * tanh(x)), Dense(8 => 1, abs))
)
params, sts = Lux.setup(rng, NN)
rules = OptimiserChain(ClipGrad(10), Adam(5e-4))
optimizer = Optimisers.setup(rules, params)
NNₚ = NeuralNetwork(NN, "Passive", sts, params, optimizer)

# Inputs: x, y, z, vx, vy, vz, dipole moment 1, dipole moment 2. Output: magnitude
NN = Chain(
    GroupNorm(12, 4, affine = true),
    Dense(12 => 12, swish),
    StatefulRecurrentCell(LSTMCell(12 => 18, train_memory = true, train_state = true)),
    Dense(18 => 31, relu),
    Dropout(0.1),
    Dense(31 => 45),
    Dropout(0.2),
    Dense(45 => 22, x -> 10 * sigmoid(x)),
    Dropout(0.2),
    Dense(22 => 1, x -> 1e2 * (sech(x))^2),
    Dense(1 => 1, x -> clamp.(abs.(x * 1e6), 1e3, Bmax))
)
params, sts = Lux.setup(rng, NN)
rules = OptimiserChain(ClipGrad(10), RMSProp(4e-3, 0.3))
optimizer = Optimisers.setup(rules, params)
NNₐ = NeuralNetwork(NN, "Active", sts, params, optimizer)


losshistory = [Float64[] for _ in 1:2]
# paramshistory = [typeof(NNₚ.params)[], typeof(NNₐ.params)[]]
TrainIT = 200
Train!(TrainIT, losshistory[1], fig, ax, label, NNₚ)
Train!(TrainIT, losshistory[2], fig, ax, label, NNₚ, NNₐ)


delete!(ax)
delete!(label)
# fig = Figure(resolution = (1080, 800))
ax = LScene(fig[1, 1], show_axis=false, height = 800, width = 1000)
ax2D = Makie.Axis(fig[1, 2][1, 1], aspect = 1, height = 300, width = 300)
label = Label(fig[0, 1:2], justification = :center, fontsize = 40)

animTime = 60.0*60
r, magnitudes, U = Simulate(animTime, NNₚ, NNₐ)
M1 = Dipole(M, [0.0, 10.0, 0.0]) # Dipole at (0, 10, 0)
M2 = Dipole(M, [0.0, -10.0, 0.0]) # Dipole at (0, -10, 0)
change_M!(M1, M2, magnitudes[1])

field = (r -> bottleB(Point3f(r), M1, M2))
streamplot!(ax, field, x, y, z, density=0.2, alpha=0.1, transparency=true)
arrows!(ax, [M1.r, M2.r], [M1.M, M2.M], arrowsize=0.7, lengthscale= 1 / M1.mag)

cam = Makie.Camera3D(ax.scene, projectiontype=Makie.Perspective)
particle3D = Observable{Point3f}(r[1])
particle2D = @lift(Point2f($particle3D[1], $particle3D[2]))

rₖ = @lift(Float64($particle3D[3]))
field = @lift((r) -> Point2(bottleB(Point3f(r..., $rₖ), M1, M2)[1:2]))
streamplot!(ax2D, field, x, y; density=1, alpha=0.5, transparency=true, depth_shift = 0.0)
tail = CircularBuffer{Point3f}(10*stepspersec)
fill!(tail, r[1])
tail = Observable(tail)
tail2D = @lift(Point2f.($tail))

Makie.scatter!(ax, particle3D, markersize=10, color=:black)
Makie.scatter!(ax2D, particle2D, markersize=10, color=:black)
lines!(ax, @lift(collect($tail)); color=tailcolor, linewidth=5, transparency=true)
lines!(ax2D, tail2D; color=tailcolor, linewidth=5, transparency=true)

Base.size(screen::WGLMakie.Screen) = size(screen.scene)
playback_duration = 10.0  # 60 seconds
framerate = 30         # 30 FPS
total_frames = playback_duration * framerate

intval = (round(Int, length(r) / total_frames)) != 0 ? (round(Int, length(r) / total_frames)) : 1
# Downsample indices
frame_indices = 1:intval:length(r)

potentials = Array{Float64}(undef, length(x), length(y), length(frame_indices))
@showprogress dt=1 desc="Calculating Potentials..." @threads for k in eachindex(frame_indices)
    for i in eachindex(x)
        for j in eachindex(y)
            potentials[i, j, k] = bottlePotential(Point3f(x[i], y[j], r[frame_indices[k]][3]), M1, M2)
        end
    end
end

Potential = Observable{Matrix{Float64}}(potentials[:, :, 1])
Pot = Makie.contourf!(ax2D, x, y, Potential; transparency=true, colormap=(:RdBu, 0.5), depth_shift = 0.0)
colbar = Colorbar(fig[1, 2][1, 2], Pot, label="Potential", labelpadding=20, width=20, ticklabelsize=10)

# Precompute data for frames
label_text = [@sprintf("Time Trapped = %.2f min. Magnitude of field = %.2f. \nPotential energy = %.2f", (i * dt) / 60, magnitudes[i], U[i]) for i in frame_indices]
@profile record(fig, "EMTrap1.mp4", frame_indices; framerate=framerate) do idx
    if idx == 1
        zoom!(ax.scene, 0.6)
    end
    i = Int(ceil(idx/intval))
    # Update track
    previdx = max(idx - intval, 1)
    particle3D[] = r[idx]

    tail[] = append!(tail[], r[previdx:idx])
    Potential[] = potentials[:, :, i]

    # Update label
    label.text[] = label_text[i]
end

delete!(ax)
delete!(ax2D)
delete!(label)
delete!(colbar)