# EM Trap: Step-by-Step Tutorial

EM Trap is a Julia-based simulation framework that models interactions between magnetic dipoles using both static (passive) and dynamic (active) neural network architectures implemented in Lux.jl. The repository demonstrates environment setup, model definitions, training routines, and real-time 3D/2D visualizations of particle trajectories and potential fields. Through modular code and reproducible practices, EM Trap enables researchers to experiment with network designs, hyperparameters, and rendering workflows for magnetic field analysis and animation.

For questions or suggestions, open an issue on GitHub.
---

## Activate Project and Load Package

```julia
using Pkg
Pkg.activate("EMTrapFunctions")
using EMTrapFunctions
```

`EMTrapFunctions` contains all the necessary functions and variables to setup and run the simulationLoading the custom package gives access to structs like `Dipole` and `LayerL2Norm` and functions like `Train!`, `Simulate`.

---

## Seed the RNG for Reproducibility

```julia
rng = Random.Xoshiro(262397)
Random.seed!(rng)
```

---

## Prepare the Figure and Axes

```julia
fig = Figure(size = (1920, 1080))
label = Label(fig[0, 1], justification = :center, fontsize = 40)

ax = Axis(
    fig[1, 1],
    xlabel = "Iterations",
    ylabel = "Loss",
    aspect = 1,
    height = 600,
    width = 600
)
```
This `Axis` will be used to display a Loss vs Iterations plot to track the neural network training process.
The `Figure` will be reused for the simulation display later on.

---

## Define Neural Networks

### Passive Network

```julia
NN = Chain(
    Dense(12 => 16, swish),             # Input: 12 features [positions (3*2), moments (3*2)]
    Dense(16 => 32, swish),             # Hidden embedding
    Dense(32 => 16, relu),              # Hidden layer
    Dense(16 => 10),
    Parallel(
        nothing,
        Dense(10 => 6, tanh),           
        Dense(10 => 6, tanh),           
        Dense(10 => 6),                 
        Dense(10 => 6, sigmoid),        
        Dense(10 => 10)                 
    ),
    Parallel(
        nothing,
        Dense(6 => 3),                  # Vector [x, y, z]
        Dense(6 => 3, softsign),        # Vector [vx, vy, vz] 
        Dense(6 => 1, x -> 20 * softsign(x)), # Scaled velocity magnitude output
        Dense(6 => 6),
        Dense(10 => 8)
    ),
    Parallel(
        nothing,
        LayerL2Norm(lim),               # Restrict spatial range to defined limits
        LayerL2Norm(1.0),               # Restrict the scaled velocity to a max magnitude of 1
        Dense(1 => 1, x -> vmax * sigmoid(x)),      # Max velocity magnitude
        Dense(6 => 1, x -> 2 * abs(lim * tanh(x))),     # New dipole separation, restricted to a positive length inside [-lim, lim]
        Dense(8 => 2, x -> Bmin .+ (Bmax - Bmin) * sigmoid.(x)) # Dipole magnitudes [D1, D2]
    )
)
params, sts = Lux.setup(rng, NN)
rules = OptimiserChain(ClipGrad(10), Adam(5e-3))
optimizer = Optimisers.setup(rules, params)
NNₚ = NeuralNetwork(NN, "Passive", sts, params, optimizer)
```
* Inputs:  12 features combining dipole positions $$(x_1,y_1,z_1,x_2,y_2,z_2)$$ and dipole moments unit vector $$(\hat{m1}_x,\hat{m1}_y,\hat{m1}_z,\hat{m2}_x,\hat{m2}_y,\hat{m2}_z)$$ into a float 32 1D array.
* Layer breakdown
    1. `Dense(12 => 16, swish)`: Initial embedding of raw dipole data.
    2. `Dense(16 => 32, swish)`: Expanded feature representation.
    3. `Dense(32 => 16, relu)`: Nonlinear compression.
    4. `Dense(16 => 10)`: Bottleneck to distill key features.
    5. First `Parallel(...)`: Multi-head streams for directions, magnitudes, separations, and combined signals.
    6. Second `Parallel(...)`: Maps these streams to intermediate outputs (vectors and magnitudes).
    7. Third `Parallel(...)`: Applies normalization, scaling, and final mapping to physical dipole and particle initial properties.
* Outputs:
  - Initial particle position $$[x, y, z]$$
  - Initial velocity direction scaled $$[ v_x, v_y, v_z ]$$ 
  - Initial velocity max magnitude $${v_0}$$
  - Initial dipole separation $$d_0$$
  - Dipoles magnitudes $$(|m_1|, |m_2|)$$


### Active Network 

```julia
NN = Chain(
    Dense(16 => 32, swish),                                        # 1. Embed input features
    StatefulRecurrentCell(
        LSTMCell(32 => 64, train_memory = true, train_state = true)
    ),                                                             # 2. LSTM for long-term temporal context
    Dropout(0.1),                                                  # 3. Regularize recurrent output
    LayerNorm(64),                                                 # 4. Stabilize activations
    StatefulRecurrentCell(
        RNNCell(64 => 32, train_state = true)
    ),                                                             # 5. RNN for short-term temporal context
    Dense(32 => 16, swish),                                        # 6. Reduce to dense features
    Dense(16 => 6),                                                # 7. Intermediate mapping
    Dense(6 => 2, x -> Bmin .+ (Bmax - Bmin) * sigmoid.(x))         # 8. Final scaling to physical range
)
params, sts = Lux.setup(rng, NN)
rules = OptimiserChain(ClipGrad(10), Adam(4e-2))
optimizer = Optimisers.setup(rules, params)
NNₐ = NeuralNetwork(NN, "Active", sts, params, optimizer)
```
* Inputs: 16-dimensional vector of:
  - Distance from the edge of the trap $$d_{edge}$$
  - Particle position from previous (immediate) step $$[x_{i-1}, y_{i-1}, z_{i-1}]$$
  - Current particle position $$[x_i,y_i,z_i]$$
  - Current particle velocity $$[v_x,v_y,v_z]$$
  - Dipole moments $$(m_1, m_2)$$ each 3-dimensional vectors
* Layer breakdown
  1. `Dense(16 => 32, swish)`: Projects inputs into higher-dimensional feature space with smooth nonlinearity.
  2. `LSTMCell(32 => 64)`: Captures long-range dependencies in time-series data, maintaining memory and state.
  3. `Dropout(0.1)`: Randomly drops 10% of entries to prevent overfitting.
  4. `LayerNorm(64)`: Normalizes across features for stable gradients.
  5. `RNNCell(64 => 32)`: Lightweight recurrent layer for refining temporal patterns, adding short-term memory.
  6. `Dense(32 => 16, swish)`: Compresses back to a moderate-sized embedding.
  7. `Dense(16 => 6)`: Maps to an intermediate representation before final scaling.
  8. `Dense(6 => 2, custom activation)`: Outputs two values, each passed through a sigmoid and linearly scaled between Bmin and Bmax.
* Outputs:
    - New dipole 1 magnitude $$|m_1|$$
    - New dipole 2 magnitude $$|m_2|$$

---

## Training 

```julia
losshistory = [Float64[] for _ in 1:2]
TrainIT = 50
Train!(TrainIT, losshistory[1], fig, ax, label, NNₚ)
serialize("NNₚ.jls", NNₚ)
Train!(TrainIT, losshistory[2], fig, ax, label, NNₚ, NNₐ)
serialize("vars/NNₐ.jls", NNₐ)
```
We then train and save, or load the trained neural networks in `TrainIt` intervals, while `Train!` allows us to keep track of the loss trend per interval and either continue or stop training.

```julia
function Train!(trainIt::Int, losshistory, fig, ax, label, NNₚ::NeuralNetwork, NNₐ::Union{NeuralNetwork, Nothing} = nothing)

    ...

    @showprogress dt=1 desc="Training Neural Network..." for epoch in 1:trainIt
        if !isnothing(NNₐ)
            (magnitude1, magnitude2, loss,), back = pullback(Loss, D1, D2, NNₚ, NNₚ.params, NNₐ, NNₐ.params)
            _, _, _, ∂paramsₚ, _, ∂paramsₐ = back((nothing, nothing, 1.0))
            (isnothing(∂paramsₐ)) ? (@warn "∂paramsₐ is nothing at epoch $epoch") : nothing
            updateOptimizer!(NNₐ, ∂paramsₐ);
            NNₐ.states = Lux.update_state(NNₐ.states, :carry, nothing)
        else
            (magnitude1, magnitude2, loss,), back = pullback(Loss, D1, D2, NNₚ, NNₚ.params)
            _, _, _, ∂paramsₚ = back((nothing, nothing, 1.0))
            isnothing(∂paramsₚ) ? (@warn "∂paramsₚ is nothing at epoch $epoch") : nothing
            updateOptimizer!(NNₚ, ∂paramsₚ)
        end

        ...
        change_m!(D1, magnitude1) # Update dipole moment 1
        change_m!(D2, magnitude2) # Update dipole moment 2

    end

  ...

end
```

---

## Simulate Particle Trajectories

```julia
animTime = 60.0 * 10
r, D1, D2, magnitudes, U = Simulate(animTime, NNₚ, NNₐ)
```

We will train the simulation for 10 minutes.
The simulation lets the passive NN define the proper initial properties and then updates the particle's properties using a Runge-Kutta 2 (RK2) method where it lets the active NN change the dipoles' magnitudes every 1s to update the trap.

* **Outputs**:
  - `r`: array of positions over time.
  - `D1`, `D2`: dipole objects.
  - `magnitudes`: dipole strengths logged each timestep.
  - `U`: potential energy history.

---

## Visualization of Fields and Trajectories

Now, we clear the `Figure` and create a `LScene` for the 3D axis of the simulation and an `Axis` for the 2D axis of the slice of magnetic potential space.
```julia
# Clear previous plots
delete!(ax)
delete!(label)

# 3D scene
ax3D = LScene(fig[1, 1], show_axis = false, height = 800, width = 1000)

# 2D scene
ax2D = Axis(fig[1, 2], aspect = 1, height = 300, width = 300)
label = Label(fig[0, 1:2], justification = :center, fontsize = 40)

# Streamlines and arrows
field = (r -> bottleB(Point3f(r), D1, D2))
streamplot!(ax3D, field, x, y, z; density = 0.2, alpha = 0.1)
arrows!(
    ax3D,
    [D1.r, D2.r],
    [D1.m, D2.m],
    arrowsize = 0.7,
    lengthscale = 1 / max(D1.mag, D2.mag)
)

...
field = @lift((r) -> Point2(bottleB(Point3f(r..., $rₖ), D1, D2)[1:2]))
streamplot!(ax2D, field, x, y; density=1, alpha=0.5, transparency=true, depth_shift = 0.0)
```

Then, we create the `Observable` objects for the particle (2D and 3D) and its "tail", representing the past position of the particle.

```julia
# Particle
particle3D = Observable{Point3f}(r[1])
particle2D = @lift(Point2f($particle3D[1], $particle3D[2]))

rₖ = @lift(Float64($particle3D[3]))
...
tail = CircularBuffer{Point3f}(10*stepspersec)
fill!(tail, r[1])
tail = Observable(tail)
tail2D = @lift(Point2f.($tail))

Makie.scatter!(ax, particle3D, markersize=5, color=:black)
Makie.scatter!(ax2D, particle2D, markersize=5, color=:black)
lines!(ax, @lift(collect($tail)); color=tailcolor, linewidth=5, transparency=true)
lines!(ax2D, tail2D; color=tailcolor, linewidth=5, transparency=true)
```

---

## Computing Potential Fields

Calculate the magnetic potentials for the 2D axis at each time step.
```julia
potentials = Array{Float64}(undef, length(x), length(y), length(frame_indices))

@showprogress dt=1 desc="Calculating Potentials..." @threads for k in eachindex(frame_indices)
    for i in eachindex(x), j in eachindex(y)
        potentials[i, j, k] = bottlePotential(Point3f(x[i], y[j], r[frame_indices[k]][3]), D1, D2)
    end
end

Potential = Observable{Matrix{Float64}}(potentials[:, :, 1])
Pot = Makie.contourf!(ax2D, x, y, Potential; transparency=true, colormap=(:RdBu, 0.5), depth_shift = 0.0)
colbar = Colorbar(fig[1, 2][1, 2], Pot, label="Potential", labelpadding=20, width=20, ticklabelsize=10)
```
`@threads` allows for parallel computing of the potentials to speed up the process.

---

## Recording the Animation

```julia
playback_duration = 10.0  # seconds
framerate = 30            # FPS
total_frames = playback_duration * framerate
intval = (round(Int, length(r) / total_frames)) != 0 ? (round(Int, length(r) / total_frames)) : 1
frame_indices = 1:intval:length(r)

@profile record(
    fig,
    "test/EMTrap.mp4",
    frame_indices;
    framerate = framerate
) do idx
    # Update camera, particle, tail, potential, and label
end
```

* \`\`: Exports an MP4 by iterating through frames.

* **Dynamic label**: Shows elapsed time, dipole logs, and energy per frame.

* **Profiled**: `@profile` assists in performance tuning if needed.

* **Why downsample?** Controls video length and frame count to balance smoothness vs. file size.

---

# Samples

All of the following videos are available in the test folder.

## Heavily Trained NNs (> 200 iterations)
https://github.com/user-attachments/assets/44c07eed-4273-4bbf-837c-4f38ea3a8415

## NNs Trained over 100 iterations
https://github.com/user-attachments/assets/5d6ef02e-d25e-4b4f-9d8c-d39b2c8d66f1

## NNs Trained over 50 iterations
https://github.com/user-attachments/assets/4ac8a4b9-cf40-4877-8678-e6ef7182362a

