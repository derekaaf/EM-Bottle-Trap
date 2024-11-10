using Pkg
Pkg.activate("EMTrapFunctions")
using EMTrapFunctions

Page(listen_url = "localhost", listen_port = 9876)
WGLMakie.activate!()
Makie.inline!(true)

fig = Figure(size=(1000, 1000))
ax = LScene(fig[1, 1], show_axis=false, width = 800)
label = Label(fig[0, 1], justification = :center, fontsize = 20)

rng = Random.Xoshiro(237897)

# Inputs: M, M. Output: x, y, z, vx, vx, vy, vz, magnitude
NN = Chain(
    Dense(6 => 18, swish),
    Dense(18 => 31, relu),
    Dense(31 => 45, tanh),
    Dense(45 => 22),
    Parallel(nothing, Dense(22 => 6, sigmoid), Dense(22 => 6, softsign), Dense(22 => 13, x -> 10 * sigmoid(x))),
    Parallel(nothing, Dense(6 => 3), Dense(6 => 6), Dense(13 => 8, x -> 10e5 * (sech(x))^2)),
    Parallel(nothing, Dense(3 => 3, x -> lim/3 * tanh(x)), Dense(6 => 3, x -> 2 * lim * tanh(x)), Dense(8 => 1, abs))
)
params, sts = Lux.setup(rng, NN)
optimizer = Optimisers.setup(Adam(1e-5), params)

# Initial conditions
magnitude = rand(10e2:10e8)
M1 = Dipole(magnitude, M, [0.0, 10.0, 0.0])
M2 = Dipole(magnitude, M, [0.0, -10.0, 0.0])
input = vcat(M1.M..., M2.M...)
input = Float32.(input)
initpred = NN(input, params, sts)[1]

losshistory = Float64[]
paramshistory = typeof(params)[]
TrainIT = 10
params = Train!(NN, TrainIT, params, sts, optimizer, losshistory, paramshistory, M1, M2)
empty!(ax)


r = [Point3f[] for _ in 1:length(paramshistory)]
for (R, param) in zip(r, paramshistory)
    (r₀, v, magnitude) = NN(input, param, sts)[1]
    change_M!(M1, M2, magnitude[1])
    push!(R, r₀)
    v = Vec3f(v)
    i = 1
    while ((!any(x -> abs(x) >= lim, R[i])) && ((i * dt) < 60))
        r₁, v = RK2step(bottleB, M1, M2, v, R[i])
        push!(R, r₁)
        i += 1
    end
    println("Time Trapped = ", i * dt)
end
GC.gc()

field = (r -> bottleB(r, M1, M2))
streamplot!(ax, field, interval(x), interval(y), interval(z), density=0.3, alpha=0.1, transparency=true)
arrows!(ax, [M1.r, M2.r], [M1.M, M2.M], arrowsize=0.7, lengthscale= 1 / M1.mag)
cam = Makie.Camera3D(ax.scene, projectiontype = Makie.Perspective)

tracks = [Observable(Point3f[]) for _ in 1:length(r)]
labels = [@sprintf("%.0f", (i/10) * TrainIT) for i in 1:length(r)]
lines = []

plt = for (track, i, label) in zip(tracks, 1:length(r), labels)
    push!(lines, lines!(ax, track, transparency=true, linestyle=:dash, colormap=:Paired_8))
end
legend = Legend(fig[1, 2], lines, labels, "Training Iterations", fontsize = 10)

stop = Threads.Atomic{Bool}(true)
i = Threads.Atomic{Int}(1)
intvl = 50
@sync while true
    stop[] = true
    @threads for j in 1:length(paramshistory)
        if (i[]) <= length(r[j])
            tracks[j][] = push!(tracks[j][], r[j][i[]])
            # tracks[j][] = append!(tracks[j][], r[j][((i-1)*fps)+1:i*fps])
            stop[] = false
        end
    end
    label.text[] = @sprintf("Time Trapped = %.2f s", i[] * dt) 
    # sleep(dt) # 1s * speed
    Threads.atomic_add!(i, intvl)
    if stop[] break end
end