module EMTrapFunctions

using Reexport

@reexport using WGLMakie
@reexport using Lux
@reexport using Optimization
@reexport using OptimizationEvolutionary
@reexport using Optimisers
@reexport using OptimizationMetaheuristics
@reexport using Zygote
@reexport using Base.Threads
@reexport using Statistics
@reexport using LinearAlgebra
@reexport using Random
@reexport using StaticArrays
@reexport using Bonito
@reexport using ComponentArrays
@reexport using Plots
@reexport using ProgressMeter
@reexport using Printf
@reexport using DataStructures
@reexport using Colors
@reexport using Profile
@reexport using ProfileView

# Export constants, functions, and types
export μ₀, mₐ, qₐ, dt, lim, x, y, z, M, interval, simTime, Bmax, stepspersec
export Dipole, NeuralNetwork
export change_M!, bottleB, bottlePotential, RK2step, Loss, Animate!, Train!, test, Simulate
export col, tailcolor

const μ₀ = 4π * 10^-7 # N/A²
const mₐ = 6.644657230e-27  # Example value for alpha particle mass in kg
const qₐ = 3.2043533e-19  # Example value for alpha particle charge in C
const dt = 5e-5 # s
const Bmax = 2e8 # A/m
const lim = 15 # m
const x = y = z = range(-lim, lim, length=100) 
const M = Vec3f(0, 1, 0) # orientation of the dipole
const simTime = 5 # s
const interval = (v -> minimum(v) .. maximum(v))
const stepspersec = Int(1/dt)

const col = to_color(:grey37)
tailcolor = [RGBAf(col.r, col.g, col.b, (i/(stepspersec))^4) for i in 1:10*stepspersec]

rng = Random.Xoshiro(2534)
Random.seed!(rng)

function Base.:+(a::Point3f, b::Tuple)
    return a .+ b
end

Base.@kwdef mutable struct NeuralNetwork
    NN::Chain
    type::String
    states::NamedTuple
    params::NamedTuple
    optimizer::NamedTuple
end

function (NN::NeuralNetwork)(inputs, params)
    return NN.NN(inputs, params, NN.states)
end

function (NN::NeuralNetwork)(inputs)
    return NN.NN(inputs, NN.params, NN.states)
end

function test(NN::NeuralNetwork, inputs, states)
    return NN.NN(inputs, NN.params, states)
end

function updateOptimizer!(NN::NeuralNetwork, ∂params)
    (NN.optimizer, NN.params) = Optimisers.update(NN.optimizer, NN.params, ∂params)
end

mutable struct Dipole
    mag::Float64
    M::Vec3f
    m::Vec3f
    r::Point3f
    Dipole(m, r) = new(0, Vec3f(0, 0, 0), m, r)
end

# Custom setter for the m field
function change_M!(dipole1::Dipole, dipole2::Dipole, new_mag::Float64)
    dipole1.mag = dipole2.mag = new_mag
    dipole1.M = dipole1.m * new_mag
    dipole2.M = dipole2.m * new_mag
end

Zygote.@adjoint Point{3, Float32}(x) = Point{3, Float32}(x), Δ -> (nothing, Δ...)
Zygote.@adjoint Vec{3, Float32}(x) = Vec{3, Float32}(x), Δ -> (nothing, Δ...)

function bottleB(r::Point3f, M1::Dipole, M2::Dipole) :: Point3f
    rA = r .- M1.r
    rmagA = sqrt(dot(rA, rA))
    B1bottle = 3 * rA * dot(M1.M, rA) / rmagA^5
    B2bottle = -(M1.M) / rmagA^3

    rB = r .- M2.r
    rmagB = sqrt(dot(rB, rB))
    B1bottle2 = 3 * rB * dot(M2.M, rB) / rmagB^5
    B2bottle2 = -(M2.M) / rmagB^3

    B_total = .+(B1bottle, B2bottle, B1bottle2, B2bottle2)

    return (μ₀ / (4 * π)) .* B_total
end

function bottlePotential(r::Point3f, M1::Dipole, M2::Dipole) :: Float64
    rA = r .- M1.r
    rmagA = sqrt(dot(rA, rA))
    Potbottle1 = dot(M1.M, rA) / rmagA^3

    rB = r .- M2.r
    rmagB = sqrt(dot(rB, rB))
    Potbottle2 = dot(M2.M, rB) / rmagB^3

    Potential_total = .+(Potbottle1, Potbottle2)

    return Potential_total/(4 * π)
end

function bottleU(r::Point3f, v::Vec3f, M1::Dipole, M2::Dipole) :: Float64
    U = dot(cross(r, v), bottleB(r, M1, M2))
    return -(qₐ/(2*mₐ)) * U
end

function RK2step(Bfield::Function, M1::Dipole, M2::Dipole, v::Vec3f, r::Point3f) :: Tuple{Point3f, Vec3f}
    dv1 = dt * (qₐ / mₐ) * cross(v, Bfield(r, M1, M2))
    dr1 = dt * v

    dv2 = dt * (qₐ / mₐ) * cross(v .+ dv1, Bfield(Point3f(r .+ dr1), M1, M2))
    dr2 = dt * (v .+ dv1)

    r_new = Point3f(r .+ (dr1 .+ dr2) / 2)
    v_new = Vec3f(v .+ (dv1 .+ dv2) / 2)

    # Check for NaN values
    if any(isnan, r_new) || any(isnan, v_new)
        println(r, v, M1, M2)
        println(dv1, dv2, dr1, dr2)
        error("NaN detected in RK2step: r_new = $r_new, v_new = $v_new")
    end

    return r_new, v_new
end

function Decisions!(r::Point3f, v::Vec3f, M1::Dipole, M2::Dipole, v₀::Vector{Float32}, magnitude₀::Float64, NNₐ::NeuralNetwork, paramsₐ::NamedTuple) :: Float64
    loss = 0
    iter = 1
    position_smoothness = 0
    magnitude_smoothness = 0
    energy_efficiency = 0
    mag = magnitude₀
    while ((!any(x -> abs(x) >= lim, r)) && ((iter * dt) < simTime))
        # if (iter * dt) % 1 == 0
        input = Float32.(reshape(vcat(r..., v..., M1.M..., M2.M...), 12, 1))
        mag, NNₐ.states = NNₐ(input, paramsₐ)
        mag = mag[1]
        # end
        energy_efficiency += (mag)^2
        magnitude_smoothness += norm(mag - M1.mag)
        change_M!(M1, M2, mag)
        r₁, v = RK2step(bottleB, M1, M2, v, r)
        position_smoothness += norm((r₁ .- r) / lim)
        r = r₁
        iter += 1
    end
    energy_efficiency /= (iter * Bmax^2)
    magnitude_smoothness /= iter * Bmax
    position_smoothness /= iter
    loss = 0.5 * position_smoothness + 0.1 * magnitude_smoothness + 0.4 * energy_efficiency
    loss *= exp(-(iter * dt))

    return loss
end

function Decisions!(r::Point3f, v::Vec3f, M1::Dipole, M2::Dipole, v₀::Vector{Float32}, magnitude₀::Float64, NNₐ::Nothing, paramsₐ::Nothing) :: Float64
    loss = 0
    iter = 1
    position_smoothness = 0
    while ((!any(x -> abs(x) >= lim, r)) && ((iter * dt) < simTime))
        r₁, v = RK2step(bottleB, M1, M2, v, r)
        position_smoothness += norm((r₁ .- r) / lim)
        r = r₁
        iter += 1
    end
    loss = position_smoothness/iter
    loss *= norm(v₀) + 1
    loss += (magnitude₀ / Bmax)^2
    loss *= exp(-(iter * dt))

    return loss
end

# Define the Loss function
function Loss(M1::Dipole, M2::Dipole, NNₚ::NeuralNetwork, paramsₚ::NamedTuple, NNₐ::Union{NeuralNetwork, Nothing} = nothing, paramsₐ::Union{NamedTuple, Nothing} = nothing) :: Tuple{Float64, Float64}
    inputs = Float32.(vcat(M1.r..., M2.r..., M1.m..., M2.m...))
    output, NNₚ.states = NNₚ(inputs, paramsₚ)
    r = Point3f(output[1])
    v = Vec3f(output[2])
    magnitude = Float64.(output[3][1])
    change_M!(M1, M2, magnitude)
    
    loss = Decisions!(r, v, M1, M2, output[2], magnitude, NNₐ, paramsₐ)
    
    return magnitude, loss
end

# function Animate!(NN::Lux.Chain, sts::NamedTuple, params::NamedTuple, inputs, M1::Dipole, M2::Dipole)
#     println("Animating...")
#     track = nothing
#     plt = nothing

#     Zygote.ignore() do
#         track = Observable(Point3f[])
#         plt = lines!(ax, track)
#     end
#     (r₀, v₀, magnitude), st = NN(inputs, params, sts)
#     magnitude = magnitude[1]
#     r₀ = Point3f(r₀)
#     v = Vec3f(v₀)
#     change_M!(M1, M2, Float64.(magnitude))
#     Zygote.ignore() do
#         streamplot!(ax, field, interval(x), interval(y), interval(z), density=0.3, alpha=0.1, transparency=true)
#         arrows!(ax, [M1.r, M2.r], [M1.M, M2.M], arrowsize=0.7, lengthscale=1 / M1.mag)
#     end
#     loss = 0
#     i = 1
#     while ((!any(x -> abs(x) >= lim, r₀)) && ((i * dt) < 2))
#         r₁, v = RK2step(bottleB, M1, M2, v, r₀)
#         Zygote.ignore() do
#             track[] = push!(track[], r₁)
#             label.text[] = @sprintf("Magnitude = %.5f, Time Trapped = %.5f s", magnitude, i * dt)
#             plt
#         end
#         loss += norm(r₁ .- r₀)/i
#         r₀ = r₁
#         i += 1
#     end
#     loss *= norm(v₀) + 1
#     loss /= log10(magnitude + 2)
#     loss *= exp(-(i * dt))

#     if isinf(loss)
#         error("Loss is $loss. Velocity is $(norm(v₀)). Magnitude is $magnitude. LogMagnitude is $(log10(magnitude + 2)*i). PLoss = $(exp(loss)).")
#     end
#     println("Animation Complete. Time Trapped = ", i * dt, "s")  
#     Zygote.ignore() do
#         GC.gc()
#         empty!(ax)
#     end
#     return Float64.(magnitude), loss, st
# end

 function Train!(trainIt::Int, losshistory, fig, ax, label, NNₚ::NeuralNetwork, NNₐ::Union{NeuralNetwork, Nothing} = nothing)
    # if (trainIt % 10 != 0) error("trainIt must be a multiple of 10") end
    M1 = Dipole(M, [0.0, 10.0, 0.0]) # Dipole at (0, 10, 0)
    M2 = Dipole(M, [0.0, -10.0, 0.0]) # Dipole at (0, -10, 0)
    losses = Float64[]

    @showprogress dt=1 desc="Training Neural Network..." for epoch in 1:trainIt
        if !isnothing(NNₐ)
            (magnitude, loss,), back = pullback(Loss, M1, M2, NNₚ, NNₚ.params, NNₐ, NNₐ.params)
            _, _, _, ∂paramsₚ, _, ∂paramsₐ = back((nothing, 1.0))
            (isnothing(∂paramsₐ)) ? (@warn "∂paramsₐ is nothing at epoch $epoch") : nothing
            updateOptimizer!(NNₐ, ∂paramsₐ);
            NNₐ.states = Lux.update_state(NNₐ.states, :carry, nothing)
        else
            (magnitude, loss,), back = pullback(Loss, M1, M2, NNₚ, NNₚ.params)
            _, _, _, ∂paramsₚ = back((nothing, 1.0))
            isnothing(∂paramsₚ) ? (@warn "∂paramsₚ is nothing at epoch $epoch") : nothing
            updateOptimizer!(NNₚ, ∂paramsₚ)
        end

        push!(losshistory, loss)
        push!(losses, loss)
        change_M!(M1, M2, magnitude)

    end

    graph = lines!(ax, 1:trainIt, losses, color=:blue)
    GC.gc()
    label.text[] = "Do you want to keep training?"
    fig[2, 1] = buttongrid = Makie.GridLayout(tellwidth = false)
    yesButton = buttongrid[1, 1] = Makie.Button(fig, label = "Yes")
    noButton = buttongrid[1, 2] = Makie.Button(fig, label = "No")
    
    click = Condition()
    on(yesButton.clicks) do _
        empty!(ax)
        rm!(buttongrid, graph)
        delete!(yesButton)
        delete!(noButton)
        label.text[] = ""
        Train!(trainIt, losshistory, fig, ax, label, NNₚ, NNₐ)
        notify(click)
    end
    on(noButton.clicks) do _
        empty!(ax)
        rm!(buttongrid, graph)
        delete!(yesButton)
        delete!(noButton)
        label.text[] = ""
        notify(click)
    end

    wait(click)
    GC.gc()
end

function Simulate(animTime::Float64, NNₚ::NeuralNetwork, NNₐ::NeuralNetwork) :: Tuple{Vector{Point3f}, Vector{Float64}, Vector{Float64}}
    r = Point3f[] 
    magnitudes = Float64[]
    U = Float64[]

    M1 = Dipole(M, [0.0, 10.0, 0.0]) # Dipole at (0, 10, 0)
    M2 = Dipole(M, [0.0, -10.0, 0.0]) # Dipole at (0, -10, 0)
    initialValues = Float32.(vcat(M1.r..., M2.r..., M1.m..., M2.m...))
    statesₚ = Lux.testmode(NNₚ.states)
    (r₀, v, magnitude) = test(NNₚ, initialValues, statesₚ)[1]
    change_M!(M1, M2, magnitude[1])
    push!(magnitudes, magnitude[1])
    push!(r, r₀)
    u = bottleU(Point3f(r₀), Vec3f(v), M1, M2)
    push!(U, u)
    v = Vec3f(v)
    i = 1 
    progress = ProgressUnknown(desc="Calculating trajectories...", spinner = true)
    statesₐ = Lux.testmode(NNₐ.states)
    while ((!any(x -> abs(x) >= lim, r[i])) && ((i * dt) < animTime))
        input = Float32.(reshape(vcat(r[i]..., v..., M1.M..., M2.M...), 12, 1))
        mag, statesₐ = test(NNₐ, input, statesₐ)
        push!(magnitudes, mag[1])
        change_M!(M1, M2, mag[1])
        r₁, v = RK2step(bottleB, M1, M2, v, r[i])
        push!(r, r₁)
        u = bottleU(r₁, v, M1, M2)
        push!(U, u)
        i += 1
        next!(progress, spinner = "⌜⌝⌟⌞", showvalues = [("Time Trapped in seconds:", i * dt)])
    end
    println("Time Trapped = ", i * dt, "s")
    GC.gc()
    return (r, magnitudes, U)
end

function rm!(args...)
    args = nothing
    GC.gc()
end

end 
