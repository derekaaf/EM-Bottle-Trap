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
@reexport using Serialization

# Export constants, functions, and types
export μ₀, mₐ, qₐ, dt, lim, x, y, z, M, interval, simTime, vmax, Bmax, Bmin, Bscale, stepspersec
export Dipole, NeuralNetwork, LayerL2Norm
export change_m!, bottleB, bottlePotential, RK2step, Loss, Animate!, Train!, test, Simulate
export col, tailcolor

const μ₀ = 4π * 10^-7 # N/A²
const mₐ = 6.644657230e-27  # Example value for alpha particle mass in kg
const qₐ = 3.2043533e-19  # Example value for alpha particle charge in C
const dt = 5e-5 # s
const vmax = 1e6 # m/s, maximum velocity of the particle
const Bmax = 2e8 # A/m
const Bmin = 1e3 # A/m
const Bscale = 2e8 # A/m
const lim = 15 # m
const x = y = z = range(-lim, lim, length=100)  # m
const D = Vec3f(0, 1, 0) # orientation of the dipole
const simTime = 5 # s
const interval = (v -> minimum(v) .. maximum(v))
const stepspersec = Int(1/dt) # 1/s

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

struct LayerL2Norm <: Lux.AbstractLuxLayer
    bound::Float64
end

Lux.initialstates(::AbstractRNG, layer::LayerL2Norm) = NamedTuple()

function (l::LayerL2Norm)(x, ps, st)
    norD2 = sqrt(sum(abs2, x))
    scale = ifelse(norD2 > l.bound, l.bound / norD2, 1f0)
    return x * scale, st
end

mutable struct Dipole
    mag::Float64
    m::Vec3f
    m̂::Vec3f
    r::Point3f
    Dipole(m̂, r) = new(0, Vec3f(0, 0, 0), m̂, r)
    Dipole(mag, m, m̂, r) = new(mag, m, m̂, r)
end

function copy(dipole::Dipole)
    new_dipole = Dipole(dipole.mag, dipole.m, dipole.m̂, dipole.r)
    return new_dipole
end

# Custom setter for the m field
function change_m!(dipole::Dipole, new_mag::Float64)
    dipole.mag = new_mag
    dipole.m = dipole.m̂ * new_mag
end

function change_r!(dipole::Dipole, new_r::Point3f)
    dipole.r = new_r
end

Zygote.@adjoint Point{3, Float32}(x) = Point{3, Float32}(x), Δ -> (nothing, Δ...)
Zygote.@adjoint Vec{3, Float32}(x) = Vec{3, Float32}(x), Δ -> (nothing, Δ...)

function bottleB(r::Point3f, D1::Dipole, D2::Dipole) :: Point3f
    rA = r .- D1.r
    rmagA = sqrt(dot(rA, rA))
    B1bottle = 3 * rA * dot(D1.m, rA) / rmagA^5
    B2bottle = -(D1.m) / rmagA^3

    rB = r .- D2.r
    rmagB = sqrt(dot(rB, rB))
    B1bottle2 = 3 * rB * dot(D2.m, rB) / rmagB^5
    B2bottle2 = -(D2.m) / rmagB^3

    B_total = .+(B1bottle, B2bottle, B1bottle2, B2bottle2)

    return (μ₀ / (4 * π)) .* B_total
end

function bottlePotential(r::Point3f, D1::Dipole, D2::Dipole) :: Float64
    rA = r .- D1.r
    rmagA = sqrt(dot(rA, rA))
    Potbottle1 = dot(D1.m, rA) / rmagA^3

    rB = r .- D2.r
    rmagB = sqrt(dot(rB, rB))
    Potbottle2 = dot(D2.m, rB) / rmagB^3

    Potential_total = .+(Potbottle1, Potbottle2)

    return Potential_total/(4 * π)
end

function bottleU(r::Point3f, v::Vec3f, D1::Dipole, D2::Dipole) :: Float64
    U = dot(cross(r, v), bottleB(r, D1, D2))
    return -(qₐ/(2*mₐ)) * U
end

function RK2step(Bfield::Function, D1::Dipole, D2::Dipole, v::Vec3f, r::Point3f) :: Tuple{Point3f, Vec3f}
    dv1 = dt * (qₐ / mₐ) * cross(v, Bfield(r, D1, D2))
    dr1 = dt * v

    dv2 = dt * (qₐ / mₐ) * cross(v .+ dv1, Bfield(Point3f(r .+ dr1), D1, D2))
    dr2 = dt * (v .+ dv1)

    r_new = Point3f(r .+ (dr1 .+ dr2) / 2)
    v_new = Vec3f(v .+ (dv1 .+ dv2) / 2)

    # Check for NaN values
    if any(isnan, r_new) || any(isnan, v_new)
        println(r, v, D1, D2)
        println(dv1, dv2, dr1, dr2)
        error("NaN detected in RK2step: r_new = $r_new, v_new = $v_new")
    end

    return r_new, v_new
end

function Decisions!(r::Point3f, v::Vec3f, D1::Dipole, D2::Dipole, v₀::Vector{Float64}, magnitude₀::Float64, NNₐ::NeuralNetwork, paramsₐ::NamedTuple) :: Float64
    loss = 0
    iter = 1
    position_smoothness = 0
    magnitude_smoothness = 0
    energy_efficiency = 0
    prev_r = r
    while ((!any(x -> abs(x) >= lim, r)) && ((iter * dt) < simTime))
        # if (iter * dt) % 1 == 0
        dedge = lim - norm(r)
        input = Float32.(reshape(vcat(dedge, prev_r..., r..., v..., D1.m..., D2.m...), 16, 1))
        mag, NNₐ.states = NNₐ(input, paramsₐ)
        # end
        energy_efficiency += (sum(mag))^2
        magnitude_smoothness += norm(mag[1] - D1.mag) + norm(mag[2] - D2.mag)
        change_m!(D1, mag[1])
        change_m!(D2, mag[2])
        r₁, v = RK2step(bottleB, D1, D2, v, r)
        position_smoothness += norm((r₁ .- r) / lim)
        r = r₁
        prev_r = r
        iter += 1
    end
    energy_efficiency /= (iter * (2*Bmax)^2)
    magnitude_smoothness /= iter * 2 * Bmax
    position_smoothness /= iter
    loss = 0.5 * position_smoothness + 0.1 * magnitude_smoothness + 0.4 * energy_efficiency
    loss /= iter * dt
    loss += abs(D1.mag - D2.mag) / (D1.mag + D2.mag)

    return loss
end

function Decisions!(r::Point3f, v::Vec3f, D1::Dipole, D2::Dipole, v₀::Vector{Float64}, magnitude₀::Float64, NNₐ::Nothing, paramsₐ::Nothing) :: Float64
    loss = 0
    iter = 1
    position_smoothness = 0
    while ((!any(x -> abs(x) >= lim, r)) && ((iter * dt) < simTime))
        r₁, v = RK2step(bottleB, D1, D2, v, r)
        position_smoothness += norm((r₁ .- r) / lim)
        r = r₁
        iter += 1
    end
    loss = position_smoothness/iter
    loss *= norm(v₀) + 1
    loss += (magnitude₀ / (2*Bmax))^2
    loss /= iter * dt
    loss += abs(D1.mag - D2.mag) / (D1.mag + D2.mag)

    return loss
end

# Define the Loss function
function Loss(D1::Dipole, D2::Dipole, NNₚ::NeuralNetwork, paramsₚ::NamedTuple, NNₐ::Union{NeuralNetwork, Nothing} = nothing, paramsₐ::Union{NamedTuple, Nothing} = nothing) :: Tuple{Float64, Float64, Float64}
    inputs = Float32.(vcat(D1.r..., D2.r..., D1.m̂..., D2.m̂...))
    output, NNₚ.states = NNₚ(inputs, paramsₚ)
    r = Point3f(output[1])
    v = Vec3f(output[2] * output[3][1])
    sep = Float64.(output[4][1])
    rdipole1 = Point3f(0, sep/2, 0)
    rdipole2 = Point3f(0, -sep/2, 0)
    change_r!(D1, rdipole1)
    change_r!(D2, rdipole2)
    magnitude1 = Float64.(output[5][1])
    magnitude2 = Float64.(output[5][2])
    change_m!(D1, magnitude1)
    change_m!(D2, magnitude2)
    
    loss = Decisions!(r, v, D1, D2, Float64.(output[2] * output[3][1]), magnitude1 + magnitude2, NNₐ, paramsₐ)
    
    return magnitude1, magnitude2, loss
end

 function Train!(trainIt::Int, losshistory, fig, ax, label, NNₚ::NeuralNetwork, NNₐ::Union{NeuralNetwork, Nothing} = nothing)
    # if (trainIt % 10 != 0) error("trainIt must be a multiple of 10") end
    D1 = Dipole(D, [0.0, 10.0, 0.0]) # Dipole at (0, 10, 0)
    D2 = Dipole(D, [0.0, -10.0, 0.0]) # Dipole at (0, -10, 0)
    losses = Float64[]

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

        push!(losshistory, loss)
        push!(losses, loss)
        change_m!(D1, magnitude1)
        change_m!(D2, magnitude2)

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

function Simulate(animTime::Float64, NNₚ::NeuralNetwork, NNₐ::Union{NeuralNetwork, Nothing} = nothing) :: Tuple{Vector{Point3f}, Dipole, Dipole, Vector{Vector{Float64}}, Vector{Float64}}
    r = Point3f[] 
    magnitudes1 = Float64[]
    magnitudes2 = Float64[]
    U = Float64[]

    D1 = Dipole(D, [0.0, 10.0, 0.0]) # Dipole at (0, 10, 0)
    D2 = Dipole(D, [0.0, -10.0, 0.0]) # Dipole at (0, -10, 0)
    initialValues = Float32.(vcat(D1.r..., D2.r..., D1.m̂..., D2.m̂...))
    statesₚ = Lux.testmode(NNₚ.states)
    (r₀, v, vmag, Dsep, mag) = test(NNₚ, initialValues, statesₚ)[1]
    Dsep = Float64.(Dsep[1])
    vmag = Float64.(vmag[1])
    v = v * vmag
    change_r!(D1, Point3f(0, Dsep/2, 0))
    change_r!(D2, Point3f(0, -Dsep/2, 0))
    change_m!(D1, mag[1])
    change_m!(D2, mag[2])
    D1₀ = copy(D1)
    D2₀ = copy(D2)
    push!(magnitudes1, mag[1])
    push!(magnitudes2, mag[2])
    push!(r, r₀)
    u = bottleU(Point3f(r₀), Vec3f(v), D1, D2)
    push!(U, u)
    v = Vec3f(v)
    prev_r = r₀
    i = 1 
    progress = ProgressUnknown(desc="Calculating trajectories...", spinner = true)
    if !isnothing(NNₐ) 
        statesₐ = Lux.testmode(NNₐ.states)
    end
    while ((!any(x -> abs(x) >= lim, r[i])) && ((i * dt) < animTime))
        if !isnothing(NNₐ)
            if (i % stepspersec == 0)
                dedge = lim - norm(r[i])
                input = Float32.(reshape(vcat(dedge, prev_r..., r[i]..., v..., D1.m..., D2.m...), 16, 1))
                mag, statesₐ = test(NNₐ, input, statesₐ)
            end
            push!(magnitudes1, mag[1])
            push!(magnitudes2, mag[2])
            change_m!(D1, mag[1])
            change_m!(D2, mag[2])
        end
        r₁, v = RK2step(bottleB, D1, D2, v, r[i])
        push!(r, r₁)
        u = bottleU(r₁, v, D1, D2)
        push!(U, u)
        prev_r = r[i]
        i += 1
        next!(progress, spinner = "⌜⌝⌟⌞", showvalues = [("Time Trapped in seconds:", i * dt)])
    end
    println("Time Trapped = ", i * dt, "s")
    GC.gc()
    return (r, D1₀, D2₀, [magnitudes1, magnitudes2], U)
end

function rm!(args...)
    args = nothing
    GC.gc()
end

end 
