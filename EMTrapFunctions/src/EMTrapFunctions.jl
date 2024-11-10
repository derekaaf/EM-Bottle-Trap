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

# Export constants, functions, and types
export μ₀, mₐ, qₐ, dt, lim, x, y, z, M, interval, Dipole, change_M!, bottleB, RK2step, Loss, Animate!, Train!

const μ₀ = 4π * 10^-7
const mₐ = 6.644657230e-27  # Example value for alpha particle mass in kg
const qₐ = 3.2043533e-19  # Example value for alpha particle charge in C
const dt = 5e-5
const lim = 15
const x = y = z = range(-lim, lim, length=100)
const M = Vec3f(0, 1, 0) # orientation of the dipole
const interval = (v -> minimum(v) .. maximum(v))

mutable struct Dipole
    mag::Float64
    M::Vec3f
    m::Vec3f
    r::Point3f
    Dipole(mag, m, r) = new(mag, mag * m, m, r)
end

# Custom setter for the m field
function change_M!(dipole1::Dipole, dipole2::Dipole, new_mag::Float64)
    dipole1.mag = dipole2.mag = new_mag
    dipole1.M = dipole1.m * new_mag
    dipole2.M = dipole2.m * new_mag
end

Zygote.@adjoint Point{3, Float32}(x::AbstractArray) = Point{3, Float32}(x), Δ -> (nothing, Δ...)
Zygote.@adjoint Vec{3, Float32}(x::AbstractArray) = Vec{3, Float32}(x), Δ -> (nothing, Δ...)

function bottleB(r, M1::Dipole, M2::Dipole) :: Point3f
    rA = r - M1.r
    rmagA = sqrt(dot(rA, rA))
    B1bottle = 3 * rA * dot(M1.M, rA) / rmagA^5
    B2bottle = -(M1.M) / rmagA^3

    rB = r - M2.r
    rmagB = sqrt(dot(rB, rB))
    B1bottle2 = 3 * rB * dot(M2.M, rB) / rmagB^5
    B2bottle2 = -(M2.M) / rmagB^3

    B_total = .+(B1bottle, B2bottle, B1bottle2, B2bottle2)

    return (μ₀ / (4 * π)) .* B_total
end

function RK2step(Bfield::Function, M1::Dipole, M2::Dipole, v::Vec3f, r::Point3f) :: Tuple{Point3f, Vec3f}
    dv1 = dt * (qₐ / mₐ) * cross(v, Bfield(r, M1, M2))
    dr1 = dt * v

    dv2 = dt * (qₐ / mₐ) * cross(v + dv1, Bfield(r + dr1, M1, M2))
    dr2 = dt * (v + dv1)

    r_new = Point3f(r + (dr1 + dr2) / 2)
    v_new = Vec3f(v + (dv1 + dv2) / 2)

    # Check for NaN values
    if any(isnan, r_new) || any(isnan, v_new)
        println(r, v, M1, M2)
        println(dv1, dv2, dr1, dr2)
        error("NaN detected in RK2step: r_new = $r_new, v_new = $v_new")
    end

    return r_new, v_new
end

# Define the Loss function
function Loss(NN::Lux.Chain, st::NamedTuple, param::NamedTuple, inputs, M1::Dipole, M2::Dipole)
    output, st = NN(inputs, param, st)
    r = Point3f(output[1])
    v = Vec3f(output[2])
    magnitude = Float64.(output[3][1])
    change_M!(M1, M2, magnitude)
    loss = 0
    i = 1
    while ((!any(x -> abs(x) >= lim, r)) && ((i * dt) < 2))
        r₁, v = RK2step(bottleB, M1, M2, v, r)
        loss += norm(r₁ .- r)/i
        r = r₁
        i += 1
    end
    loss *= norm(output[2]) + 1
    loss /= log10(magnitude + 2)
    loss *= exp(-(i * dt))
    # loss = log(loss + 2)

    if isinf(loss)
        error("Loss is $loss. Velocity is $(norm(output[2])). Magnitude is $magnitude. LogMagnitude is $(log10(magnitude + 2)*i). PLoss = $(exp(loss)).")
    end
    return magnitude, loss, st
end

function Animate!(NN::Lux.Chain, sts::NamedTuple, params::NamedTuple, inputs, M1::Dipole, M2::Dipole)
    println("Animating...")
    track = nothing
    plt = nothing

    Zygote.ignore() do
        track = Observable(Point3f[])
        plt = lines!(ax, track)
    end
    (r₀, v₀, magnitude), st = NN(inputs, params, sts)
    magnitude = magnitude[1]
    r₀ = Point3f(r₀)
    v = Vec3f(v₀)
    change_M!(M1, M2, Float64.(magnitude))
    Zygote.ignore() do
        streamplot!(ax, field, interval(x), interval(y), interval(z), density=0.3, alpha=0.1, transparency=true)
        arrows!(ax, [M1.r, M2.r], [M1.M, M2.M], arrowsize=0.7, lengthscale=1 / M1.mag)
    end
    loss = 0
    i = 1
    while ((!any(x -> abs(x) >= lim, r₀)) && ((i * dt) < 2))
        # println("Step $i: r₀ = $r₀, v = $v")
        r₁, v = RK2step(bottleB, M1, M2, v, r₀)
        Zygote.ignore() do
            track[] = push!(track[], r₁)
            label.text[] = @sprintf("Magnitude = %.5f, Time Trapped = %.5f s", magnitude, i * dt)
            plt
        end
        loss += norm(r₁ .- r₀)/i
        r₀ = r₁
        i += 1
    end
    loss *= norm(v₀) + 1
    loss /= log10(magnitude + 2)
    loss *= exp(-(i * dt))
    # loss = log(loss + 2)

    if isinf(loss)
        error("Loss is $loss. Velocity is $(norm(v₀)). Magnitude is $magnitude. LogMagnitude is $(log10(magnitude + 2)*i). PLoss = $(exp(loss)).")
    end
    println("Animation Complete. Time Trapped = ", i * dt, "s")  
    Zygote.ignore() do
        GC.gc()
        empty!(ax)
    end  
    return Float64.(magnitude), loss, st
end

 function Train!(NN, trainIt, params, sts, optimizer, losshistory, paramshistory, M1, M2)
    if (trainIt % 10 != 0) error("trainIt must be a multiple of 10") end
    magnitude = rand(10e3:10e6)
    best_loss = Inf
    patience = 0.4 * trainIt
    M1 = Dipole(magnitude, M, [0.0, 10.0, 0.0])
    M2 = Dipole(magnitude, M, [0.0, -10.0, 0.0])
    patience_counter = 0
    progress = Progress(trainIt, 1, "Training Neural Network")
    for epoch in 1:trainIt
        inputs = vcat(M1.M..., M2.M...)  # Ensure input is defined
        inputs = Float32.(inputs)  # Use CUDA.fill to create a GPU array
        Fun = epoch % (0.1 * trainIt) == 0 ? Loss : Loss
        (magnitude, loss, sts,), back = pullback(Fun, NN, sts, params, inputs, M1, M2)

        _, _, ∂params, _ = back((nothing, 1.0, nothing))
        optimizer, params = Optimisers.update(optimizer, params, ∂params)

        push!(losshistory, loss)
        change_M!(M1, M2, magnitude)

        #Early stopping
        if loss < (1.2 * best_loss)
            best_loss = loss
            patience_counter = 0
        else
            patience_counter += 1
        end
        # if patience_counter > patience
        #     println("Early stopping at epoch $epoch with best loss $best_loss")
        #     break
        # end
        if epoch % (0.1 * trainIt) == 0
            # println("Epoch: $epoch, Loss: $loss")
            # predictions = NN(inputs, params, sts)[1]
            # println("Predictions: $predictions")
            push!(paramshistory, params)
        else
            # println("Epoch: $epoch")
        end

        next!(progress)
    end
    GC.gc()
    return params
    
end

end 
