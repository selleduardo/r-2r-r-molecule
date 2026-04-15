using DelimitedFiles
using Dates  # For detailed time tracking
using OffsetArrays  
using DataFrames
using CSV


# EXPLANATION OF INDEXING AND NOTATION:
# a_n1 -> Red Pump (mode u = -1)
# a0 -> DOPO (mode u = 0)
# a1 -> Blue pump (mode u = 1)

# Physical parameters
s = 1.4  # pump (symmetric in power and detuning)
d2 = 0
Δ_n1 = 3.2

# Integration parameters
noise = 1e-8
error = 1e-6 
step = 0.1
Max_steps =10000 
T_max = Max_steps * step
T_min = 800*step
N_iterations = 10000

function coupled_equations(u, p)
    Δ_n1, Δ1, s_n1, s1 = p
    Δ0 = ((Δ_n1 + Δ1) / 2) - d2

    a_n1, a_0, a_1 = u
    du = OffsetArray(zeros(ComplexF64, 3), -1:1)
    
    # Don't try to check these equations. I determined the four-wave-mixing term with another code in this folder, named "FWM_term.ipynb"
    du[-1] = s_n1  - (1 + im*Δ_n1)*a_n1   +0+   im*(a_n1*conj(a_n1)*a_n1 + a_n1*conj(a_0)*a_0 + a_n1*conj(a_1)*a_1 + a_0*conj(a_0)*a_n1 + a_0*conj(a_1)*a_0 + a_1*conj(a_1)*a_n1)

    du[0] =           - (1 + im*Δ0)*a_0   +0+   im*(a_n1*conj(a_n1)*a_0 + a_n1*conj(a_0)*a_1 + a_0*conj(a_n1)*a_n1 + a_0*conj(a_0)*a_0 + a_0*conj(a_1)*a_1 + a_1*conj(a_0)*a_n1 + a_1*conj(a_1)*a_0)

    du[1] = s1        - (1 + im*Δ1)*a_1   +0+   im*(a_n1*conj(a_n1)*a_1 + a_0*conj(a_n1)*a_0 + a_0*conj(a_0)*a_1 + a_1*conj(a_n1)*a_n1 + a_1*conj(a_0)*a_0 + a_1*conj(a_1)*a_1 )

    return du
end

function rk4_step(f, u, p, h)
    k0 = f(u, p)
    k1 = f(u .+ 0.5 .* h .* k0, p)
    k2 = f(u .+ 0.5 .* h .* k1, p)
    k3 = f(u .+ h .* k2, p)
    return u .+ (h / 6.0) .* (k0 .+ 2 .* k1 .+ 2 .* k2 .+ k3)
end

function solve_rk4(f, u0, p, T_max, step)
    u = copy(u0)
    time = 0.0
    while time < T_max
        u_old = copy(u)
        u = rk4_step(f, u, p, step)
        difference = sum(abs.(u .- u_old))
        if difference < error && time > T_min
            break
        else
            time += step
        end
    end
    return u
end

# Function to generate complex Gaussian noise vector
function complex_noise(n)
    return randn(n) .+ im .* randn(n)
end

# Define parameter ranges
Δ_p1_values = -3:0.05:17 # THIS IS THE CORRECT WAY (SOFT EXCITATION), DETUNING SMALL -> BIG
results = DataFrame(Iteration=Float64[], Δ=Float64[], a_n1=ComplexF64[], a0=ComplexF64[], a1=ComplexF64[])


# Iterate over Δ and keep track of time:
start_time = now()
println("Solving...")

for iter in 1:N_iterations
    u0 = OffsetArray(zeros(ComplexF64, 3), -1:1) 
    for Δ_p1 in Δ_p1_values
        p = (Δ_n1, Δ_p1, s, s)
        noise_vec = OffsetArray(complex_noise(3), -1:1)
        u0 .= u0 .+ noise * noise_vec
        u = solve_rk4(coupled_equations, u0, p, T_max, step)
        push!(results, [iter, Δ_p1, u[-1], u[0], u[1]])
        u0 = copy(u)
    end
end

end_time = now()
elapsed_time = end_time - start_time

# # Save to CSV without header
folder_path = "/home/eduardo/Documents/Ciencia/UNICAMP/[Proj]onChipOPO/Calculations_new/r-2r-r-molecule/notebooks/testjl/"
filename = folder_path * "depleted_3_modes_d2=0_s=1.4_dr=3.2_lots.csv"
CSV.write(filename, results)

println("Data saved, running time =$(elapsed_time)")