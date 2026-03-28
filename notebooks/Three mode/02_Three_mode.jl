using DifferentialEquations
using Plots
using Polynomials


function cmt3!(du, u, p, tau)
    Delta_p, Delta_i, Delta_s, s = p

    # Reconstruct complex amplitudes
    ap  = u[1] + 1im * u[2]
    ai  = u[3] + 1im * u[4]
    as = u[5] + 1im * u[6]


    dap = ap*(-1 - 1im*Delta_p) - 1im*(2*ai*ap*conj(ai) + abs2(ap)*conj(ap) + 2*ai*as*conj(ap) + 2*ap*as*conj(as)) + s[0]

    dai = ai*(-1 - 1im*Delta_i) - 1im*(abs2(ai)*conj(ai) + 2*ai*ap*conj(ap) + abs2(ap)*conj(as) + 2*ai*as*conj(as))

    das = as*(-1 - 1im*Delta_s) - 1im*(abs2(ap)*conj(ai) + 2*ai*as*conj(ai) + 2*ap*as*conj(ap) + abs2(as)*conj(as))


    du[1] = real(dap)
    du[2] = imag(dap)
    du[3] = real(dai)
    du[4] = imag(dai)
    du[5] = real(das)
    du[6] = imag(das)
end