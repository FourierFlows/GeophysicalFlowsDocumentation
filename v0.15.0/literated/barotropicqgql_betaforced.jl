using GeophysicalFlows, CUDA, Random, Printf, CairoMakie

using Statistics: mean

parsevalsum = FourierFlows.parsevalsum
record = CairoMakie.record

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution = n^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 8000           # total number of time-steps
 nsubs  = 10             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.01      # bottom drag
nothing # hide

forcing_wavenumber = 14.0 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 0.001                         # energy input rate by the forcing

grid = TwoDGrid(dev; nx=n, Lx=L)

K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0       # normalize forcing to inject energy at rate ε
nothing # hide

if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
nothing # hide

random_uniform = dev==CPU() ? rand : CUDA.rand

function calcF!(Fh, sol, t, clock, vars, params, grid)
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)

  return nothing
end
nothing # hide

prob = BarotropicQGQL.Problem(dev; nx=n, Lx=L, β, μ, dt, stepper,
                              calcF=calcF!, stochastic=true, aliased_fraction=0)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x, grid.y
Lx, Ly = grid.Lx, grid.Ly
nothing # hide

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

fig = Figure()

ax = Axis(fig[1, 1],
          xlabel = "x",
          ylabel = "y",
          aspect = 1,
          title = "a forcing realization",
          limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

heatmap!(ax, x, y, Array(irfft(vars.Fh, grid.nx));
         colormap = :balance, colorrange = (-8, 8))

fig

BarotropicQGQL.set_zeta!(prob, device_array(dev)(zeros(grid.nx, grid.ny)))
nothing # hide

E = Diagnostic(BarotropicQGQL.energy, prob; nsteps)
Z = Diagnostic(BarotropicQGQL.enstrophy, prob; nsteps)
nothing # hide

zetaMean(prob) = prob.sol[1, :]

zMean = Diagnostic(zetaMean, prob; nsteps, freq=10)  # the zonal-mean vorticity
nothing # hide

diags = [E, Z, zMean] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_forcedbetaQLturb"
plotname = "snapshots"
filename = joinpath(filepath, "forcedbetaQLturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = prob.sol # extracts the Fourier-transformed solution

function get_u(prob)
  grid, vars = prob.grid, prob.vars

  @. vars.uh = im * grid.l  * grid.invKrsq * sol
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))

  return  vars.u
end

out = Output(prob, filename, (:sol, get_sol), (:u, get_u))

title_ζ = Observable(@sprintf("vorticity, μt = %.2f", μ * clock.t))
title_ψ = "streamfunction ψ"

fig = Figure(resolution=(1000, 600))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axζ = Axis(fig[1, 1]; title = title_ζ, axis_kwargs...)

axψ = Axis(fig[2, 1]; title = title_ψ, axis_kwargs...)

axζ̄ = Axis(fig[1, 2],
           xlabel = "zonal mean ζ",
           ylabel = "y",
           aspect = 1,
           limits = ((-3, 3), (-Ly/2, Ly/2)))

axū = Axis(fig[2, 2],
           xlabel = "zonal mean u",
           ylabel = "y",
           aspect = 1,
           limits = ((-0.5, 0.5), (-Ly/2, Ly/2)))

axE = Axis(fig[1, 3],
           xlabel = "μ t",
           ylabel = "energy",
           aspect = 1,
           limits = ((-0.1, 4.1), (0, 0.05)))

axZ = Axis(fig[2, 3],
           xlabel = "μ t",
           ylabel = "enstrophy",
           aspect = 1,
           limits = ((-0.1, 4.1), (0, 5)))

ζ̄, ζ′= prob.vars.Zeta, prob.vars.zeta
ζ = Observable(Array(@. ζ̄ + ζ′))
ψ̄, ψ′= prob.vars.Psi,  prob.vars.psi
ψ = Observable(Array(@. ψ̄ + ψ′))
ζ̄ₘ = Observable(Array(vec(mean(ζ̄, dims=1))))
ūₘ = Observable(Array(vec(mean(prob.vars.U, dims=1))))

μt = Observable(μ * E.t[1:1])
energy = Observable(E.data[1:1])
enstrophy = Observable(Z.data[1:1])

heatmap!(axζ, x, y, ζ;
         colormap = :balance, colorrange = (-8, 8))

heatmap!(axψ, x, y, ψ;
         colormap = :viridis, colorrange = (-0.22, 0.22))

lines!(axζ̄, ζ̄ₘ, y; linewidth = 3)
lines!(axζ̄, 0y, y; linewidth = 1, linestyle=:dash)

lines!(axū, ūₘ, y; linewidth = 3)
lines!(axū, 0y, y; linewidth = 1, linestyle=:dash)

lines!(axE, μt, energy; linewidth = 3)
lines!(axZ, μt, enstrophy; linewidth = 3, color = :red)

nothing # hide

startwalltime = time()

frames = 0:round(Int, nsteps / nsubs)

record(fig, "barotropicqgql_betaforced.mp4", frames, framerate = 18) do j
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u .+ vars.U) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i],
      (time()-startwalltime)/60)

    println(log)
  end

  ζ[] = @. ζ̄ + ζ′
  ψ[] = @. ψ̄ + ψ′
  ζ̄ₘ[] = vec(mean(ζ̄, dims=1))
  ūₘ[] = vec(mean(prob.vars.U, dims=1))

  μt.val = μ * E.t[1:E.i]
  energy[] = E.data[1:E.i]
  enstrophy[] = Z.data[1:E.i]

  title_ζ[] = @sprintf("vorticity, μt = %.2f", μ * clock.t)

  stepforward!(prob, diags, nsubs)
  BarotropicQGQL.updatevars!(prob)
end
nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

