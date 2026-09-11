using GeophysicalFlows, CUDA, Random, Printf, Plots
using Statistics: mean
parsevalsum = FourierFlows.parsevalsum

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

grid = TwoDGrid(dev, n, L)

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

prob = BarotropicQGQL.Problem(dev; nx=n, Lx=L, β=β, μ=μ, dt=dt, stepper=stepper,
                              calcF=calcF!, stochastic=true, aliased_fraction=0)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

heatmap(x, y, Array(irfft(vars.Fh, grid.nx)'),
     aspectratio = 1,
               c = :balance,
            clim = (-8, 8),
           xlims = (-grid.Lx/2, grid.Lx/2),
           ylims = (-grid.Ly/2, grid.Ly/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)

BarotropicQGQL.set_zeta!(prob, ArrayType(dev)(zeros(grid.nx, grid.ny)))
nothing # hide

E = Diagnostic(BarotropicQGQL.energy, prob; nsteps=nsteps)
Z = Diagnostic(BarotropicQGQL.enstrophy, prob; nsteps=nsteps)
nothing # hide

function zetaMean(prob)
  sol = prob.sol
  sol[1, :]
end

zMean = Diagnostic(zetaMean, prob; nsteps=nsteps, freq=10)  # the zonal-mean vorticity
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

get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im * g.l .* g.invKrsq .* sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))

function plot_output(prob)
  ζ̄, ζ′= prob.vars.Zeta, prob.vars.zeta
  ζ = @. ζ̄ + ζ′
  ψ̄, ψ′= prob.vars.Psi,  prob.vars.psi
  ψ = @. ψ̄ + ψ′
  ζ̄ₘ = mean(ζ̄, dims=1)'
  ūₘ = mean(prob.vars.U, dims=1)'

  pζ = heatmap(x, y, Array(ζ'),
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ζ=∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, Array(ψ'),
            levels = -0.32:0.04:0.32,
       aspectratio = 1,
         linewidth = 1,
            legend = false,
              clim = (-0.22, 0.22),
                 c = :viridis,
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pζm = plot(Array(ζ̄ₘ), y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-3, 3),
            xlabel = "zonal mean ζ",
            ylabel = "y")
  plot!(pζm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(Array(ūₘ), y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.5, 0.5),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  pE = plot(1,
             label = "energy",
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 0.05),
            xlabel = "μt")

  pZ = plot(1,
             label = "enstrophy",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 5),
            xlabel = "μt")

  l = @layout Plots.grid(2, 3)
  p = plot(pζ, pζm, pE, pψ, pum, pZ, layout=l, size = (1000, 600))

  return p
end
nothing # hide

p = plot_output(prob)

startwalltime = time()

anim = @animate for j = 0:round(Int, nsteps / nsubs)
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u .+ vars.U) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i],
      (time()-startwalltime)/60)

    println(log)
  end

  p[1][1][:z] = Array(@. vars.zeta + vars.Zeta)
  p[1][:title] = "vorticity, μt=" * @sprintf("%.2f", μ * clock.t)
  p[4][1][:z] = Array(@. vars.psi + vars.Psi)
  p[2][1][:x] = Array(mean(vars.Zeta, dims=1)')
  p[5][1][:x] = Array(mean(vars.U, dims=1)')
  push!(p[3][1], μ * E.t[E.i], E.data[E.i])
  push!(p[6][1], μ * Z.t[Z.i], Z.data[Z.i])

  stepforward!(prob, diags, nsubs)
  BarotropicQGQL.updatevars!(prob)
end

mp4(anim, "barotropicqgql_betaforced.mp4", fps=18)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

