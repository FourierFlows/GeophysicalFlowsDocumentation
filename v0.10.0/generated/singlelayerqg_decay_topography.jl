using FourierFlows, Plots, Printf, Random

using Statistics: mean
using FFTW: irfft

import GeophysicalFlows.SingleLayerQG
import GeophysicalFlows.SingleLayerQG: energy, enstrophy

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 2000           # total number of time-steps
 nsubs  = 10             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

L = 2π        # domain size
nothing # hide

σx, σy = 0.4, 0.8
topographicPV(x, y) = 3exp(-(x-1)^2/(2σx^2) -(y-1)^2/(2σy^2)) - 2exp(-(x+1)^2/(2σx^2) -(y+1)^2/(2σy^2))
nothing # hide

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, eta=topographicPV, dt=dt, stepper=stepper)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

contourf(grid.x, grid.y, params.eta',
          aspectratio = 1,
            linewidth = 0,
               levels = 10,
                    c = :balance,
                 clim = (-3, 3),
                xlims = (-grid.Lx/2, grid.Lx/2),
                ylims = (-grid.Ly/2, grid.Ly/2),
               xticks = -3:3,
               yticks = -3:3,
               xlabel = "x",
               ylabel = "y",
                title = "topographic PV η=f₀h/H")

E₀ = 0.04 # energy of initial condition

K = @. sqrt(grid.Krsq)                             # a 2D array with the total wavenumber

Random.seed!(1234)
qih = randn(Complex{eltype(grid)}, size(sol))
@. qih = ifelse(K < 6  * 2π/L, 0, qih)
@. qih = ifelse(K > 12 * 2π/L, 0, qih)
qih *= sqrt(E₀ / energy(qih, vars, params, grid))  # normalize qi to have energy E₀
qi = irfft(qih, grid.nx)

SingleLayerQG.set_q!(prob, qi)
nothing # hide

p1 = heatmap(x, y, vars.q',
         aspectratio = 1,
              c = :balance,
           clim = (-8, 8),
          xlims = (-grid.Lx/2, grid.Lx/2),
          ylims = (-grid.Ly/2, grid.Ly/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial vorticity ∂v/∂x-∂u/∂y",
     framestyle = :box)

p2 = contourf(x, y, vars.ψ',
        aspectratio = 1,
             c = :viridis,
        levels = range(-0.35, stop=0.35, length=10),
          clim = (-0.35, 0.35),
         xlims = (-grid.Lx/2, grid.Lx/2),
         ylims = (-grid.Ly/2, grid.Ly/2),
        xticks = -3:3,
        yticks = -3:3,
        xlabel = "x",
        ylabel = "y",
         title = "initial streamfunction ψ",
    framestyle = :box)

l = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout=l, size=(900, 400))

E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
filename = joinpath(filepath, "decayingbetaturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
out = Output(prob, filename, (:sol, get_sol))
nothing # hide

function plot_output(prob)
  q = prob.vars.q
  ψ = prob.vars.ψ
  η = prob.params.eta

  pq = heatmap(x, y, q',
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-6, 6),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ∂v/∂x-∂u/∂y",
        framestyle = :box)

  contour!(pq, x, y, η',
          levels=0.5:0.5:3,
          lw=2, c=:black, ls=:solid, alpha=0.7)

  contour!(pq, x, y, η',
          levels=-2:0.5:-0.5,
          lw=2, c=:black, ls=:dash, alpha=0.7)

  pψ = contourf(x, y, ψ',
       aspectratio = 1,
            legend = false,
                 c = :viridis,
            levels = range(-0.7, stop=0.7, length=10),
              clim = (-0.7, 0.7),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  l = @layout Plots.grid(1, 2)
  p = plot(pq, pψ, layout = l, size = (900, 400))

  return p
end
nothing # hide

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)

  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  p[1][1][:z] = vars.q
  p[1][:title] = "vorticity, t="*@sprintf("%.2f", clock.t)
  p[2][1][:z] = vars.ψ

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)
end

gif(anim, "barotropicqg_decay_topography.gif", fps=12)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

