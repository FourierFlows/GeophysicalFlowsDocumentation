using GeophysicalFlows, Plots, Printf, Random

using Statistics: mean

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution: n² grid points
stepper = "FilteredRK4"  # timestepper
     dt = 0.04           # timestep
 nsteps = 2000           # total number of time-steps
 nsubs  = 20             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.0       # bottom drag
nothing # hide

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β=β, μ=μ,
                                  dt=dt, stepper=stepper, aliased_fraction=0)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

E₀ = 0.08 # energy of initial condition

K = @. sqrt(grid.Krsq)                          # a 2D array with the total wavenumber

Random.seed!(1234)
q₀h = ArrayType(dev)(randn(Complex{eltype(grid)}, size(sol)))
@. q₀h = ifelse(K < 6  * 2π/L, 0, q₀h)
@. q₀h = ifelse(K > 10 * 2π/L, 0, q₀h)
@. q₀h[1, :] = 0    # remove any power from zonal wavenumber k=0
q₀h *= sqrt(E₀ / SingleLayerQG.energy(q₀h, vars, params, grid)) # normalize q₀ to have energy E₀
q₀ = irfft(q₀h, grid.nx)

SingleLayerQG.set_q!(prob, q₀)
nothing # hide

p1 = heatmap(x, y, Array(vars.q'),
         aspectratio = 1,
              c = :balance,
           clim = (-12, 12),
          xlims = (-grid.Lx/2, grid.Lx/2),
          ylims = (-grid.Ly/2, grid.Ly/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial vorticity ∂v/∂x-∂u/∂y",
     framestyle = :box)

p2 = contourf(x, y, Array(vars.ψ'),
        aspectratio = 1,
             c = :viridis,
        levels = range(-0.7, stop=0.7, length=20),
          clim = (-0.35, 0.35),
         xlims = (-grid.Lx/2, grid.Lx/2),
         ylims = (-grid.Ly/2, grid.Ly/2),
        xticks = -3:3,
        yticks = -3:3,
        xlabel = "x",
        ylabel = "y",
         title = "initial streamfunction ψ",
    framestyle = :box)

layout = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout = layout, size = (800, 360))

E = Diagnostic(SingleLayerQG.energy, prob; nsteps=nsteps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_decayingbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "decayingbetaturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
out = Output(prob, filename, (:sol, get_sol))
nothing # hide

function plot_output(prob)
  q = Array(prob.vars.q)
  ψ = Array(prob.vars.ψ)
  q̄ = Array(mean(q, dims=1)')
  ū = Array(mean(prob.vars.u, dims=1)')

  pq = heatmap(x, y, q',
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-12, 12),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, ψ',
       aspectratio = 1,
            legend = false,
                 c = :viridis,
            levels = range(-0.7, stop=0.7, length=20),
              clim = (-0.35, 0.35),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pqm = plot(q̄, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-2.2, 2.2),
            xlabel = "zonal mean q",
            ylabel = "y")
  plot!(pqm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ū, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.55, 0.55),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  layout = @layout Plots.grid(2, 2)
  p = plot(pq, pqm, pψ, pum, layout = layout, size = (800, 720))

  return p
end
nothing # hide

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)

  if j % round(Int, nsteps/nsubs / 4) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  p[1][1][:z] = Array(vars.q)
  p[1][:title] = "vorticity, t="*@sprintf("%.2f", clock.t)
  p[3][1][:z] = Array(vars.ψ)
  p[2][1][:x] = Array(mean(vars.q, dims=1)')
  p[4][1][:x] = Array(mean(vars.u, dims=1)')

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)

end

mp4(anim, "singlelayerqg_betadecay.mp4", fps=8)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

