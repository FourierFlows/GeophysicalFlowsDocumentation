using GeophysicalFlows, CairoMakie, Printf, Random

using Statistics: mean

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
topographicPV(x, y) = 3exp(-(x - 1)^2 / 2σx^2 - (y - 1)^2 / 2σy^2) - 2exp(- (x + 1)^2 / 2σx^2 - (y + 1)^2 / 2σy^2)
nothing # hide

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, eta=topographicPV,
                             dt, stepper, aliased_fraction=0)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
Lx, Ly = grid.Lx, grid.Ly
nothing # hide

η = Array(params.eta)

fig = Figure()
ax = Axis(fig[1, 1];
          xlabel = "x",
          ylabel = "y",
          title = "topographic PV η=f₀h/H",
          limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

contourf!(ax, x, y, η;
          levels = collect(-3:0.4:3), colormap = :balance, colorrange = (-3, 3))

fig

E₀ = 0.04 # energy of initial condition

K = @. sqrt(grid.Krsq)                             # a 2D array with the total wavenumber

Random.seed!(1234)
qih = device_array(dev)(randn(Complex{eltype(grid)}, size(sol)))
@. qih = ifelse(K < 6  * 2π/L, 0, qih)
@. qih = ifelse(K > 12 * 2π/L, 0, qih)
qih *= sqrt(E₀ / SingleLayerQG.energy(qih, vars, params, grid))  # normalize qi to have energy E₀
qi = irfft(qih, grid.nx)

SingleLayerQG.set_q!(prob, qi)
nothing # hide

q = Observable(Array(vars.q))
ψ = Observable(Array(vars.ψ))

fig = Figure(resolution=(800, 380))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

title_q = Observable("initial vorticity ∂v/∂x-∂u/∂y")
axq = Axis(fig[1, 1]; title = title_q, axis_kwargs...)

title_ψ = Observable("initial streamfunction ψ")
axψ = Axis(fig[1, 3]; title = title_ψ, axis_kwargs...)

hm = heatmap!(axq, x, y, q;
         colormap = :balance, colorrange = (-8, 8))

Colorbar(fig[1, 2], hm)

levels = collect(range(-0.28, stop=0.28, length=11))

hc = contourf!(axψ, x, y, ψ;
          levels, colormap = :viridis, colorrange = (-0.28, 0.28),
          extendlow = :auto, extendhigh = :auto)
contour!(axψ, x, y, ψ;
         levels, color = :black)

Colorbar(fig[1, 4], hc)

fig

E = Diagnostic(SingleLayerQG.energy, prob; nsteps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
filename = joinpath(filepath, "decayingbetaturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
nothing # hide

get_sol(prob) = prob.sol # extracts the Fourier-transformed solution
out = Output(prob, filename, (:sol, get_sol))
nothing # hide

contour!(axq, x, y, η;
         levels = collect(0.5:0.5:3), linewidth = 2, color = (:black, 0.5))

contour!(axq, x, y, η;
         levels = collect(-2:0.5:-0.5), linewidth = 2, color = (:grey, 0.7), linestyle = :dash)

title_q[] = "vorticity, t=" * @sprintf("%.2f", clock.t)
title_ψ[] = "streamfunction ψ"

nothing # hide

startwalltime = time()

record(fig, "singlelayerqg_decaying_topography.mp4", 0:round(Int, nsteps/nsubs), framerate = 12) do j
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  q[] = vars.q
  ψ[] = vars.ψ

  title_q[] = "vorticity, t="*@sprintf("%.2f", clock.t)
  title_ψ[] = "streamfunction ψ"

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)
end
nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

