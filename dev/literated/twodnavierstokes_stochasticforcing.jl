using GeophysicalFlows, CUDA, Random, Printf, CairoMakie

parsevalsum = FourierFlows.parsevalsum
record = CairoMakie.record

dev = CPU()     # Device (CPU/GPU)
nothing # hide

 n, L  = 256, 2π             # grid resolution and domain length
 ν, nν = 2e-7, 2             # hyperviscosity coefficient and hyperviscosity order
 μ, nμ = 1e-1, 0             # linear drag coefficient
    dt = 0.005               # timestep
nsteps = 4000                # total number of steps
 nsubs = 20                  # number of steps between each plot
nothing # hide

forcing_wavenumber = 14.0 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 0.1                           # energy input rate by the forcing

grid = TwoDGrid(dev; nx=n, Lx=L)

K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0        # normalize forcing to inject energy at rate ε
nothing # hide

if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
nothing # hide

random_uniform = dev==CPU() ? rand : CUDA.rand

function calcF!(Fh, sol, t, clock, vars, params, grid)
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)

  return nothing
end
nothing # hide

prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν, nν, μ, nμ, dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

x, y = grid.x, grid.y
nothing # hide

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

fig = Figure()

ax = Axis(fig[1, 1],
          xlabel = "x",
          ylabel = "y",
          aspect = 1,
          title = "a forcing realization",
          limits = ((-L/2, L/2), (-L/2, L/2)))

heatmap!(ax, x, y, Array(irfft(vars.Fh, grid.nx));
         colormap = :balance, colorrange = (-200, 200))

fig

TwoDNavierStokes.set_ζ!(prob, device_array(dev)(zeros(grid.nx, grid.ny)))

E  = Diagnostic(TwoDNavierStokes.energy,    prob; nsteps) # energy
Z  = Diagnostic(TwoDNavierStokes.enstrophy, prob; nsteps) # enstrophy
diags = [E, Z] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.
nothing # hide

ζ = Observable(Array(vars.ζ))
title_ζ = Observable("vorticity, μ t=" * @sprintf("%.2f", μ * clock.t))

energy = Observable(Point2f[(μ * E.t[1], E.data[1])])
enstrophy = Observable(Point2f[(μ * Z.t[1], Z.data[1] / forcing_wavenumber^2)])

fig = Figure(resolution = (800, 360))

axζ = Axis(fig[1, 1];
           xlabel = "x",
           ylabel = "y",
           title = title_ζ,
           aspect = 1,
           limits = ((-L/2, L/2), (-L/2, L/2)))

ax2 = Axis(fig[1, 2],
           xlabel = "μ t",
           limits = ((0, 1.1 * μ * nsteps * dt), (0, 0.55)))

heatmap!(axζ, x, y, ζ;
         colormap = :balance, colorrange = (-40, 40))

hE = lines!(ax2, energy; linewidth = 3)
hZ = lines!(ax2, enstrophy; linewidth = 3, color = :red)
Legend(fig[1, 3], [hE, hZ], ["energy E(t)" "enstrophy Z(t) / k_f²"])

fig

startwalltime = time()

record(fig, "twodturb_forced.mp4", 0:round(Int, nsteps / nsubs), framerate = 18) do j
  if j % (1000/nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Z: %.4f, walltime: %.2f min",
          clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)
    println(log)
  end

  ζ[] = vars.ζ

  energy[] = push!(energy[], Point2f(μ * E.t[E.i], E.data[E.i]))
  enstrophy[] = push!(enstrophy[], Point2f(μ * Z.t[E.i], Z.data[Z.i] / forcing_wavenumber^2))

  title_ζ[] = "vorticity, μ t=" * @sprintf("%.2f", μ * clock.t)

  stepforward!(prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(prob)
end
nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

