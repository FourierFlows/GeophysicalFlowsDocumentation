using GeophysicalFlows, Random, Printf, Plots
using FourierFlows: parsevalsum

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

grid = TwoDGrid(dev, n, L)

K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0        # normalize forcing to inject energy at rate ε
nothing # hide

if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
nothing # hide

random_uniform = dev==CPU() ? rand : CUDA.rand

function calcF!(Fh, sol, t, clock, vars, params, grid)
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)

  @CUDA.allowscalar Fh[1, 1] = 0 # make sure forcing has zero domain-average

  return nothing
end
nothing # hide

prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

x, y = grid.x, grid.y
nothing # hide

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

heatmap(x, y, Array(irfft(vars.Fh, grid.nx)'),
     aspectratio = 1,
               c = :balance,
            clim = (-200, 200),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)

TwoDNavierStokes.set_ζ!(prob, ArrayType(dev)(zeros(grid.nx, grid.ny)))

E  = Diagnostic(TwoDNavierStokes.energy,                prob, nsteps=nsteps) # energy
Z  = Diagnostic(TwoDNavierStokes.enstrophy,             prob, nsteps=nsteps) # enstrophy
diags = [E, Z] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.
nothing # hide

p1 = heatmap(x, y, Array(vars.ζ'),
         aspectratio = 1,
                   c = :balance,
                clim = (-40, 40),
               xlims = (-L/2, L/2),
               ylims = (-L/2, L/2),
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "vorticity, t=" * @sprintf("%.2f", clock.t),
          framestyle = :box)

p2 = plot(2, # this means "a plot with two series"
               label = ["energy E(t)" "enstrophy Z(t) / k_f²"],
              legend = :right,
           linewidth = 2,
               alpha = 0.7,
              xlabel = "μ t",
               xlims = (0, 1.1 * μ * nsteps * dt),
               ylims = (0, 0.55))

l = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout = l, size = (900, 420))

startwalltime = time()

anim = @animate for j = 0:round(Int, nsteps / nsubs)
  if j % (1000/nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Z: %.4f, walltime: %.2f min",
          clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)
    println(log)
  end

  p[1][1][:z] = Array(vars.ζ)
  p[1][:title] = "vorticity, μt = " * @sprintf("%.2f", μ * clock.t)
  push!(p[2][1], μ * E.t[E.i], E.data[E.i])
  push!(p[2][2], μ * Z.t[Z.i], Z.data[Z.i] / forcing_wavenumber^2)

  stepforward!(prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(prob)
end

mp4(anim, "twodturb_forced.mp4", fps=18)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

