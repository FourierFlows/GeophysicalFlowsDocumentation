using GeophysicalFlows, Printf, Random, CairoMakie

using GeophysicalFlows: peakedisotropicspectrum
using LinearAlgebra: ldiv!
using Random: seed!

dev = CPU()     # Device (CPU/GPU)
nothing # hide

n, L  = 128, 2π             # grid resolution and domain length
deformation_radius = 0.35   # the deformation radius
nothing # hide

# Then we pick the time-stepper parameters
    dt = 1e-2  # timestep
nsteps = 4000  # total number of steps
 nsubs = 20    # number of steps between each plot
nothing # hide

stepper="FilteredRK4"

prob_bqg = SingleLayerQG.Problem(dev; nx=n, Lx=L, dt, stepper, aliased_fraction=0)
prob_eqbqg = SingleLayerQG.Problem(dev; nx=n, Lx=L, deformation_radius, dt, stepper, aliased_fraction=0)
nothing # hide

seed!(1234)
k₀, E₀ = 6, 0.5
∇²ψ₀ = peakedisotropicspectrum(prob_bqg.grid, k₀, E₀, mask=prob_bqg.timestepper.filter)
nothing # hide

∇²ψ₀h = rfft(∇²ψ₀)
ψ₀h = @. 0 * ∇²ψ₀h
SingleLayerQG.streamfunctionfrompv!(ψ₀h, ∇²ψ₀h, prob_bqg.params, prob_bqg.grid)
nothing # hide

q₀_bqg   = irfft(-prob_bqg.grid.Krsq .* ψ₀h, prob_bqg.grid.nx)
q₀_eqbqg = irfft(-(prob_eqbqg.grid.Krsq .+ 1/prob_eqbqg.params.deformation_radius^2) .* ψ₀h, prob_bqg.grid.nx)
nothing # hide

SingleLayerQG.set_q!(prob_bqg, q₀_bqg)
SingleLayerQG.set_q!(prob_eqbqg, q₀_eqbqg)
nothing # hide

function relativevorticity(prob)
  vars, grid = prob.vars, prob.grid

  ldiv!(vars.q, grid.rfftplan, - grid.Krsq .* vars.ψh)

  return vars.q
end

x,  y  = prob_bqg.grid.x,  prob_bqg.grid.y
Lx, Ly = prob_bqg.grid.Lx, prob_bqg.grid.Ly

fig = Figure(resolution=(800, 380))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

t_bqg = Observable(prob_bqg.clock.t)
t_eqbqg = Observable(prob_eqbqg.clock.t)

title_bqg = @lift "barotropic\n ∇²ψ, t=" * @sprintf("%.2f", $t_bqg)
title_eqbqg = @lift "equivalent barotropic; deformation radius: " * @sprintf("%.2f", prob_eqbqg.params.deformation_radius) * "\n ∇²ψ, t=" * @sprintf("%.2f", $t_eqbqg)

ax1 = Axis(fig[1, 1]; title = title_bqg, axis_kwargs...)
ax2 = Axis(fig[1, 2]; title = title_eqbqg, axis_kwargs...)

ζ_bqg = Observable(Array(relativevorticity(prob_bqg)))
ζ_eqbqg = Observable(Array(relativevorticity(prob_eqbqg)))

heatmap!(ax1, x, y, ζ_bqg;
         colormap = :balance, colorrange = (-40, 40))

heatmap!(ax2, x, y, ζ_eqbqg;
         colormap = :balance, colorrange = (-40, 40))

fig

startwalltime = time()

cfl(prob) = prob.clock.dt * maximum([maximum(prob.vars.u) / prob.grid.dx, maximum(prob.vars.v) / prob.grid.dy])

record(fig, "singlelayerqg_barotropic_equivalentbarotropic.mp4", 0:Int(nsteps/nsubs), framerate = 18) do j
  if j % (1000 / nsubs) == 0
    log_bqg = @sprintf("barotropic; step: %04d, t: %d, cfl: %.2f, walltime: %.2f min",
        prob_bqg.clock.step, prob_bqg.clock.t, cfl(prob_bqg), (time()-startwalltime)/60)
    println(log_bqg)

    log_eqbqg = @sprintf("equivalent barotropic; step: %04d, t: %d, cfl: %.2f, walltime: %.2f min",
        prob_eqbqg.clock.step, prob_eqbqg.clock.t, cfl(prob_eqbqg), (time()-startwalltime)/60)
    println(log_eqbqg)
  end

  stepforward!(prob_bqg, nsubs)
  SingleLayerQG.updatevars!(prob_bqg)

  stepforward!(prob_eqbqg, nsubs)
  SingleLayerQG.updatevars!(prob_eqbqg)

  t_bqg[] = prob_bqg.clock.t
  t_eqbqg[] = prob_eqbqg.clock.t
  ζ_bqg[] = relativevorticity(prob_bqg)
  ζ_eqbqg[] = relativevorticity(prob_eqbqg)
end
nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

