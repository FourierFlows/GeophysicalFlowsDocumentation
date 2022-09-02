using GeophysicalFlows, Plots, Printf

using Random: seed!

dev = CPU()     # Device (CPU/GPU)
nothing # hide

n = 128                  # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
     dt = 2.5e-3         # timestep
 nsteps = 20000          # total number of time-steps
 nsubs  = 50             # number of time-steps for plotting (nsteps must be multiple of nsubs)
nothing # hide

L = 2π                   # domain size
μ = 5e-2                 # bottom drag
β = 5                    # the y-gradient of planetary PV

nlayers = 2              # number of layers
f₀, g = 1, 1             # Coriolis parameter and gravitational constant
 H = [0.2, 0.8]          # the rest depths of each layer
 ρ = [4.0, 5.0]          # the density of each layer

 U = zeros(nlayers) # the imposed mean zonal flow in each layer
 U[1] = 1.0
 U[2] = 0.0
nothing # hide

prob = MultiLayerQG.Problem(nlayers, dev;
                            nx=n, Lx=L, f₀=f₀, g=g, H=H, ρ=ρ, U=U, μ=μ, β=β,
                            dt=dt, stepper=stepper, aliased_fraction=0)
nothing # hide

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y
nothing # hide

seed!(1234) # reset of the random number generator for reproducibility
q₀  = 1e-2 * ArrayType(dev)(randn((grid.nx, grid.ny, nlayers)))
q₀h = prob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
q₀  = irfft(q₀h, grid.nx, (1, 2))                 # apply irfft only in dims=1, 2

MultiLayerQG.set_q!(prob, q₀)
nothing # hide

E = Diagnostic(MultiLayerQG.energies, prob; nsteps=nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_2layer"
plotname = "snapshots"
filename = joinpath(filepath, "2layer.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
function get_u(prob)
  sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l * vars.ψh
  invtransform!(vars.u, vars.uh, params)

  return vars.u
end

out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

symlims(data) = maximum(abs.(extrema(data))) |> q -> (-q, q)

function plot_output(prob)
  Lx, Ly = prob.grid.Lx, prob.grid.Ly

  l = @layout Plots.grid(2, 3)
  p = plot(layout=l, size = (1000, 600))

  for m in 1:nlayers
    heatmap!(p[(m-1) * 3 + 1], x, y, Array(vars.q[:, :, m]'),
         aspectratio = 1,
              legend = false,
                   c = :balance,
               xlims = (-Lx/2, Lx/2),
               ylims = (-Ly/2, Ly/2),
               clims = symlims,
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "q_"*string(m),
          framestyle = :box)

    contourf!(p[(m-1) * 3 + 2], x, y, Array(vars.ψ[:, :, m]'),
              levels = 8,
         aspectratio = 1,
              legend = false,
                   c = :viridis,
               xlims = (-Lx/2, Lx/2),
               ylims = (-Ly/2, Ly/2),
               clims = symlims,
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "ψ_"*string(m),
          framestyle = :box)
  end

  plot!(p[3], 2,
             label = ["KE₁" "KE₂"],
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 2.35),
             ylims = (1e-9, 1e0),
            yscale = :log10,
            yticks = 10.0.^(-9:0),
            xlabel = "μt")

  plot!(p[6], 1,
             label = "PE",
            legend = :bottomright,
         linecolor = :red,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 2.35),
             ylims = (1e-9, 1e0),
            yscale = :log10,
            yticks = 10.0.^(-9:0),
            xlabel = "μt")

end
nothing # hide

p = plot_output(prob)

startwalltime = time()

anim = @animate for j = 0:round(Int, nsteps / nsubs)
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, walltime: %.2f min", clock.step, clock.t, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)

    println(log)
  end

  for m in 1:nlayers
    p[(m-1) * 3 + 1][1][:z] = Array(vars.q[:, :, m])
    p[(m-1) * 3 + 2][1][:z] = Array(vars.ψ[:, :, m])
  end

  push!(p[3][1], μ * E.t[E.i], E.data[E.i][1][1])
  push!(p[3][2], μ * E.t[E.i], E.data[E.i][1][2])
  push!(p[6][1], μ * E.t[E.i], E.data[E.i][2][1])

  stepforward!(prob, diags, nsubs)
  MultiLayerQG.updatevars!(prob)
end

mp4(anim, "multilayerqg_2layer.mp4", fps=18)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

