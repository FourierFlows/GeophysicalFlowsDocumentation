using GeophysicalFlows, CairoMakie, Printf, Random

using Statistics: mean
using Random: seed!

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 256                       # 2D resolution = n²
stepper = "FilteredETDRK4"          # timestepper
     dt = 0.03                      # timestep
     tf = 60                        # length of time for simulation
 nsteps = Int(tf / dt)              # total number of time-steps
 nsubs  = round(Int, nsteps/100)    # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

 L = 2π        # domain size
 ν = 1e-19     # hyper-viscosity coefficient
nν = 4         # hyper-viscosity order
nothing # hide

prob = SurfaceQG.Problem(dev; nx=n, Lx=L, dt, stepper, ν, nν)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
Lx, Ly = grid.Lx, grid.Ly

X, Y = gridpoints(grid)
b₀ = @. exp(-(X^2 + 4Y^2))

SurfaceQG.set_b!(prob, b₀)
nothing # hide

fig = Figure(resolution = (500, 500))

ax = Axis(fig[1, 1],
          xlabel = "x",
          ylabel = "y",
          aspect = 1,
          title = "buoyancy bₛ",
          limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

hm = heatmap!(ax, x, y, Array(vars.b);
              colormap = :deep, colorrange = (0, 1))

Colorbar(fig[1, 2], hm)

fig

B  = Diagnostic(SurfaceQG.buoyancy_variance, prob; nsteps)
KE = Diagnostic(SurfaceQG.kinetic_energy, prob; nsteps)
Dᵇ = Diagnostic(SurfaceQG.buoyancy_dissipation, prob; nsteps)
diags = [B, KE, Dᵇ] # A list of Diagnostics types passed to `stepforward!`. Diagnostics are updated every timestep.
nothing # hidenothing # hide

base_filename = string("SurfaceQG_decaying_n_", n)

datapath = "./"
plotpath = "./"

dataname = joinpath(datapath, base_filename)
plotname = joinpath(plotpath, base_filename)
nothing # hide

if !isdir(plotpath); mkdir(plotpath); end
if !isdir(datapath); mkdir(datapath); end
nothing # hide

get_sol(prob) = prob.sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im * prob.grid.l .* sqrt.(prob.grid.invKrsq) .* prob.sol, prob.grid.nx)

out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide

b = Observable(Array(vars.b))

ke = Observable([Point2f(KE.t[1], KE.data[1])])
b² = Observable([Point2f(B.t[1], B.data[1])])

title_b = Observable("buoyancy, t=" * @sprintf("%.2f", clock.t))

fig = Figure(resolution = (900, 600))

axb = Axis(fig[1:2, 1];
           xlabel = "x",
           ylabel = "y",
           title = title_b,
           aspect = 1,
           limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axE = Axis(fig[1, 2];
           xlabel = "t",
           limits = ((0, tf), (0, 2e-2)))

heatmap!(axb, x, y, b;
         colormap = :deep, colorrange = (0, 1))

hE  = lines!(axE, ke; linewidth = 3)
hb² = lines!(axE, b²; linewidth = 3)

Legend(fig[2, 2], [hE, hb²], ["kinetic energy ∫½(uₛ²+vₛ²)dxdy/L²", "buoyancy variance ∫bₛ²dxdy/L²"])

fig

startwalltime = time()

record(fig, "sqg_ellipticalvortex.mp4", 0:round(Int, nsteps/nsubs), framerate = 14) do j
  if j % (500 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log1 = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
          clock.step, clock.t, cfl, (time()-startwalltime)/60)

    log2 = @sprintf("buoyancy variance: %.2e, buoyancy variance dissipation: %.2e",
              B.data[B.i], Dᵇ.data[Dᵇ.i])

    println(log1)

    println(log2)
  end

  b[] = vars.b

  ke[] = push!(ke[], Point2f(KE.t[KE.i], KE.data[KE.i]))
  b²[] = push!(b²[], Point2f(B.t[B.i], B.data[B.i]))

  title_b[] = "buoyancy, t=" * @sprintf("%.2f", clock.t)

  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)
end
nothing # hide

fig = Figure(resolution = (800, 380))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axb = Axis(fig[1, 1]; title = "bₛ(x, y, t=" * @sprintf("%.2f", clock.t) * ")", axis_kwargs...)
axu = Axis(fig[1, 2]; title = "uₛ(x, y, t=" * @sprintf("%.2f", clock.t) * ")", axis_kwargs...)
axv = Axis(fig[1, 3]; title = "vₛ(x, y, t=" * @sprintf("%.2f", clock.t) * ")", axis_kwargs...)

hb = heatmap!(axb, x, y, Array(vars.b);
             colormap = :deep, colorrange = (0, 1))

Colorbar(fig[2, 1], hb, vertical = false)

hu = heatmap!(axu, x, y, Array(vars.u);
             colormap = :balance, colorrange = (-maximum(abs.(vars.u)), maximum(abs.(vars.u))))

Colorbar(fig[2, 2], hu, vertical = false)

hv = heatmap!(axv, x, y, Array(vars.v);
             colormap = :balance, colorrange = (-maximum(abs.(vars.v)), maximum(abs.(vars.v))))

Colorbar(fig[2, 3], hv, vertical = false)

fig

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

