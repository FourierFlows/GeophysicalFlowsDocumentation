using GeophysicalFlows, CUDA, JLD2, CairoMakie, Random, Printf

using Statistics: mean
using LinearAlgebra: ldiv!

parsevalsum = FourierFlows.parsevalsum
record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record
nothing # hide

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution: n² grid points
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 8000           # total number of timesteps
 save_substeps = 10      # number of timesteps after which output is saved

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
  T = eltype(grid)
  @. Fh = sqrt(forcing_spectrum) * cis(2π * random_uniform(T)) / sqrt(clock.dt)

  return nothing
end
nothing # hide

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β, μ, dt, stepper,
                             calcF=calcF!, stochastic=true)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
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

SingleLayerQG.set_q!(prob, device_array(dev)(zeros(grid.nx, grid.ny)))

E = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_forcedbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "singlelayerqg_forcedbeta.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution

function get_u(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  @. vars.qh = sol

  SingleLayerQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  ldiv!(vars.u, grid.rfftplan, -im * grid.l .* vars.ψh)

  return Array(vars.u)
end

output = Output(prob, filename, (:qh, get_sol), (:u, get_u))
nothing # hide

saveproblem(output)

saveoutput(output)

startwalltime = time()

while clock.step <= nsteps
  if clock.step % 50save_substeps == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  stepforward!(prob, diags, save_substeps)
  SingleLayerQG.updatevars!(prob)

  if clock.step % save_substeps == 0
    saveoutput(output)
  end
end

savediagnostic(E, "energy", output.path)
savediagnostic(Z, "enstrophy", output.path)

file = jldopen(output.path)

iterations = parse.(Int, keys(file["snapshots/t"]))
t = [file["snapshots/t/$i"] for i ∈ iterations]

qh = [file["snapshots/qh/$i"] for i ∈ iterations]
u  = [file["snapshots/u/$i"] for i ∈ iterations]

E_t  = file["diagnostics/energy/t"]
Z_t  = file["diagnostics/enstrophy/t"]
E_data = file["diagnostics/energy/data"]
Z_data = file["diagnostics/enstrophy/data"]

x,  y  = file["grid/x"],  file["grid/y"]
nx, ny = file["grid/nx"], file["grid/ny"]
Lx, Ly = file["grid/Lx"], file["grid/Ly"]

close(file)

j = Observable(1)

q = @lift irfft(qh[$j], nx)
ψ = @lift irfft(- Array(grid.invKrsq) .* qh[$j], nx)
q̄ = @lift real(ifft(qh[$j][1, :] / ny))
ū = @lift vec(mean(u[$j], dims=1))

title_q = @lift @sprintf("vorticity, μt = %.2f", μ * t[$j])

energy = Observable([Point2f(E_t[1], E_data[1])])
enstrophy = Observable([Point2f(Z_t[1], Z_data[1])])

fig = Figure(resolution=(1000, 600))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axq = Axis(fig[1, 1]; title = title_q, axis_kwargs...)

axψ = Axis(fig[2, 1]; title = "streamfunction ψ", axis_kwargs...)

axq̄ = Axis(fig[1, 2],
           xlabel = "zonal mean vorticity",
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
           limits = ((-0.1, 4.1), (0, 0.055)))

axZ = Axis(fig[2, 3],
           xlabel = "μ t",
           ylabel = "enstrophy",
           aspect = 1,
           limits = ((-0.1, 4.1), (0, 3.1)))

heatmap!(axq, x, y, q;
         colormap = :balance, colorrange = (-8, 8))

levels = collect(-0.32:0.04:0.32)

contourf!(axψ, x, y, ψ;
          levels, colormap = :viridis, colorrange = (-0.22, 0.22))
contour!(axψ, x, y, ψ;
         levels, color = :black)

lines!(axq̄, q̄, y; linewidth = 3)
lines!(axq̄, 0y, y; linewidth = 1, linestyle=:dash)

lines!(axū, ū, y; linewidth = 3)
lines!(axū, 0y, y; linewidth = 1, linestyle=:dash)

lines!(axE, energy; linewidth = 3)
lines!(axZ, enstrophy; linewidth = 3, color = :red)

fig

frames = 1:length(t)
record(fig, "singlelayerqg_betaforced.mp4", frames, framerate = 18) do i
  j[] = i

  energy[] = push!(energy[], Point2f(μ * E_t[i], E_data[i]))
  enstrophy[] = push!(enstrophy[], Point2f(μ * Z_t[i], Z_data[i]))
end
nothing # hide

# we delete the .jld2 file before deploying the docs (takes too much space) #hide
rm(output.path) #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

