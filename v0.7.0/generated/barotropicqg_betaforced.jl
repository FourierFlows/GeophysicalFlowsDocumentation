using FourierFlows, Plots, Statistics, Printf, Random

using FourierFlows: parsevalsum
using FFTW: irfft
using Statistics: mean
using Random: seed!

import GeophysicalFlows.BarotropicQG
import GeophysicalFlows.BarotropicQG: energy, enstrophy

dev = CPU()    # Device (CPU/GPU)
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

forcing_wavenumber = 14.0    # the central forcing wavenumber for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5     # the width of the forcing spectrum
ε = 0.001                    # energy input rate by the forcing

gr  = TwoDGrid(n, L)

K  = @. sqrt(gr.Krsq)                      # a 2D array with the total wavenumber
k = [gr.kr[i] for i=1:gr.nkr, j=1:gr.nl]   # a 2D array with the zonal wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
forcing_spectrum[K .<  2 * 2π/L] .= 0    # no power at low wavenumbers
forcing_spectrum[K .> 20 * 2π/L] .= 0    # no power at high wavenumbers
forcing_spectrum[k .< 2π/L] .= 0         # make sure forcing does not have power at k=0
ε0 = parsevalsum(forcing_spectrum .* gr.invKrsq / 2, gr) / (gr.Lx * gr.Ly)
@. forcing_spectrum *= ε/ε0               # normalize forcing to inject energy at rate ε

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

function calcFq!(Fh, sol, t, clock, vars, params, grid)
  ξ = ArrayType(dev)(exp.(2π * im * rand(eltype(grid), size(sol))) / sqrt(clock.dt))
  @. Fh = ξ*sqrt.(forcing_spectrum)
  Fh[abs.(grid.Krsq) .== 0] .= 0
  return nothing
end
nothing # hide

prob = BarotropicQG.Problem(dev; nx=n, Lx=L, β=β, μ=μ, dt=dt, stepper=stepper,
                            calcFq=calcFq!, stochastic=true)
nothing # hide

sol, cl, vs, pr, gr = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gr.x, gr.y
nothing # hide

calcFq!(vs.Fqh, sol, 0.0, cl, vs, pr, gr)

heatmap(x, y, irfft(vs.Fqh, gr.nx),
     aspectratio = 1,
               c = :balance,
            clim = (-8, 8),
           xlims = (-gr.Lx/2, gr.Lx/2),
           ylims = (-gr.Ly/2, gr.Ly/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)

BarotropicQG.set_zeta!(prob, zeros(gr.nx, gr.ny))

E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_forcedbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "forcedbetaturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.l.*gr.invKrsq.*sol, gr.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

function plot_output(prob)
  ζ = prob.vars.zeta
  ψ = prob.vars.psi
  ζ̄ = mean(ζ, dims=1)'
  ū = mean(prob.vars.u, dims=1)'

  pζ = heatmap(x, y, ζ,
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ζ=∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, ψ,
            levels = -0.32:0.04:0.32,
       aspectratio = 1,
         linewidth = 1,
            legend = false,
              clim = (-0.22, 0.22),
                 c = :viridis,
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pζm = plot(ζ̄, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-3, 3),
            xlabel = "zonal mean ζ",
            ylabel = "y")
  plot!(pζm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ū, y,
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
             ylims = (0, 2.5),
            xlabel = "μt")

  l = @layout grid(2, 3)
  p = plot(pζ, pζm, pE, pψ, pum, pZ, layout=l, size = (1000, 600), dpi=150)

  return p
end
nothing # hide

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = cl.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])

  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
  cl.step, cl.t, cfl, E.data[E.i], Z.data[Z.i],
  (time()-startwalltime)/60)

  if j%(1000/nsubs)==0; println(log) end

  p[1][1][:z] = Array(vs.zeta)
  p[1][:title] = "vorticity, μt="*@sprintf("%.2f", μ*cl.t)
  p[4][1][:z] = Array(vs.psi)
  p[2][1][:x] = mean(vs.zeta, dims=1)'
  p[5][1][:x] = mean(vs.u, dims=1)'
  push!(p[3][1], μ*E.t[E.i], E.data[E.i])
  push!(p[6][1], μ*Z.t[Z.i], Z.data[Z.i])

  stepforward!(prob, diags, nsubs)
  BarotropicQG.updatevars!(prob)

end

mp4(anim, "barotropicqg_betaforced.mp4", fps=18)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

