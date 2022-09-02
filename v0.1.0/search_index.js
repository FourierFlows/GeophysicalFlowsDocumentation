var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#GeophysicalFlows.jl-Documentation-1",
    "page": "Home",
    "title": "GeophysicalFlows.jl Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Overview-1",
    "page": "Home",
    "title": "Overview",
    "category": "section",
    "text": "GeophysicalFlows.jl is a collection of modules which leverage the  FourierFlows.jl framework to provide solvers for problems in Geophysical Fluid Dynamics, on periodic domains and using Fourier-based pseudospectral  methods."
},

{
    "location": "index.html#Developers-1",
    "page": "Home",
    "title": "Developers",
    "category": "section",
    "text": "GeophysicalFlows is currently being developed by Gregory L. Wagner and Navid C. Constantinou."
},

{
    "location": "modules/twodturb.html#",
    "page": "TwoDTurb Module",
    "title": "TwoDTurb Module",
    "category": "page",
    "text": ""
},

{
    "location": "modules/twodturb.html#TwoDTurb-Module-1",
    "page": "TwoDTurb Module",
    "title": "TwoDTurb Module",
    "category": "section",
    "text": "newcommandJmathsfJ"
},

{
    "location": "modules/twodturb.html#Basic-Equations-1",
    "page": "TwoDTurb Module",
    "title": "Basic Equations",
    "category": "section",
    "text": "This module solves two-dimensional incompressible turbulence. The flow is given through a streamfunction psi as (uupsilon) = (-partial_ypsi partial_xpsi). The dynamical variable used here is the component of the vorticity of the flow normal to the plane of motion, q=partial_x upsilon- partial_y u = nabla^2psi. The equation solved by the module is:partial_t q + J(psi q) = underbrace-leftmu(-1)^n_mu nabla^2n_mu\n+nu(-1)^n_nu nabla^2n_nuright q_textrmdissipation + f where J(a b) = (partial_x a)(partial_y b)-(partial_y a)(partial_x b). On the right hand side, f(xyt) is forcing, mu is hypoviscosity, and nu is hyperviscosity. Plain old linear drag corresponds to n_mu=0, while normal viscosity corresponds to n_nu=1."
},

{
    "location": "modules/twodturb.html#Implementation-1",
    "page": "TwoDTurb Module",
    "title": "Implementation",
    "category": "section",
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatq = - widehatJ(psi q) -left(mu k^2n_mu\n+nu k^2n_nuright) widehatq  + widehatf In doing so the Jacobian is computed in the conservative form: J(ab) = partial_y  (partial_x a) b -partial_x (partial_y a) b.Thus:mathcalL = -mu k^-2n_mu - nu k^2n_nu mathcalN(widehatq) = - mathrmik_x mathrmFFT(u q)-\n	mathrmik_y mathrmFFT(upsilon q) + widehatf "
},

{
    "location": "modules/twodturb.html#AbstractTypes-and-Functions-1",
    "page": "TwoDTurb Module",
    "title": "AbstractTypes and Functions",
    "category": "section",
    "text": "ParamsFor the unforced case (f=0) parameters AbstractType is build with Params and it includes:nu:   Float; viscosity or hyperviscosity coefficient.\nnnu: Integer0; the order of viscosity n_nu. Case n_nu=1 give normal viscosity.\nmu: Float; bottom drag or hypoviscosity coefficient.\nnmu: Integerge 0; the order of hypodrag n_mu. Case n_mu=0 give plain linear drag mu.For the forced case (fne 0) parameters AbstractType is build with ForcedParams. It includes all parameters in Params and additionally:calcF!: Function that calculates the forcing widehatfVarsFor the unforced case (f=0) variables AbstractType is build with Vars and it includes:q: Array of Floats; relative vorticity.\nU: Array of Floats; x-velocity, u.\nV: Array of Floats; y-velocity, v.\nsol: Array of Complex; the solution, widehatq.\nqh: Array of Complex; the Fourier transform widehatq.\nUh: Array of Complex; the Fourier transform widehatu.\nVh: Array of Complex; the Fourier transform widehatv.For the forced case (fne 0) variables AbstractType is build with ForcedVars. It includes all variables in Vars and additionally:Fh: Array of Complex; the Fourier transform widehatf.\nprevsol: Array of Complex; the values of the solution sol at the previous time-step (useful for calculating the work done by the forcing).calcN! functionThe nonlinear term mathcalN(widehatq) is computed via functions:calcN_advection!: computes - widehatJ(psi q) and stores it in array N.function calcN_advection!(N, sol, t, s, v, p, g)\n  @. v.Uh =  im * g.l  * g.invKKrsq * sol\n  @. v.Vh = -im * g.kr * g.invKKrsq * sol\n  @. v.qh = sol\n\n  A_mul_B!(v.U, g.irfftplan, v.Uh)\n  A_mul_B!s(v.V, g.irfftplan, v.Vh)\n  A_mul_B!(v.q, g.irfftplan, v.qh)\n\n  @. v.U *= v.q # U*q\n  @. v.V *= v.q # V*q\n\n  A_mul_B!(v.Uh, g.rfftplan, v.U) # \\hat{U*q}\n  A_mul_B!(v.Vh, g.rfftplan, v.V) # \\hat{U*q}\n\n  @. N = -im*g.kr*v.Uh - im*g.l*v.Vh\n  nothing\nendcalcN_forced!: computes - widehatJ(psi q) via calcN_advection! and then adds to it the forcing widehatf computed via calcF! function. Also saves the solution widehatq of the previous time-step in array prevsol.function calcN_forced!(N, sol, t, s, v, p, g)\n  calcN_advection!(N, sol, t, s, v, p, g)\n  if t == s.t # not a substep\n    v.prevsol .= s.sol # used to compute budgets when forcing is stochastic\n    p.calcF!(v.Fh, sol, t, s, v, p, g)\n  end\n  @. N += v.Fh\n  nothing\nendupdatevars!: uses sol to compute q, u, v, widehatu, and widehatv and stores them into corresponding arrays of Vars/ForcedVars."
},

{
    "location": "modules/twodturb.html#Examples-1",
    "page": "TwoDTurb Module",
    "title": "Examples",
    "category": "section",
    "text": "examples/twodturb/McWilliams.jl: A script that simulates decaying two-dimensional turbulence reproducing the results of the paper by\nMcWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. J. Fluid Mech., 146, 21-43.\nexamples/twodturb/IsotropicRingForcing.jl: A script that simulates stochastically forced two-dimensional turbulence. The forcing is temporally delta-corraleted and its spatial structure is isotropic with power in a narrow annulus of total radius k_f in wavenumber space."
},

{
    "location": "modules/barotropicqg.html#",
    "page": "BarotropicQG Module",
    "title": "BarotropicQG Module",
    "category": "page",
    "text": ""
},

{
    "location": "modules/barotropicqg.html#BarotropicQG-Module-1",
    "page": "BarotropicQG Module",
    "title": "BarotropicQG Module",
    "category": "section",
    "text": "newcommandJmathsfJ"
},

{
    "location": "modules/barotropicqg.html#Basic-Equations-1",
    "page": "BarotropicQG Module",
    "title": "Basic Equations",
    "category": "section",
    "text": "This module solves the quasi-geostrophic barotropic vorticity equation on a beta-plane of variable fluid depth H-h(xy). The flow is obtained through a streamfunction psi as (u upsilon) = (-partial_ypsi partial_xpsi). All flow fields can be obtained from the quasi-geostrophic potential vorticity (QGPV). Here the QGPV isunderbracef_0 + beta y_textplanetary PV + underbrace(partial_x upsilon\n	- partial_y u)_textrelative vorticity +\n	underbracefracf_0 hH_texttopographic PVThe dynamical variable is the component of the vorticity of the flow normal to the plane of motion, zetaequiv partial_x upsilon- partial_y u = nabla^2psi. Also, we denote the topographic PV with etaequiv f_0 hH. Thus, the equation solved by the module is:partial_t zeta + J(psi underbracezeta + eta_equiv q) +\nbetapartial_xpsi = underbrace-leftmu + nu(-1)^n_nu nabla^2n_nu\nright zeta _textrmdissipation + f where J(a b) = (partial_x a)(partial_y b)-(partial_y a)(partial_x b). On the right hand side, f(xyt) is forcing, mu is linear drag, and nu is hyperviscosity. Plain old viscosity corresponds to n_nu=1. The sum of relative vorticity and topographic PV is denoted with qequivzeta+eta."
},

{
    "location": "modules/barotropicqg.html#Implementation-1",
    "page": "BarotropicQG Module",
    "title": "Implementation",
    "category": "section",
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatzeta = - widehatJ(psi q) +betafracmathrmik_xk^2widehatzeta -left(mu\n+nu k^2n_nuright) widehatzeta  + widehatf In doing so the Jacobian is computed in the conservative form: J(fg) = partial_y  (partial_x f) g -partial_x (partial_y f) g.Thus:mathcalL = betafracmathrmik_xk^2 - mu - nu k^2n_nu mathcalN(widehatzeta) = - mathrmik_x mathrmFFT(u q)-\n	mathrmik_y mathrmFFT(upsilon q) "
},

{
    "location": "modules/barotropicqg.html#Examples-1",
    "page": "BarotropicQG Module",
    "title": "Examples",
    "category": "section",
    "text": "examples/barotropicqg/decayingbetaturb.jl: An script that simulates decaying quasi-geostrophic flow on a beta-plane demonstrating zonation.\nexamples/barotropicqg/forcedbetaturb.jl: An script that simulates forced-dissipative quasi-geostrophic flow on a beta-plane demonstrating zonation. The forcing is temporally delta-corraleted and its spatial structure is isotropic with power in a narrow annulus of total radius kf in wavenumber space.\nexamples/barotropicqg/ACConelayer.jl: A script that simulates barotropic quasi-geostrophic flow above topography reproducing the results of the paper by\nConstantinou, N. C. (2018). A barotropic model of eddy saturation. J. Phys. Oceanogr., 48 (2), 397-411."
},

{
    "location": "modules/boussinesq.html#",
    "page": "Thin-layer Boussinesq modules",
    "title": "Thin-layer Boussinesq modules",
    "category": "page",
    "text": ""
},

{
    "location": "modules/boussinesq.html#Thin-layer-Boussinesq-modules-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Thin-layer Boussinesq modules",
    "category": "section",
    "text": "newcommandbcdotboldsymbol cdot\nnewcommandbnablaboldsymbol nabla\nnewcommandpnablabnabla_ perp\n\nnewcommandcom \nnewcommandper \n\nnewcommandbuboldsymbol u\nnewcommandbUboldsymbol U\nnewcommandbuuboldsymbolu\nnewcommandbbb\nnewcommandppp\nnewcommandwww\nnewcommanduuu\nnewcommandvupsilon\nnewcommandvvupsilon\nnewcommandzzetazeta\nnewcommandoomegaomega\nnewcommandboomegaboldsymboloomega\n\nnewcommandbxhwidehatboldsymbolx\nnewcommandbyhwidehatboldsymboly\nnewcommandbzhwidehatboldsymbolz\nnewcommandiimathrmi\nnewcommandeemathrme\nnewcommandccmathrmcc\nnewcommandJmathsfJ\n\nnewcommandppartialThese modules solve various thin-layer approximations to the hydrostatic Boussinesq equations. A thin-layer approximation is one that is appropriate for dynamics with small aspect ratios, or small vertical scales and large horizontal scales. Thin layer approximations include the shallow-water system, layered system, and spectral approximations that apply a Fourier or Sin/Cos eigenfunction expansion in the vertical coordinate to the Boussinesq equations, and truncate the expansion at just two or three modes. Approximations of this last flavor are described here.The three-dimensional rotating, stratified, hydrostatic Boussinesq equations arep_tbuu + left ( buu bcdot bnabla right ) buu + f bzh times buu + bnabla pp = D^buu com \np_z pp = bb com \np_tbb + ww N^2 = D^bb com \nbnabla bcdot buu = 0 comwhere bu = (u v w) is the three-dimensional velocity, b is buoyancy, p is pressure, N^2 is the buoyancy frequency (constant), and f is the rotation or Coriolis frequency. The operators D^buu and D^bb are arbitrary dissipation that we define only after projecting onto vertical Fourier or Sin/Cos modes. Taking the curl of the horizontal momentum equation yields an evolution equation for vertical vorticity, zzeta = p_x vv - p_y uu:p_tzzeta + buu bcdot bnabla zzeta - left (f bzh + boomega right )\n    bcdot bnabla ww = D^zzeta per"
},

{
    "location": "modules/boussinesq.html#Vertically-Fourier-Boussinesq-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Vertically Fourier Boussinesq",
    "category": "section",
    "text": "The vertically-Fourier Boussinesq module solves the Boussinesq system obtained by expanding the hydrostatic Boussinesq equations in a Fourier series. The horizontal velocity uu, for example, is expanded withuu(x y z t) mapsto U(x y t) + ee^ii m z u(x y t) + ee^-ii m z u^*(x y t) comThe other variables vv, bb, pp, zzeta, and boomega are expanded identically. The barotropic horizontal velocity is V and the barotropic vertical vorticity is Z = p_x V - p_y U. The barotropic vorticity obeysp_t Z + J left ( Psi Z right )\n    + bnabla bcdot left ( bu zeta^* right ) + ii m pnabla bcdot left ( bu w^* right ) + cc\n    = D_0 Z comwhere cc denotes the complex conjugate and contraction with pnabla = -p_y bxh + p_x byh gives the vertical component of the curl.The baroclinic components obeyp_t u - f v + p_x p = - J left ( Psi u right ) - bu bcdot bnabla U + D_1 u com \np_t v + f u + p_y p = - J left ( Psi v right ) - bu bcdot bnabla V + D_1 v com \np_t p - tfracN^2m w = - J left ( Psi p right ) + D_1 p perThe dissipation operators are definedD_0 = nu_0 (-1)^n_0 nabla^2n_0 + mu_0 (-1)^m_0 nabla^2m_0 com \nD_1 = nu_1 (-1)^n_1 nabla^2n_1 + mu_1 (-1)^m_1 nabla^2m_1where U is the barotropic velocity and u is the amplitude of the first baroclinic mode with periodic vertical structure mathrme^mathrmi m z."
},

{
    "location": "modules/boussinesq.html#Implementation-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Implementation",
    "category": "section",
    "text": "Coming soon."
},

{
    "location": "modules/boussinesq.html#Vertically-Cosine-Boussinesq-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Vertically Cosine Boussinesq",
    "category": "section",
    "text": "The vertically-Cosine Boussinesq module solves the Boussinesq system obtained by expanding the hydrostatic Boussinesq equations in a Sin/Cos series. The horizontal velocity, for example, becomesuu(x y z t) mapsto U(x y t) + cos(mz) u(x y t) perThe horizontal velocity vv, pressure pp, and vertical vorticity zzeta are also expanded in cos(mz), where Z = p_x V - p_y U denotes the barotropic component of the vertical vorticity. The vertical velocity ww and buoyancy bb are expanded with sin(mz)."
},

{
    "location": "modules/boussinesq.html#Basic-governing-equations-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Basic governing equations",
    "category": "section",
    "text": "Projecting the vertical vorticity equation onto Sin/Cos modes an equation for the evolution of Z,p_t Z + J left ( Psi Z right )\n    + tfrac12 bnabla bcdot left ( bu zeta right ) + tfracm2 pnabla bcdot left ( bu w right )\n    = D_0 Z comwhere J(a b) = (p_x a)(p_y b) - (p_y a)(p_x b) is the Jacobian operator, contraction with pnabla = -p_y bxh + p_x byh gives the vertical component of the curl, and Psi is the barotropic streamfunction defined so thatbU = -p_yPsi bxh + p_xPsi byh qquad textand qquad Z = nabla^2 Psi perThe baroclinic components obeyp_t u - f v + p_x p = - J left ( Psi u right ) - bu bcdot bnabla U + D_1u com \np_t v + f u + p_y p = - J left ( Psi v right ) - bu bcdot bnabla V + D_1v com \np_t p - tfracN^2m w = - J left ( Psi p right ) + D_1p perThe dissipation operators are definedD_0 = nu_0 (-1)^n_0 nabla^2n_0 + mu_0 (-1)^m_0 nabla^2m_0 com \nD_1 = nu_1 (-1)^n_1 nabla^2n_1 + mu_1 (-1)^m_1 nabla^2m_1 comwhere 2n_0 and 2m_0 are the hyperviscous orders of the arbitrary barotropic dissipation operators with coefficients nu_0 and mu_0, while 2n_1 and 2m_1 are the orders of the baroclinic dissipation operators.A passive tracer in the Vertically Cosine Boussinesq system is assumed to satisfy a no-flux condition at the upper and lower boundaries, and thus expanded in cosine modes so thatc(x y z t) = C(x y t) + cos(mz) c(x y t) perThe barotropic and baroclinic passive tracer components then obeyp_t C + J(Psi C) + tfrac12 bnabla bcdot left ( bu c right ) =\n    kappa (-1)^n_kappa nabla^2n_kappa C com \np_t c + J(Psi c) + bu bcdot bnabla C = kappa (-1)^n_kappa nabla^2n_kappa c comwhere kappa and n_kappa are the tracer hyperdiffusivity and order of the hyperdiffusivity, respectively. The choice n_kappa = 1 corresponds to ordinary Fickian diffusivity."
},

{
    "location": "modules/boussinesq.html#Implementation-2",
    "page": "Thin-layer Boussinesq modules",
    "title": "Implementation",
    "category": "section",
    "text": "Coming soon."
},

{
    "location": "man/types.html#",
    "page": "Private types",
    "title": "Private types",
    "category": "page",
    "text": ""
},

{
    "location": "man/types.html#Private-types-1",
    "page": "Private types",
    "title": "Private types",
    "category": "section",
    "text": ""
},

{
    "location": "man/types.html#Private-types-in-module-GeophysicalFlows:-1",
    "page": "Private types",
    "title": "Private types in module GeophysicalFlows:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#GeophysicalFlows.TwoDTurb.ForcedVars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.TwoDTurb.ForcedVars",
    "category": "method",
    "text": "ForcedVars(g)\n\nReturns the vars for forced two-dimensional turbulence with grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.TwoDTurb.Params",
    "page": "Private types",
    "title": "GeophysicalFlows.TwoDTurb.Params",
    "category": "type",
    "text": "Params(nu, nnu, mu, nmu, calcF!)\n\nReturns the params for two-dimensional turbulence.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.TwoDTurb.StochasticForcedVars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.TwoDTurb.StochasticForcedVars",
    "category": "method",
    "text": "StochasticForcedVars(g; T)\n\nReturns the vars for stochastically forced two-dimensional turbulence with grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.TwoDTurb.Vars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.TwoDTurb.Vars",
    "category": "method",
    "text": "Vars(g)\n\nReturns the vars for unforced two-dimensional turbulence with grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-TwoDTurb:-1",
    "page": "Private types",
    "title": "Private types in module TwoDTurb:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.TwoDTurb]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#GeophysicalFlows.BarotropicQG.ForcedVars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.BarotropicQG.ForcedVars",
    "category": "method",
    "text": "ForcedVars(g)\n\nReturns the vars for forced two-dimensional barotropic QG problem with grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.BarotropicQG.Params",
    "page": "Private types",
    "title": "GeophysicalFlows.BarotropicQG.Params",
    "category": "type",
    "text": "Params(g::TwoDGrid, f0, beta, FU, eta, mu, nu, nnu, calcFU, calcFq)\n\nReturns the params for an unforced two-dimensional barotropic QG problem.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.BarotropicQG.Params-Tuple{FourierFlows.TwoDGrid,Any,Any,Function,Any,Any,Any,Any,Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.BarotropicQG.Params",
    "category": "method",
    "text": "Params(g::TwoDGrid, f0, beta, eta::Function, mu, nu, nnu, calcFU, calcFq)\n\nConstructor for Params that accepts a generating function for the topographic PV.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.BarotropicQG.StochasticForcedVars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.BarotropicQG.StochasticForcedVars",
    "category": "method",
    "text": "StochasticForcedVars(g)\n\nReturns the vars for stochastically forced two-dimensional barotropic QG problem with grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.BarotropicQG.Vars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.BarotropicQG.Vars",
    "category": "method",
    "text": "Vars(g)\n\nReturns the vars for unforced two-dimensional barotropic QG problem with grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-BarotropicQG:-1",
    "page": "Private types",
    "title": "Private types in module BarotropicQG:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.BarotropicQG]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#GeophysicalFlows.VerticallyFourierBoussinesq.Params",
    "page": "Private types",
    "title": "GeophysicalFlows.VerticallyFourierBoussinesq.Params",
    "category": "type",
    "text": "Params(nu0, nnu0, nu1, nnu1, f, N, m, Ub, Vb)\n\nConstruct parameters for the Two-Fourier-mode Boussinesq problem. Suffix 0 refers to zeroth mode; 1 to first mode. f, N, m are Coriolis frequency, buoyancy frequency, and vertical wavenumber of the first mode, respectively. The optional constant background velocity (Ub,Vb) is set to zero by default. The viscosity is applied only to the first-mode horizontal velocities.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-VerticallyFourierBoussinesq:-1",
    "page": "Private types",
    "title": "Private types in module VerticallyFourierBoussinesq:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.VerticallyFourierBoussinesq]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#GeophysicalFlows.VerticallyCosineBoussinesq.ForcedVars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.ForcedVars",
    "category": "method",
    "text": "ForcedVars(g)\n\nReturns the vars for forced two-vertical-cosine-mode Boussinesq dynamics on the grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.VerticallyCosineBoussinesq.Params",
    "page": "Private types",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.Params",
    "category": "type",
    "text": "Params(nu0, nnu0, nu1, nnu1, mu0, nmu0, mu1, nmu1, f, N, m; Ub=0, Vb=0)\n\nConstruct parameters for the Two-Fourier-mode Boussinesq problem. Suffix 0 refers to zeroth mode; 1 to first mode. f, N, m are Coriolis frequency, buoyancy frequency, and vertical wavenumber of the first mode, respectively. The optional constant background velocity (Ub, Vb) is set to zero by default. The viscosity is applied only to the first-mode horizontal velocities.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.VerticallyCosineBoussinesq.TracerForcedVars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.TracerForcedVars",
    "category": "method",
    "text": "TracerForcedVars(g)\n\nReturns the vars for forced two-vertical-cosine-mode Boussinesq dynamics on the grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#GeophysicalFlows.VerticallyCosineBoussinesq.Vars-Tuple{Any}",
    "page": "Private types",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.Vars",
    "category": "method",
    "text": "Vars(g)\n\nReturns the vars for unforced two-vertical-cosine-mode Boussinesq dynamics on the grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-VerticallyCosineBoussinesq:-1",
    "page": "Private types",
    "title": "Private types in module VerticallyCosineBoussinesq:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.VerticallyCosineBoussinesq]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/functions.html#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "man/functions.html#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": ""
},

{
    "location": "man/functions.html#Functions-exported-from-GeophysicalFlows:-1",
    "page": "Functions",
    "title": "Functions exported from GeophysicalFlows:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.Problem-Tuple{}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.Problem",
    "category": "method",
    "text": "Problem(; parameters...)\n\nConstruct a 2D turbulence problem.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.dissipation-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.dissipation",
    "category": "method",
    "text": "dissipation(prob)\ndissipation(s, v, p, g)\n\nReturns the domain-averaged dissipation rate. nnu must be >= 1.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.drag-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.drag",
    "category": "method",
    "text": "drag(prob)\ndrag(s, v, p, g)\n\nReturns the extraction of domain-averaged energy by drag/hypodrag mu.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.energy-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.energy",
    "category": "method",
    "text": "energy(prob)\nenergy(s, v, g)\n\nReturns the domain-averaged kinetic energy in the Fourier-transformed vorticity solution s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.enstrophy-Tuple{Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.enstrophy",
    "category": "method",
    "text": "enstrophy(s, g)\n\nReturns the domain-averaged enstrophy in the Fourier-transformed vorticity solution s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.set_q!-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.set_q!",
    "category": "method",
    "text": "set_q!(prob, q)\nset_q!(s, v, g, q)\n\nSet the solution s.sol as the transform of q and update variables v on the grid g.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.updatevars!-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.updatevars!",
    "category": "method",
    "text": "updatevars!(v, s, g)\n\nUpdate the vars in v on the grid g with the solution in s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.TwoDTurb.work-Tuple{Any,GeophysicalFlows.TwoDTurb.ForcedVars,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.TwoDTurb.work",
    "category": "method",
    "text": "work(prob)\nwork(s, v, p, g)\n\nReturns the domain-averaged rate of work of energy by the forcing Fh.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#Functions-exported-from-TwoDTurb:-1",
    "page": "Functions",
    "title": "Functions exported from TwoDTurb:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.TwoDTurb]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.Problem-Tuple{}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.Problem",
    "category": "method",
    "text": "Problem(; parameters...)\n\nConstruct a BarotropicQG turbulence problem.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.dissipation-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.dissipation",
    "category": "method",
    "text": "dissipation(prob)\ndissipation(s, v, p, g)\n\nReturns the domain-averaged dissipation rate. nnu must be >= 1.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.drag-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.drag",
    "category": "method",
    "text": "drag(prob)\ndrag(s, v, p, g)\n\nReturns the extraction of domain-averaged energy by drag mu.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.energy-Tuple{FourierFlows.AbstractProblem}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.energy",
    "category": "method",
    "text": "Calculate the domain-averaged kinetic energy.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.enstrophy-Tuple{Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.enstrophy",
    "category": "method",
    "text": "Returns the domain-averaged enstrophy.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.meanenergy-Tuple{Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.meanenergy",
    "category": "method",
    "text": "Returns the energy of the domain-averaged U.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.meanenstrophy-Tuple{Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.meanenstrophy",
    "category": "method",
    "text": "Returns the enstrophy of the domain-averaged U.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.set_zeta!-Tuple{Any,GeophysicalFlows.BarotropicQG.Vars,Any,Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.set_zeta!",
    "category": "method",
    "text": "set_zeta!(prob, zeta)\nset_zeta!(s, v, g, zeta)\n\nSet the solution s.sol as the transform of zeta and update variables v on the grid g.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.updatevars!-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.updatevars!",
    "category": "method",
    "text": "updatevars!(v, s, g)\n\nUpdate the vars in v on the grid g with the solution in s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.BarotropicQG.work-Tuple{Any,GeophysicalFlows.BarotropicQG.ForcedVars,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.BarotropicQG.work",
    "category": "method",
    "text": "work(prob)\nwork(s, v, p, g)\n\nReturns the domain-averaged rate of work of energy by the forcing Fqh.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#Functions-exported-from-BarotropicQG:-1",
    "page": "Functions",
    "title": "Functions exported from BarotropicQG:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.BarotropicQG]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#Functions-exported-from-VerticallyFourierBoussinesq:-1",
    "page": "Functions",
    "title": "Functions exported from VerticallyFourierBoussinesq:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.VerticallyFourierBoussinesq]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode0apv-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode0apv",
    "category": "method",
    "text": "mode0apv(prob)\n\nReturns the barotropic available potential vorticity.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode0apv-NTuple{7,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode0apv",
    "category": "method",
    "text": "mode0apv(uh, vh, ph, Zh, m, N, g)\n\nReturns the barotropic available potential vorticity.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode0dissipation-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode0dissipation",
    "category": "method",
    "text": "mode0dissipation(prob)\n\nReturns the domain-averaged barotropic dissipation rate. nnu0 must be >= 1.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode0drag-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode0drag",
    "category": "method",
    "text": "mode0drag(prob)\n\nReturns the extraction of domain-averaged barotropic energy by drag μ.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode0energy-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode0energy",
    "category": "method",
    "text": "mode0energy(prob)\n\nReturns the domain-averaged energy in the zeroth mode.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode0enstrophy-Tuple{Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode0enstrophy",
    "category": "method",
    "text": "mode0enstrophy(prob)\n\nReturns the domain-averaged enstrophy in the Fourier-transformed vorticity solution s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode1dissipation-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode1dissipation",
    "category": "method",
    "text": "mode1dissipation(prob)\n\nReturns the domain-averaged kinetic energy dissipation of the first mode by horizontal viscosity.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode1drag-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode1drag",
    "category": "method",
    "text": "mode1drag(prob)\n\nReturns the domain-averaged kinetic energy dissipation of the first mode by horizontal viscosity.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode1energy-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode1energy",
    "category": "method",
    "text": "mode1energy(prob)\n\nReturns the domain-averaged total energy in the first mode.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode1ke-Tuple{Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode1ke",
    "category": "method",
    "text": "mode1ke(prob)\n\nReturns the domain-averaged kinetic energy in the first mode.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.mode1pe-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.mode1pe",
    "category": "method",
    "text": "mode1pe(prob)\n\nReturns the domain-averaged potential energy in the first mode.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.set_C!-NTuple{5,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.set_C!",
    "category": "method",
    "text": "set_C!(prob, C)\n\nSet zeroth mode tracer concentration and update vars.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.set_Z!-NTuple{5,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.set_Z!",
    "category": "method",
    "text": "set_Z!(prob, Z)\n\nSet zeroth mode vorticity and update vars.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.set_planewave!",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.set_planewave!",
    "category": "function",
    "text": "set_planewave!(prob, u₀, κ, θ=0)\n\nSet a plane wave solution with initial speed u₀, non-dimensional wave number κ, and angle θ with the horizontal. The non-dimensional wavenumber vector is (k, l) = (κ cos θ, κ sin θ), is normalized by 2π/Lx, and is rounded to the nearest integer.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.set_uvp!-NTuple{7,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.set_uvp!",
    "category": "method",
    "text": "set_uvp!(prob)\n\nSet first mode u, v, and p and update vars.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.totalenergy-NTuple{4,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.totalenergy",
    "category": "method",
    "text": "totalenergy(prob)\n\nReturns the total energy projected onto the zeroth mode.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#GeophysicalFlows.VerticallyCosineBoussinesq.updatevars!-NTuple{5,Any}",
    "page": "Functions",
    "title": "GeophysicalFlows.VerticallyCosineBoussinesq.updatevars!",
    "category": "method",
    "text": "updatevars!(prob)\n\nUpdate variables to correspond to the solution in s.sol or prob.state.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#Functions-exported-from-VerticallyCosineBoussinesq:-1",
    "page": "Functions",
    "title": "Functions exported from VerticallyCosineBoussinesq:",
    "category": "section",
    "text": "Modules = [GeophysicalFlows.VerticallyCosineBoussinesq]\nPrivate = false\nOrder = [:function]"
},

]}
