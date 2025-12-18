"""
    ConstrainedChemPotVSaturation <: SaturationMethod
    ChemPotVSaturation(V0)
    ChemPotVSaturation(;vl = nothing,
                        vv = nothing,
                        crit = nothing,
                        crit_retry = true
                        f_limit = 0.0,
                        atol = 1e-8,
                        rtol = 1e-12,
                        max_iters = 10^4,
                        s_scale = 1.0)
Constrained/Bounded version of ChemPotVSaturation. This method requires `crit`. If not provided it will be calculated.
Solves for the saturated volumes by constraining vv > vc, vl < vc, dP/dv|_v=vl < 0, dP/dv|_v=vv < 0 and P(vv) > 0. 
The problem is reformulated into a unconstrained root finding problem by making the firstly making the following transformations:
    ηc = vlb/vc, where vlb is the lower bound on the volume computed from lb_volume(model::EoSModel)
    vlb/vl = ηl = (1 - ηc)/(1 + exp(-xl)) + ηc, bounds ηl ∈ (ηc, 1)
    vlb/vv = ηv = ηc/(1 + exp(-xv)), bounds ηv ∈ (0, ηc)
To deal with three other constraints, we introduce 3 slack variables s1,s2,s3 and reformulate the problem as:
    F1 = P(xl) - P(xv) = 0
    F2 = μ(xl) - μ(xv) = 0
    F3 = dP/dv|_vl + exp(s1) = 0
    F4 = dP/dv|_vv + exp(s2) = 0
    F5 = P(xv) - exp(s3) = 0

The addition of equations F3,F4 in combinations the volume transformations ensures that we circumvent the trivial solution, vl=vv.
The addition of F5 ensures that the pressures remains positive at the vapor volume (and hence the liquid volume, if we converge).
"""
struct ConstrainedChemPotVSaturation{T,C} <: SaturationMethod
    vl::T
    vv::T
    crit::C
    f_limit::Float64
    atol::Float64
    rtol::Float64
    max_iters::Int
    s_scale::Float64
end

function ConstrainedChemPotVSaturation(;vl = nothing,
                                   vv = nothing,
                                   crit = nothing,
                                   f_limit = 0.0,
                                   atol = 1e-8,
                                   rtol = 1e-12,
                                   max_iters = 10000,
                                   s_scale = 1.0)
    if (vl === nothing) && (vv === nothing)
        return ConstrainedChemPotVSaturation{Nothing,typeof(crit)}(nothing,nothing,crit,f_limit,atol,rtol,max_iters,s_scale)
    elseif !(vl === nothing) && !(vv === nothing)
        T = one(vl)/one(vv)
        vl,vv,_ = promote(vl,vv,T)
        return ConstrainedChemPotVSaturation(vl,vv,crit,f_limit,atol,rtol,max_iters,s_scale)
    else
        throw(ArgumentError("you need to specify both vl and vv."))
    end
end

function saturation_pressure_impl(model::EoSModel, T, method::ConstrainedChemPotVSaturation{TT,Nothing}) where TT
    crit = crit_pure(model) # compute crit
    vl = method.vl
    vv = method.vv
    f_limit = method.f_limit
    atol = method.atol
    rtol = method.rtol
    max_iters = method.max_iters
    s_scale = method.s_scale
    new_method = ConstrainedChemPotVSaturation{TT,typeof(crit)}(vl,vv,crit,f_limit,atol,rtol,max_iters,s_scale)
    return saturation_pressure_impl(model,T,new_method)
end

ConstrainedChemPotVSaturation(x::Tuple) = ConstrainedChemPotVSaturation(vl = first(x),vv = last(x))
ConstrainedChemPotVSaturation(x::Vector) = ConstrainedChemPotVSaturation(vl = first(x),vv = last(x))

function sigmoidal(x;lb=0.0,ub=1.0,s=1.0)
    return (ub - lb)/(1 + exp(-s*x)) + lb
end

function inv_sigmoidal(x;lb=0.0,ub=1.0,s=1.0)
    return -log((ub - x)/(x - lb))/s
end

function μp_equality1_p_constrained(model,vl_raw,vv_raw,slack,T,ps,mus,lbv,eta_c,z=SA[1.0];s_scale=1.0)
    RT = Rgas(model)*T
    #transform variables
    etal = sigmoidal(vl_raw;lb=eta_c,s=s_scale)
    etav = sigmoidal(vv_raw;ub=eta_c,s=s_scale)
    vl = lbv/etal
    vv = lbv/etav
    #compute properties
    f1(V) = a_res(model,V,T,z)
    f2(V) = a_res(model,V,T,z)
    Al,dAl,d2Al = Solvers.f∂f∂2f(f1,vl)
    Av,dAv,d2Av =Solvers.f∂f∂2f(f2,vv)
    pl,pv = RT*(-dAl + 1/vl),RT*(-dAv + 1/vv)
    dpl,dpv = RT*(-d2Al - 1/vl^2),RT*(-d2Av - 1/vv^2)
    Δμᵣ = Al - vl*dAl - Av + vv*dAv + log(vv/vl)
    Fμ = Δμᵣ
    Fp = (pl - pv)*ps
    #constraints, if constraints are violated, we add slack variables
    F_dP1 = dpl < 0 ? 0.0 : dpl + exp(slack[1]) # dP/dv|_vl < 0
    F_dP2 = dpv < 0 ? 0.0 : dpv + exp(slack[2]) # dP/dv|_vv < 0
    F_p2 = pv > 0 ? 0.0 : pv - exp(slack[3]) # P(vv) > 0
    return SVector(Fμ,Fp,F_dP1,F_dP2,F_p2)
end

function μp_equality1_p_constrained!(F,model,vl_raw,vv_raw,slack,T,ps,mus,lbv,eta_c,z=SA[1.0];s_scale=1.0)
    vals = μp_equality1_p_constrained(model,vl_raw,vv_raw,slack,T,ps,mus,lbv,eta_c,z;s_scale=s_scale)
    for i in eachindex(F)
        F[i] = vals[i]
    end
    return F
end

function try_2ph_pure_pressure(model,T,vl,vv,ps,mus,method::ConstrainedChemPotVSaturation)
    TT = T*oneunit(eltype(model))
    s_scale=method.s_scale
    
    #determine critical point if not provided
    crit = method.crit
    if crit === nothing
        crit = crit_pure(model)
    end
    vc = crit[3]
    lbv = lb_volume(model)
    eta_c = lbv/vc

    x0 = [0.0,-10.0,0.0,0.0,0.0] # Default starting points on unconstrained space. Start at etal = 0.5*(1+eta_c), etav close to 0
    if vl !== nothing # convert provided starting points to unconstrained space
        @assert vl < vc "Initial liquid volume must be less than critical volume."
        @assert vv > vc "Initial vapor volume must be greater than critical volume."
        etal0 = lbv/vl
        etav0 = lbv/vv
        xl0 = inv_sigmoidal(etal0;lb=eta_c,s=s_scale)
        xv0 = inv_sigmoidal(etav0;ub=eta_c,s=s_scale)
        x0[1] = xl0
        x0[2] = xv0
    end

    f!(F,x) = μp_equality1_p_constrained!(F,model,x[1],x[2],x[3:end],TT,ps,mus,lbv,eta_c;s_scale=s_scale)
    solver_res = Solvers.nlsolve(f!, x0, TrustRegion(Newton(), NLSolvers.NWI()), NEqOptions(method))
    r = Solvers.x_sol(solver_res)
    # @show solver_res.info.best_residual
    max_res = maximum(abs.(solver_res.info.best_residual))
    converged = max_res <= solver_res.options.f_abstol
    vl = sigmoidal(r[1];lb=eta_c,s=s_scale)
    vl = lbv/vl
    vv = sigmoidal(r[2];ub=eta_c,s=s_scale)
    vv = lbv/vv
    return (vl,vv),converged,max_res # don't compute pressure yet. May have not converged, but may be close enough such that a different may work.
end

function saturation_pressure_impl(model::EoSModel, T, method::ConstrainedChemPotVSaturation)
    vl0 = method.vl
    vv0 = method.vv
    _0 = zero(T*oneunit(eltype(model)))
    nan = _0/_0
    fail = (nan,nan,nan)
    ps,μs = equilibria_scale(model)
    (vl,vv),converged,max_res = try_2ph_pure_pressure(model,T,vl0,vv0,ps,μs,method)
    if !converged
        if max_res < 1 # we are close to optimum, so maybe ChemPotV to refine.
            @info "ConstrainedChemPotV did not converge, but max residual $max_res < 1. Trying ChemPotV to refine."
            chempot_method = ChemPotVSaturation(;vl=vl,vv=vv, crit=method.crit,
                                                f_limit=method.f_limit,
                                                atol=method.atol,
                                                rtol=method.rtol,
                                                max_iters=method.max_iters)
            return saturation_pressure_impl(model,T,chempot_method)
        end
        return fail
    end
    p = pressure(model,vl,T)
    return (p,vl,vv)
end

export ConstrainedChemPotVSaturation
