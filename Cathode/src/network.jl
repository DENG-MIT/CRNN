
#17 parameters total.
#3 rxn orders 
#2 stoich coeffs (products only)
#3 Ea
#3 A
#3 b 
#3 delta H
np = 17+1 #with slope at the end
p = randn(Float64, np) .* 1.e-2;
p[1:3] .+= 1;  # A
#Ea initial condition. Roughly initialized to be in the right order (i.e. rxn 1 -> rxn 2 -> rxn 3)
Ea_IC=[1.0, 1.1, 1.2] 
p[4]+=Ea_IC[1] 
p[5]+=Ea_IC[2] 
p[6]+=Ea_IC[3] 

p[10] += 1; #delta H, roughly guessed based on how big the peaks in the data sort of look
p[11] += 0.2; #delta H
p[12] += 0.3; #delta H
p[13:15] .+= 1; #reaction order n 
p[16:17] .+= 1; #stoich coeff nu 

p[18]= 0.1; #slope, as per original CRNN code


function p2vec(p)
    #some clamps in place during debugging to make sure none of the parameters get too large or small 
    #these don't appear to be necessary in the final runs

    slope = p[end] .* 1.e1
    w_A = p[1:3] .* (slope * 20.0) #logA
    w_A = clamp.(w_A, 0, 50) 

    w_out = p[16:17] #product stoich. coeffs
    w_out=[1, w_out[1], w_out[2]]
    w_out=clamp.(w_out, 0.01, 5)
    w_in_order=p[13:15] #rxn orders
    w_in_order=clamp.(w_in_order, 0.01, 10)

    w_in_Ea = abs.(p[4:6]) #Ea
    w_in_Ea = clamp.(w_in_Ea, 0.0, 3)

    w_in_b = (p[7:9]) #non-exponential temp dependence, can be negative, no clamp

    w_delH = abs.(p[10:12])*100
    w_delH=clamp.(w_delH, 10, 300) 

    return w_in_Ea, w_in_b, w_out, w_delH, w_in_order, w_A
end

function display_p(p)
    w_in_Ea, w_in_b, w_out, w_delH, w_in_order, w_A = p2vec(p)
    println("\n species (column) reaction (row)")
    println("rxn ord | Ea | b | delH | lnA | stoich coeff")
    show(stdout, "text/plain", round.(hcat(w_in_order, w_in_Ea, w_in_b, w_delH, w_A, w_out), digits = 2))
    println("\n")
end

function getsampletemp(t, T0, beta)
    if beta[1] < 100
        T = T0 .+ beta[1] / 60 * t  # K/min to K/s
    end
    return T
end

const R = -1.0 / 8.314  # J/mol*K
@inbounds function crnn!(du, u, p, t) 
    #given a current concentration, parameter vector, and time
    #return the three concentration gradients
    logX = @. log(clamp(u, lb, 10.0)) 
    T = getsampletemp(t, T0, beta) 
    temp_term= reshape(hcat(log(T), R/T)*hcat(w_in_b, w_in_Ea*10^5)', 3)
    rxn_ord_term=w_in_order.*logX
    rxn_rates= @. exp(temp_term+rxn_ord_term+w_A )
    du .=  -rxn_rates #each reaction consumes the corresponding reactant
    #first and second reactions also produce c2 and c3:
    du[2]=du[2]+w_out[2]*rxn_rates[1]
    du[3]=du[3]+w_out[3]*rxn_rates[2]
end

function HRR_getter(times, u_outputs)
    #Take the concentration trajectories solved by crnn!(),
    #and compute the raw reaction rates, to multiply layer against dH to obtain the exothermic heat release.
    logX = @. log(clamp(u_outputs, lb, 10.0)) 
    T = getsampletemp(times, T0, beta) 
    temp_term=@. log(T).*w_in_b'+R/T*(w_in_Ea*10^5)'
    rxn_ord_term=transpose(w_in_order).*transpose(logX)
    rxn_rates= @. exp(temp_term+rxn_ord_term.+w_A' )
    return rxn_rates
end

tspan = [0.0, 1.0];
u0 = zeros(ns);
u0[1] = 1.0; #start with unity normalized mass of c1 only: c2 and c3 are produced *sequentially* as products.
prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb)

condition(u, t, integrator) = u[1] < lb * 10.0
affect!(integrator) = terminate!(integrator)
_cb = DiscreteCallback(condition, affect!)

alg = AutoTsit5(TRBDF2(autodiff = true));
function pred_n_ode(p, i_exp, exp_data)
    global beta = l_exp_info[i_exp, :]
    global T0=100+273.15 #degrees K 
    global w_in_Ea, w_in_b, w_out, w_delH, w_in_order, w_A = p2vec(p)
    ts = @view(exp_data[:, 1])
    tspan = [ts[1], ts[end]]

    #solve the species trajetory, which is independent of heat release (idealized DSC system):
    sol = solve(
        prob,
        alg,
        tspan = tspan,
        p = p,
        saveat = ts,
        maxiters = maxiters,
    )

    #post-processing: compute the heat release (actual desired solution) from the species trajectory:
    heat_rel= HRR_getter(ts, sol[:, :])*w_delH


    if sol.retcode == :Success
        nothing
    else
        @sprintf("solver failed beta: %.0f",  beta[1])
    end

    return heat_rel, ts, sol
end



function loss_neuralode(p, i_exp)
    exp_data = l_exp_data[i_exp]
    pred = Array(pred_n_ode(p, i_exp, exp_data)[1]) #index=1 for times

    loss = mae(pred, @view(exp_data[:, 2])) #index=2 for heat releases
    return loss
end

@time loss = loss_neuralode(p, 1)
