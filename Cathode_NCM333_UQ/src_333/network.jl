#17 parameters total.
#3 rxn orders 
#2 stoich coeffs 
#3 Ea
#3 A
#3 b 
#3 delta H
np = 17 
p = zeros(1,np);

@load string(ckpt_path_det, "/mymodel.bson") p_opt

#all parameters are normalized to be equal to 1 at the optimal determinsitic value
p[1,1:17].=1. 
#scale by the deterministic optimal value, so that p=1 --> p=p_opt
p_scales=deepcopy(p_opt)

slope_from_det=p_scales[18]*10
p_scales[1:3]=p_scales[1:3].*(20*slope_from_det) #lnA scaling
println("loaded delta H values:")
println(p_scales[10:12].*100)
p_scales[10:12]=p_scales[10:12].*100 #dH scaling

p=repeat(p, num_particles, 1)+0.001 * (randn!(zeros(num_particles,size(p)[2]))) #add random noise

##intialize lnA and Ea to have a slight correlation as per the governing laws
RT=8.314*(270+273.15) #roughly the middle of the R1 peaks at 225C
point_picker=randn!(zeros(num_particles, 1))
lnA_init=point_picker .+ p_scales[1]
Ea_init=point_picker*RT/10^5 .+ p_scales[4]
p[:, 1]=lnA_init./p_scales[1]
p[:, 4]=Ea_init./p_scales[4]

RT=8.314*(310+273.15) #roughly the middle of the R1 peaks at 275C
point_picker=randn!(zeros(num_particles, 1))
lnA_init=point_picker .+ p_scales[2]
Ea_init=point_picker*RT/10^5 .+ p_scales[5]
p[:, 2]=lnA_init./p_scales[2]
p[:, 5]=Ea_init./p_scales[5]

RT=8.314*(430+273.15) #roughly the middle of the R1 peaks at 410C
point_picker=randn!(zeros(num_particles, 1))
lnA_init=point_picker .+ p_scales[3]
Ea_init=point_picker*RT/10^5 .+ p_scales[6]
p[:, 3]=lnA_init./p_scales[3]
p[:, 6]=Ea_init./p_scales[6]

mutable struct SVGD
    p_his
    sol_his
end

function extend_dims(A,which_dim) #for expanding p in sigma()
    s = [size(A)...]
    insert!(s,which_dim,1)
    return reshape(A, s...)
end

function norm2(A; dims) #for taking the norm of a slice of p in sigma()
    B = sum(x -> x^2, A; dims=dims)
    B .= sqrt.(B)
    return B
end



function svgd_kernel(svgd, p, h=-1)
    distances=pairwise(Euclidean(), p, dims = 1)
    sq_dist=distances[tril!(trues(size(distances)), -1)] #non-diagonal terms
    pairwise_dists = distances.^2

    if h < 0 # if h < 0, using median trick
        h = median(sq_dist)^2 
        h = sqrt(0.5*h / log(size(p)[1]+1))
    end

    Kxy = exp.( -pairwise_dists / h^2 / 2) # the RBF kernel function

    dxkxy = -Kxy*p
    sumkxy = sum(Kxy, dims=2)

    for i = 1:size(p)[2]
        dxkxy[:, i] = dxkxy[:,i] + p[:,i].*sumkxy
    end
    dxkxy = dxkxy / (h^2)
    return Kxy, dxkxy
end


function p2vec(p)

    w_A_temp = p[1:3]  
    w_out_temp = p[16:17] 
    w_out_temp=[1, w_out_temp[1], w_out_temp[2]]
    w_in_order_temp=p[13:15]

    w_in_Ea_temp = (p[4:6]) 


    w_in_b_temp = (p[7:9]) 


    w_delH_temp = (p[10:12])


    return w_in_Ea_temp, w_in_b_temp, w_out_temp, w_delH_temp, w_in_order_temp, w_A_temp
end

function p2vec_reporting(p)

    w_A_report = p[:,1:3]  
    w_out_temp = p[:,16:17] 
    w_out_report=hcat(ones(size(p)[1]), w_out_temp[:,1], w_out_temp[:,2])
    w_in_order_report=p[:,13:15]

    w_in_Ea_report = (p[:,4:6])


    w_in_b_report = (p[:,7:9]) 


     w_delH_report = (p[:,10:12])


    return w_in_Ea_report, w_in_b_report, w_out_report, w_delH_report, w_in_order_report, w_A_report
end


function display_p(p)

    w_in_Ea_report, w_in_b_report, w_out_report, w_delH_report, w_in_order_report, w_A_report=p2vec_reporting(p)
    println("\n species (column) reaction (row)")
    println("rxn ord | Ea | b | delH | lnA | stoich coeff")
    println("means")
    show(stdout, "text/plain", round.(hcat(mean(w_in_order_report.*p_scales[13:15]', dims=1)', mean(w_in_Ea_report.*p_scales[4:6]', dims=1)', mean(w_in_b_report.*p_scales[7:9]', dims=1)', mean(w_delH_report.*p_scales[10:12]', dims=1)', mean(w_A_report.*p_scales[1:3]', dims=1)', mean(w_out_report.*[1, p_scales[16], p_scales[17]]', dims=1)'), digits = 2))
    println("")
    println("standard deviations:")
    show(stdout, "text/plain", round.(hcat(std(w_in_order_report.*p_scales[13:15]', dims=1)', std(w_in_Ea_report.*p_scales[4:6]', dims=1)', std(w_in_b_report.*p_scales[7:9]', dims=1)', std(w_delH_report.*p_scales[10:12]', dims=1)', std(w_A_report.*p_scales[1:3]', dims=1)', std(w_out_report.*[1, p_scales[16], p_scales[17]]', dims=1)'), digits = 2))

    println("\n")
end

function getsampletemp(t, T0, beta)
    if beta[1] < 100
        T = T0 .+ beta[1] / 60 * t  # K/min to K/s
    
    end
    return T
end

const R = -1.0 / 8.314  # universal gas constant, J/mol*K
@inbounds function crnn!(du, u, p_for_crnn, t)
    w_in_Ea_temp, w_in_b_temp, w_out_temp, w_delH_temp, w_in_order_temp, w_A_temp=p_for_crnn
    logX = @. log(clamp(u, lb_clamp, 10.0)) #log of species concentrations
    T = getsampletemp(t, T0, beta) #current temperature 
    #all params have to be scaled back up for physically meaningful application in ode's
    temp_term= reshape(hcat(log(T), R/T)*hcat(w_in_b_temp.*p_scales[7:9], w_in_Ea_temp.*p_scales[4:6]*10^5)', 3)
    rxn_ord_term=w_in_order_temp.*p_scales[13:15].*logX
    rxn_rates= @. exp(temp_term+rxn_ord_term+w_A_temp.*p_scales[1:3] )
    du .=  -rxn_rates #consume each of three reactants at specified rates
    #produce c2 and c3 at the according rates:
    du[2]=du[2]+w_out_temp[2]*p_scales[16]*rxn_rates[1]
    du[3]=du[3]+w_out_temp[3]*p_scales[17]*rxn_rates[2]

end

function HRR_getter(times, u_outputs, p_hrr)
    logX = @. log(clamp(u_outputs, lb_clamp, 10.0)) 
    T = getsampletemp(times, T0, beta) 
    temp_term=@. log(T).*(p_hrr[7:9].*p_scales[7:9])'+R/T*(p_hrr[4:6].*p_scales[4:6]*10^5)'
    rxn_ord_term=transpose(p_hrr[13:15].*p_scales[13:15]).*transpose(logX)
    rxn_rates= @. exp(temp_term+rxn_ord_term.+(p_hrr[1:3].*p_scales[1:3])' )
    hrr=rxn_rates*(p_hrr[10:12].*p_scales[10:12])
    return hrr
end

function HRR_getter_post(times, u_outputs, p_hrr)
    logX = @. log(clamp(u_outputs, lb_clamp, 10.0)) #
    T = getsampletemp(times, T0, beta) 
    temp_term=@. log(T).*(p_hrr[7:9].*p_scales[7:9])'+R/T*(p_hrr[4:6].*p_scales[4:6]*10^5)'
    rxn_ord_term=transpose(p_hrr[13:15].*p_scales[13:15]).*transpose(logX)
    rxn_rates= @. exp(temp_term+rxn_ord_term.+(p_hrr[1:3].*p_scales[1:3])' )
    return rxn_rates
end
tspan = [0.0, 1.0];
u0 = zeros(ns);
u0[1] = 1.0;
prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb_abstol)
global const T0=100+273.15

condition(u, t, integrator) = u[1] < lb_abstol * 10.0
affect!(integrator) = terminate!(integrator)
_cb = DiscreteCallback(condition, affect!)

alg = AutoTsit5(TRBDF2(autodiff = true));
function pred_n_ode(p_temp, i_exp, exp_data)



    global beta = l_exp_info[i_exp, :]
    ts = @view(exp_data[:, 1])
    tspan = [ts[1], ts[end]]
    
    p_for_crnn= p2vec(p_temp)
    sol = solve(
        prob,
        alg,
        tspan = tspan,
        p = p_for_crnn,
        saveat = ts,
        maxiters = maxiters,
    )
    
    trunc_time=ts[1:length(sol.u)] #so that if maxiters is reached, the rest of the code can still run
    heat_rel= HRR_getter(trunc_time, sol[:, :], p_temp) #compute heat release from species profiles

    return heat_rel, trunc_time, sol
end



function dlnprob(p, i_exp) #to evaluate gradient of loss_neuralode
    size_curr=size(l_exp_data[i_exp])[1]
    loss=zeros(size(p)[1])
    grad=zeros(size(p)[1],size(p)[2])

    for j = 1:size(p)[1]
        p_temp=deepcopy(p[j, :])

        loss_curr=loss_neuralode(p_temp, i_exp)

        grad_curr = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p_temp) #first entry is loss, other three are just reporting
    
        grad_curr[1]/=(Normalizer[i_exp, 1]^2) 
        grad_curr[2]/=(Normalizer[i_exp, 2]^2)
        grad_curr[3]/=(Normalizer[i_exp, 3]^2)
        grad_curr[4]/=Normalizer[i_exp, 1]^2
        grad_curr[5]/=Normalizer[i_exp, 2]^2
        grad_curr[6]/=Normalizer[i_exp, 3]^2
        grad_curr[7]/=Normalizer[i_exp, 1]^2
        grad_curr[8]/=Normalizer[i_exp, 2]^2
        grad_curr[9]/=Normalizer[i_exp, 3]^2
        grad_curr[10]/=(Normalizer[i_exp, 1]^2) 
        grad_curr[11]/=(Normalizer[i_exp, 2]^2)
        grad_curr[12]/=(Normalizer[i_exp, 3]^2)
        grad_curr[13]/=Normalizer[i_exp, 1]^2
        grad_curr[14]/=Normalizer[i_exp, 2]^2
        grad_curr[15]/=Normalizer[i_exp, 3]^2
        grad_curr[16]/=Normalizer[i_exp, 2]^2
        grad_curr[17]/=Normalizer[i_exp, 3]^2


        loss[j]=loss_curr
        grad[j, :]=grad_curr
    end
    loss=sum(loss)./size(p)[1] #normalize the saved loss by # of particles, note that gradient is still un-normalized to match svgd_kernel


    return loss, -grad
end

function loss_neuralode(p_temp, i_exp)
    exp_data = l_exp_data[i_exp]
    pred, trunc_time = (pred_n_ode(p_temp, i_exp, exp_data)) #1 grabs HRR. 2 is times, for use elsewhere.
    num_datasets=size(exp_data)[2]-1
    loss=sum(((abs.(pred.-exp_data[1:length(trunc_time), 2:end]).^2)))/num_datasets/size(exp_data)[1]

    #in the uploaded case, we do not have a prior.
    #in the case at this linked CEJ paper, we do have a prior for certain runs: https://doi.org/10.1016/j.cej.2025.160402
    #in that case, we would account for the prior loss term with pseudocode like this:
    ##prior_loss=-logpdf(prior_distribution, current_parameters)
    ##loss+=prior_loss

    return loss
end


