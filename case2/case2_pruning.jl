using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load

Random.seed!(1234);

###################################
# Argments
is_restart = true;
p_cutoff = 0.0;
n_epoch = 100000;
n_plot = 100;
opt = ADAMW(1.f-3, (0.9, 0.999), 1.f-6); 
datasize = 50;
tstep = 1;
n_exp_train = 20;
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;
noise = 0.05;
ns = 6;
nr = 3;
alg = Tsit5();
atol = 1e-5;
rtol = 1e-2;

p = randn(Float32, nr * (ns + 2) + 1) .* 1.f-1;
p[1:nr] .+= 0.8;
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;
p[nr * (ns + 2) + 1] = 0.1;

lambda = 0.1
alpha = ones(Float32, ns);

lb = 1.f-5;
ub = 1.f1;
####################################

function trueODEfunc(dydt, y, k, t)
    # TG(1),ROH(2),DG(3),MG(4),GL(5),R'CO2R(6)
    r1 = k[1] * y[1] * y[2];
    r2 = k[2] * y[3] * y[2];
    r3 = k[3] * y[4] * y[2];
    dydt[1] = - r1;  # TG
    dydt[2] = - r1 - r2 - r3;  # TG
    dydt[3] = r1 - r2;  # DG
    dydt[4] = r2 - r3;  # MG
    dydt[5] = r3;  # GL
    dydt[6] = r1 + r2 + r3;  # R'CO2R
    dydt[7] = 0.f0;
end

logA = Float32[18.60f0, 19.13f0, 7.93f0];
Ea = Float32[14.54f0, 14.42f0, 6.47f0];  # kcal/mol

function Arrhenius(logA, Ea, T)
    R = 1.98720425864083f-3;
    k = exp.(logA) .* exp.(-Ea ./ R ./ T);
    return k;
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns + 1));
u0_list[:, 1:2] = u0_list[:, 1:2] .* 2.0 .+ 0.2;
u0_list[:, 3:ns] .= 0.0;
u0_list[:, ns + 1] = u0_list[:, ns + 1] .* 20.0 .+ 323.0;  # T[K]
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];

# u0 = u0_list[1, :];
# k = Arrhenius(logA, Ea, u0[end]);
# prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
# ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

for i in 1:n_exp
    u0 = u0_list[i, :];
    k = Arrhenius(logA, Ea, u0[end]);
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))[1:end - 1, :];
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end

y_std = maximum(hcat(std_list...), dims=2);
# y_std[3:4] ./= 2.f0;

function p2vec(p)
    slope = p[nr * (ns + 2) + 1] .* 1.f2;
    w_b = p[1:nr] .* slope;
    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr);

    # w_out[3, 1] = -1

    index = findall(abs.(w_out) .< p_cutoff);
    w_out[index] .= 0;

    # w_out = clamp.(w_out, -2.5, 2.5);
    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope);
    # w_in_Ea = clamp.(w_in_Ea, 0.f0, 30.f0);

    w_in = clamp.(-w_out, 0.f0, 4.f0);
    w_in = vcat(w_in, w_in_Ea');
    
    return w_in, w_b, w_out
end

w_in, w_b, w_out = p2vec(p);

R = 1.98720425864083f-3;
function crnn(du, u, p, t)
    logX = @. log(clamp(u[1:end - 1], lb, ub));
    w_in_x = w_in' * vcat(logX, -1.f0 / R / u[end]);
    du .= vcat(w_out * (@. exp(w_in_x + w_b)), 0.f0);
end

u0 = u0_list[1, :];
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, 
                  atol=atol, rtol=rtol)
# @benchmark sol = solve(prob, alg, u0=u0, p=p)

function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p);
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p)), -ub, ub)
    return pred
end
# @benchmark predict_neuralode(u0, p)
predict_neuralode(u0, p)

function display_p(p)
    w_in, w_b, w_out = p2vec(p);

    println("species (column) reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=3))

    println("\nw_b")
    show(stdout, "text/plain", round.(w_b', digits=3))

    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))



    println("\n\n")
end
display_p(p)

i_obs = [1, 2, 3, 4, 5, 6];
function loss_neuralode(p, i_exp)
    ode_data = @view ode_data_list[i_exp, i_obs, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)[i_obs, :]
    loss = mae(ode_data ./ y_std[i_obs], pred ./ y_std[i_obs])
    # loss = mean(abs.(ode_data .- pred) ./ y_std[i_obs], dims=2)
    return loss
end
# loss_neuralode(p, 1);

species = ["TG","ROH","DG","MG","GL","R'CO2R"];
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)

    @save "./checkpoint/cbi27_m3.bson" ode_data pred tsteps;

    list_plt = []
    for i in 1:ns
        plt = scatter(tsteps, ode_data[i,:], 
                      markercolor=:transparent,
                      label="Exp",
                      framestyle=:box)
        plot!(plt, tsteps, pred[i,:], label="CRNN")
        plot!(xlabel="Time (minute)", ylabel="["*species[i]*"]")

        if i == 1
            plot!(plt, legend=true, framealpha=0)
        else
            plot!(plt, legend=false)
        end

        push!(list_plt, plt)
    end
    plt_all = plot(list_plt...)
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

list_loss_train = []
list_loss_val = []
iter = 1
cb = function (p, loss_train, loss_val)
    global list_loss_train, list_loss_val, iter
    push!(list_loss_train, loss_train)
    push!(list_loss_val, loss_val)

    if iter % n_plot == 0
        display_p(p)

        @printf("min loss train %.4e val %.4e\n", minimum(list_loss_train), minimum(list_loss_val))

        list_exp = randperm(n_exp)[1:1];
        println("update plot for ", list_exp)
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss_train, xscale=:log10, yscale=:log10, framestyle=:box, label="Training");
        plot!(plt_loss, list_loss_val, label="Validation");
        plot!(xlabel="Epoch", ylabel="Loss")
        plot!(size=(400, 350))
        png(plt_loss, "figs/loss");

        # @save "./checkpoint/mymodel.bson" p opt list_loss_train list_loss_val iter;
    end

    iter += 1;
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss_train list_loss_val iter;
    iter += 1;
end

p_cutoff = 0.01

i_exp = 1
epochs = ProgressBar(iter:n_epoch)
loss_epoch = zeros(Float32, n_exp);

for i_exp in 1:n_exp
    loss_epoch[i_exp] = loss_neuralode(p, i_exp)
end
loss_train = mean(loss_epoch[1:n_exp_train]);
loss_val = mean(loss_epoch[n_exp_train + 1:end]);
# cb(p, loss_train, loss_val);
cbi(p, 27);

display_p(p)

pw = vcat(w_in, w_b', w_out)'
using DelimitedFiles
writedlm( "weights.csv",  pw, ',')

@printf("clip loss p_cutoff train val [%.1e, %.4e, %.4e]\n", p_cutoff, loss_train, loss_val)


# @printf("Fine tunning")

# opt = ADAMW(1.e-4, (0.9, 0.999), 1.f-10);

# for epoch in epochs
#     global p
#     for i_exp in randperm(n_exp_train)
#         grad = gradient(p) do x
#             Zygote.forwarddiff(x) do x
#                 loss_neuralode(x, i_exp)
#             end
#         end
#         update!(opt, p, grad[1])
#     end
#     for i_exp in 1:n_exp
#         loss_epoch[i_exp] = loss_neuralode(p, i_exp)
#     end
#     loss_train = mean(loss_epoch[1:n_exp_train]);
#     loss_val = mean(loss_epoch[n_exp_train + 1:end]);
#     set_description(epochs, string(@sprintf("Loss train %.4e val %.4e", loss_train, loss_val)))
#     cb(p, loss_train, loss_val);
# end

# @printf("fine tunning loss p_cutoff train val [%.1e, %.4e, %.4e]\n", p_cutoff, loss_train, loss_val)
