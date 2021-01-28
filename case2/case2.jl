using OrdinaryDiffEq, Flux, Optim, Random, Plots
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load

Random.seed!(1234);

###################################
# Arguments
is_restart = true;
n_epoch = 10000;
n_plot = 50;
datasize = 50;
tstep = 1;
n_exp_train = 20;
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;
noise = 0.05;
ns = 6;
nr = 3;
alg = AutoTsit5(Rosenbrock23(autodiff=false));
atol = 1e-6;
rtol = 1e-3;

# opt = ADAMW(5.f-3, (0.9, 0.999), 1.f-6);
opt = Flux.Optimiser(ExpDecay(5e-3, 0.5, 500 * n_exp_train, 1e-4),
                     ADAMW(0.005, (0.9, 0.999), 1.f-6));

lb = 1.f-6;
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
    R = 1.98720425864083f-3
    k = exp.(logA) .* exp.(-Ea ./ R ./ T)
    return k
end

# Generate datasets
u0_list = rand(Float32, (n_exp, ns + 1));
u0_list[:, 1:2] = u0_list[:, 1:2] .* 2.0 .+ 0.2;
u0_list[:, 3:ns] .= 0.0;
u0_list[:, ns + 1] = u0_list[:, ns + 1] .* 20.0 .+ 323.0;  # T[K]
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);

ode_data_list = zeros(Float32, (n_exp, ns, datasize));
yscale_list = [];
function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end
for i in 1:n_exp
    u0 = u0_list[i, :]
    k = Arrhenius(logA, Ea, u0[end])
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))[1:end - 1, :]
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(yscale_list, max_min(ode_data))
end
yscale = maximum(hcat(yscale_list...), dims=2);

np = nr * (ns + 2) + 1;
p = randn(Float32, np) .* 0.1;
p[1:nr] .+= 0.8;
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;
p[end] = 0.1;

function p2vec(p)
    slope = p[nr * (ns + 2) + 1] .* 100
    w_b = p[1:nr] .* slope
    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr)
    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope)
    w_in = clamp.(-w_out, 0, 4)
    w_in = vcat(w_in, w_in_Ea')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("species (column) reaction (row)")
    println("w_in | w_b")
    w_in_ = vcat(w_in, w_b')'
    show(stdout, "text/plain", round.(w_in_, digits=3))
    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end
display_p(p)

inv_R = - 1 / 1.98720425864083f-3;
function crnn(du, u, p, t)
    logX = @. log(clamp(u[1:end - 1], lb, ub))
    w_in_x = w_in' * vcat(logX, inv_R / u[end])
    du .= vcat(w_out * (@. exp(w_in_x + w_b)), 0.f0)
end

u0 = u0_list[1, :];
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol)

sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p)
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p, sensalg=sense)), -ub, ub)
    return pred
end
predict_neuralode(u0, p)

i_obs = [1, 2, 3, 4, 5, 6];
function loss_neuralode(p, i_exp)
    ode_data = @view ode_data_list[i_exp, i_obs, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)[i_obs, :]
    loss = mae(ode_data ./ yscale[i_obs], pred ./ yscale[i_obs])
    return loss
end

cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    l_plt = []
    for i in 1:ns
        plt = scatter(tsteps, ode_data[i,:], markercolor=:transparent,
                      title=string(i), label=string("data_", i))
        plot!(plt, tsteps, pred[i,:], label=string("pred_", i))
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false)
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

l_loss_train = []
l_loss_val = []
iter = 1
cb = function (p, loss_train, loss_val)
    global l_loss_train, l_loss_val, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)

    if iter % n_plot == 0
        display_p(p)
        @printf("min loss train %.4e val %.4e\n", minimum(l_loss_train), minimum(l_loss_val))

        l_exp = randperm(n_exp)[1:1];
        println("update plot for ", l_exp)
        for i_exp in l_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(l_loss_train, xscale=:log10, yscale=:log10, 
                        framestyle=:box, label="Training")
        plot!(plt_loss, l_loss_val, label="Validation")
        plot!(xlabel="Epoch", ylabel="Loss")
        png(plt_loss, "figs/loss")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    end
    iter += 1;
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    iter += 1;
end

i_exp = 1
epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p)
        grad_norm[i_exp] = norm(grad, 2)
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.2e val %.2e gnorm %.1e lr %.1e", 
                                             loss_train, loss_val, mean(grad_norm), opt[1].eta)))
    cb(p, loss_train, loss_val);
end

for i_exp in 1:n_exp
    cbi(p, i_exp)
end
