using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DiffEqSensitivity

Random.seed!(1234);

# Argments
is_restart = false;
n_epoch = 1000000;
n_plot = 1;

n_exp_train = 20;
n_exp_val = 10;
n_exp = n_exp_train + n_exp_val;

noise = 1.f-3;

ntotal = 300;
batch_min = 32;
tstep = 5 / ntotal;

ns = 7;
ns_ = 12;
nr = 12;

alg = AutoTsit5(TRBDF2(autodiff=false));
atol = 1e-5;
rtol = 1e-2;
lb = atol;
ub = 1.f2;

opt = Flux.Optimiser(ExpDecay(5e-3, 0.5, 100 * n_exp_train, 1e-5),
                     ADAMW(0.005, (0.9, 0.999), 1.f-6));

q = 4
K1 = 0.52
A = 4
N = 1
J0 = 2.5
phi = 0.1

function trueODEfunc(dsdt, s, k, t)
    r1 = k[1] * s[1] * s[6] / (1 + (s[6] / K1)^q)
    r2 = k[2] * s[2] * (N - s[5])
    r3 = k[3] * s[3] * (A - s[6])
    r4 = k[4] * s[4] * s[5]
    r5 = k[5] * s[6]
    r6 = k[6] * s[2] * s[5]
    r7 = 13 * s[7]
    r8 = 13 * (s[4] - s[7])

    dsdt[1] = J0 - r1
    dsdt[2] = 2 * r1 - r2 - r6
    dsdt[3] = r2 - r3
    dsdt[4] = r3 - r4 - r8
    dsdt[5] = r2 - r4 - r6
    dsdt[6] = -2 * r1 + 2 * r3 - r5
    dsdt[7] = phi * r8 - r7
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
ic_lb = [0.15, 1.19, 0.04, 0.10, 0.08, 0.14, 0.05];
ic_ub = [1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10];
for i in 1:ns
    u0_list[:, i] .= rand(n_exp) .* (ic_ub[i] - ic_lb[i]) .+ ic_lb[i]
end

tspan = Float32[0.0, ntotal * tstep];
tsteps = range(tspan[1], tspan[2], length=ntotal);
k = [100, 6, 16, 100, 1.28, 12];
ode_data_list = zeros(Float32, (n_exp, ns, ntotal));
yscale_list = [];

u0 = u0_list[1, :];
prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

for i = 1:n_exp
    u0 = u0_list[i, 1:ns]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))

    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    
    push!(yscale_list, std(ode_data, dims=2))
end

y_scale = maximum(hcat(yscale_list...), dims=2) .+ lb;
show(stdout, "text/plain", round.(y_scale', digits=8))

np = nr * (ns_ + 1) + ns + 1;
pcrnn = (rand(Float32, np) .- 0.5) * 2 * sqrt(6 / (ns_ + nr));
pcrnn[end] = 0.1

function p2vec(p)
    slope = p[end] * 100
    w_b = p[1:nr] .* slope
    w_out = reshape(@view(p[nr + 1:nr * (ns_ + 1)]), ns_, nr)
    w_in = clamp.(-w_out, 0.f0, 4.f0)
    w_J = p[nr * (ns_ + 1) + 1:np - 1]
    return w_in, w_b, w_out, w_J
end

function display_p(p)
    w_in, w_b, w_out, w_J = p2vec(p)
    println("\nspecies (column) reaction (row) w_out | w_b")
    w_in_ = vcat(w_out, exp.(w_b'), )
    show(stdout, "text/plain", round.(w_in_', digits=2))

    println("\nw_J")
    show(stdout, "text/plain", round.(w_J', digits=2))
end
display_p(pcrnn)

node = ns_ - ns;
dudt2 = Chain(x -> x,
              Dense(ns, node, gelu),
              Dense(node, node, gelu),
              Dense(node, node, gelu),
              Dense(node, ns_ - ns, softplus));
pnn, re = Flux.destructure(dudt2);
rep = re(pnn);
p = vcat(pcrnn, pnn);

function crnn(du, u, pcrnn, t)
    u_ = vcat(u, rep(u))
    w_in_x = w_in' * @. log(clamp(u_, lb, ub))
    du .= (w_out * (@. exp(w_in_x + w_b)))[1:ns] .+ w_J
end

i_exp = 1;
u0 = u0_list[i_exp, :];
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol)

sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
function predict_neuralode(u0, p, batch=ntotal)
    pcrnn = p[1:np]
    pnn = p[np + 1:end]
    global w_in, w_b, w_out, w_J = p2vec(pcrnn)
    global rep = re(pnn)
    prob_ = remake(prob, u0=u0, p=pcrnn)
    pred = clamp.(Array(solve(prob_, alg, saveat=tsteps[1:batch], sensalg=sense)), lb, ub)
    return pred
end
predict_neuralode(u0, p)

function loss_neuralode(p, i_exp, batch=ntotal)
    pred = predict_neuralode(u0_list[i_exp, :], p, batch)
    ode_data = clamp.(ode_data_list[i_exp, 1:ns, 1:size(pred)[2]], lb, ub)
    loss = mae(ode_data ./ y_scale, pred[1:ns, :] ./ y_scale)
    return loss
end
loss_neuralode(p, 1)

# Callback function to observe training
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    rep = re(pnn)
    pred_ = rep(pred)
    l_plt = []
    for i = 1:ns
        plt = scatter(
            tsteps,
            ode_data[i, :],
            markercolor=:transparent
        )
        plot!(plt, tsteps, pred[i, :], lw=3)
        xlabel!(plt, "Time (minutes)")
        ylabel!(plt, "S$i (mM)")
        push!(l_plt, plt)
    end
    for i = 1:ns_ - ns
        plt = plot(tsteps, pred_[i, :], lw=3)
        xlabel!(plt, "Time (minutes)")
        ylabel!(plt, "vS_$i (mM)")
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false, size=(800, 800))
    png(plt_all, string("figs/i_exp_", i_exp))

    return false
end
cbi(p, 1)

l_loss_train = []
l_loss_val = []
l_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global l_loss_train, l_loss_val, l_grad, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(l_grad, g_norm)

    if iter % n_plot == 0
        display_p(p)
        l_exp = [1, n_exp]
        println("\n plot ", l_exp)
        for i_exp in l_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(l_loss_train, xscale=:identity, yscale=:log10, label="Training")
        plot!(plt_loss, l_loss_val, xscale=:identity, yscale=:log10, label="Validation")
        plt_grad = plot(l_grad, xscale=:identity, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        # ylims!(plt_loss, (-Inf, 1e0))
        plt_all = plot([plt_loss, plt_grad]..., legend=:top)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val l_grad iter
    end
    iter += 1
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val l_grad iter
    iter += 1
    # opt = ADAMW(0.0005, (0.9, 0.999), 1.f-6);
end

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)
        batch = rand(batch_min:ntotal)
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp, batch), p);
        # loss, back = Zygote.pullback(x -> loss_neuralode(x, i_exp, batch), p)
        # grad = back(one(loss))[1]
        grad_norm[i_exp] = norm(grad, 2)
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, 
                    string(@sprintf("Loss train %.4e val %.4e lr %.1e", loss_train, loss_val, opt[1].eta)))
    cb(p, loss_train, loss_val, mean(grad_norm));
end

@printf("min loss train %.4e val %.4e\n", minimum(l_loss_train), minimum(l_loss_val))