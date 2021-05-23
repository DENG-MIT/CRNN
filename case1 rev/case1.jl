using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load
using Catalyst

Random.seed!(1234);

# TODO: use YAML file for configuration
# Argments
is_restart = false;
p_cutoff = 0.0;
n_epoch = 1000000;
n_plot = 100;
opt = ADAMW(0.0001, (0.9, 0.999), 0.0);
datasize = 100;
tstep = 0.1;
n_exp_train = 20;
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;
noise = 1.f-3;
ns = 5;
nr = 10;
k = [];
alg = Tsit5();
atol = 1e-5;
rtol = 1e-2;

maxiters = 10000;

lb = 1.f-5;

rn = @reaction_network begin
    (1.0, 1.0), A ↔ B
    (1.0, 1.0), B ↔ C
    (1.0, 1.0), C ↔ D
    # (1.0, 1.0), 2A ↔ B + C
    # (1.0, 1.0), 2B ↔ C + D
    (1.0, 1.0), 2C ↔ D + E
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
u0_list[:, 1:2] .+= 0.2;
# u0_list[:, 3:end] .= 0.0;
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

for i in 1:n_exp
    u0 = u0_list[i, :];
    prob_trueode = ODEProblem(rn, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end
y_std = maximum(hcat(std_list...), dims=2);


p = randn(Float32, nr * (ns + 1)) .* 0.5;

function p2vec(p)
    w_kf = p[1:nr];
    w_kb = w_kf;  # assume Kc = 1
    w_out = reshape(p[nr + 1:end], ns, nr);
    w_out = clamp.(w_out, -2.5, 2.5)
    return w_kf, w_kb, w_out
end

function crnn!(du, u, p, t)
    w_in_f = clamp.(-w_out, 0, 2.5);
    w_in_b = clamp.(w_out, 0, 2.5);

    u_in = @. log(clamp(u, lb, Inf))
    w_in_x_f = w_in_f' * u_in;
    w_in_x_b = w_in_b' * u_in;
    
    du .= w_out * @. (exp(w_in_x_f + w_kf) - exp(w_in_x_b + w_kb));
end

u0 = u0_list[1, :]
prob = ODEProblem(crnn!, u0, tspan, saveat=tsteps,
                  atol=atol, rtol=rtol)

function pred_ode(u0, p)
    global w_kf, w_kb, w_out
    w_kf, w_kb, w_out = p2vec(p)
    pred = Array(solve(prob, alg, u0=u0, p=p; maxiters=maxiters))
    return pred
end
pred_ode(u0, p);

function display_p(p)
    local w_kf, w_kb, w_out
    w_kf, w_kb, w_out = p2vec(p);

    println("species (column) reaction (row)")

    println("\n w_out | w_kf | w_kb")
    println(rn.states)
    show(stdout, "text/plain", round.(vcat(w_out, w_kf', w_kb')', digits=3))

    println("\n")
end
display_p(p)

function loss_ode(p, i_exp)
    pred = pred_ode(u0_list[i_exp, :], p)
    loss = mae(ode_data_list[i_exp, :, :] ./ y_std, pred ./ y_std)
    return loss
end
loss_ode(p, 1)

# Callback function to observe training

species = rn.states;
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = pred_ode(u0_list[i_exp, :], p)
    l_plt = []
    for i in 1:ns
        plt = Plots.scatter(tsteps, ode_data[i,:], 
                      markercolor=:transparent,
                      label="Exp",
                      framestyle=:box)
        plot!(plt, tsteps, pred[i,:], lw=3, label="CRNN")
        plot!(xlabel="Time", ylabel=species[i])

        if i == 1
            plot!(plt, legend=true, framealpha=0)
        else
        plot!(plt, legend=false)
        end

        push!(l_plt, plt)
    end
    plt_all = plot(l_plt...)
    png(plt_all, string("figs/i_exp_", i_exp))

    println("plot for exp $i_exp is updated \n")
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

        plt_loss = plot(l_loss_train, xscale=:log10, yscale=:log10, label="train");
        plot!(plt_loss, l_loss_val, label="val");

        png(plt_loss, "figs/loss");

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter
    end

iter += 1;
end
    
if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    iter += 1;
end


i_exp = 1
epochs = ProgressBar(iter:n_epoch)
loss_epoch = zeros(Float32, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)
        grad = ForwardDiff.gradient(x -> loss_ode(x, i_exp), p)
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_ode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.4e val %.4e", loss_train, loss_val)))
    cb(p, loss_train, loss_val);
end
