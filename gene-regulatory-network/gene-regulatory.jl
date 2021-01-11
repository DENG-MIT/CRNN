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

###################################
# Argments
is_restart = true;
n_epoch = 1000000;
n_plot = 10;
opt = ADAMW(0.001, (0.9, 0.999), 1.f-6);
datasize = 40;
tstep = 0.1;
n_exp_train = 70;
n_exp_val = 30;
n_exp = n_exp_train + n_exp_val;
noise = 1.f-2;
ns = 9;
nr = 15;
alg = Tsit5();
atol = 1e-5;
rtol = 1e-2;

np = nr * (2 * ns + 1)
p = (rand(Float32, np) .- 0.5) * 2 * sqrt(6 / (ns + nr))

lb = 1.f-5;
ub = 1.f2;
####################################

function p2vec(p)
    w_b = @view p[1:nr]
    w_in = reshape(@view(p[nr * (ns + 1) + 1:nr * (2 * ns + 1)]), ns, nr)

    w_out = reshape(@view(p[nr + 1:nr * (ns + 1)]), ns, nr)
    w_out[[1, 4, 7], :] .= 0

    w_out = @. -w_in * abs(w_out);
    w_in = clamp.(w_in, 0.f0, 4.f0);

    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p)
    println("species (column) reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=3))

    println("\nw_b")
    show(stdout, "text/plain", round.(exp.(w_b'), digits=3))

    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end
display_p(p)

function display_grad(grad)

    grad_w_out = reshape(grad[nr * (ns + 1) + 1:nr * (2 * ns + 1)], ns, nr);
    
    println("\ngrad w_out")
    show(stdout, "text/plain", round.(grad_w_out', digits=6))
    println("\n")

end

function trueODEfunc(dydt, y, k, t)
    #= Species:
        1: DNA_A
        2: mRNA_A
        3: A
        4: DNA_B
        5: mRNA_B
        6: B
        7: DNA_C
        8: mRNA_C
        9: C
        
    Reactions: i = [1,2,3]
        i+0: DNA_i -> DNA_i + mRNA_i - Transcription of mRNA
        i+1: mRNA_i -> mRNA_i + i - Translation of proteins
        i+2: mRNA_i -> 0 - mRNA decay
        i+3: i -> 0 - Dacay of proteins

        i+12: mRNA_i + (i+1) -> (i+1) Cyclic - Regulation of mRNA =#

    R = zeros(15)

    R[1] = k[1] * y[1]
    R[2] = k[2] * y[2]
    R[3] = k[3] * y[2]
    R[4] = k[4] * y[3]

    R[5] = k[5] * y[4]
    R[6] = k[6] * y[5]
    R[7] = k[7] * y[5]
    R[8] = k[8] * y[6]

    R[9] = k[9] * y[7]
    R[10] = k[10] * y[8]
    R[11] = k[11] * y[8]
    R[12] = k[12] * y[9]
    
    R[13] = k[13] * y[8] * y[3]
    R[14] = k[14] * y[5] * y[9]
    R[15] = k[15] * y[2] * y[6]

    dydt[1] = 0
    dydt[4] = 0
    dydt[7] = 0

    dydt[2] = R[1] - R[3] - R[15]
    dydt[5] = R[5] - R[7] - R[14]
    dydt[8] = R[9] - R[11] - R[13]

    dydt[3] = R[2] - R[4]
    dydt[6] = R[6] - R[8]
    dydt[9] = R[10] - R[12]
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
# @. u0_list[:, [1, 4, 7]] .= 1.f0;
# @. u0_list[:, [3, 5, 8]] .*= 0.f0;

tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
k = [1.8, 2.1, 1.3, 1.5, 2.2, 2, 2, 2.5, 3.2, 3, 2.3, 2.5, 6, 4, 3];
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];

u0 = u0_list[1, :];
prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

for i = 1:n_exp
    u0 = u0_list[i, :]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))

    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    
    push!(std_list, max_min(ode_data))
end

y_std = maximum(hcat(std_list...), dims=2) .+ lb;
show(stdout, "text/plain", round.(y_std', digits=8))

function crnn(du, u, p, t)
    w_in_x = w_in' * @. log(clamp(u, lb, ub))
    du .= w_out * (@. exp(w_in_x + w_b))
end

u0 = u0_list[1, :]
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol)

# sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
sense = ForwardDiffSensitivity(convert_tspan=false)
batch = datasize
function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p)
    prob_ = remake(prob, u0=u0, p=p)
    pred = clamp.(Array(solve(prob_, alg, saveat=tsteps[1:batch], sensalg=sense)), lb, ub)
    return pred
end
predict_neuralode(u0, p)

function loss_neuralode(p, i_exp)
    pred = predict_neuralode(u0_list[i_exp, :], p)
    ode_data = clamp.(ode_data_list[i_exp, :, 1:batch], lb, ub)
    loss = mae(ode_data, pred)
    return loss
end

# Callback function to observe training
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    list_plt = []
    for i = 1:ns
        plt = scatter(
            tsteps,
            ode_data[i, :],
            markercolor=:transparent,
            title=string(i),
            label=string("data_", i),
        )
        plot!(plt, tsteps, pred[i, :], label=string("pred_", i))
        push!(list_plt, plt)
    end
    plt_all = plot(list_plt..., legend=false)
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

list_loss_train = []
list_loss_val = []
list_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global list_loss_train, list_loss_val, list_grad, iter
    push!(list_loss_train, loss_train)
    push!(list_loss_val, loss_val)
    push!(list_grad, g_norm)

    if iter % n_plot == 0
        display_p(p)

        @printf("min loss train %.4e val %.4e\n", minimum(list_loss_train), minimum(list_loss_val))

        list_exp = randperm(5)[1:1]
        println("update plot for ", list_exp)
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss_train, xscale=:log10, yscale=:log10, framestyle=:box, label="Training");
        plot!(plt_loss, list_loss_val, label="Validation");
        plot!(xlabel="Epoch", ylabel="Loss")
        png(plt_loss, "figs/loss");

        plt_grad = plot(list_grad, xscale=:log10, yscale=:log10, label="grad_norm")
        png(plt_grad, "figs/grad")

        @save "./checkpoint/mymodel.bson" p opt list_loss_train list_loss_val list_grad iter
    end
    iter += 1
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss_train list_loss_val list_grad iter
    iter += 1
    # opt = ADAMW(0.0001, (0.9, 0.999), 1.f-8);
end

i_exp = 1;
epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        global batch = rand(2:datasize)
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p);
        grad_norm[i_exp] = norm(grad, 2)
        update!(opt, p, grad)
    end
    batch = datasize
    for i_exp in 1:n_exp_train + n_exp_val
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.4e val %.4e", loss_train, loss_val)))
    cb(p, loss_train, loss_val, mean(grad_norm));
end