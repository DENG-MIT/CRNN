using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load

Random.seed!(1234);

###################################
# Argments
is_restart = false;
p_cutoff = 0.0;
n_epoch = 1000000;
n_plot = 20;
# opt = ADAMW(0.005, (0.9, 0.999), 1.f-5);
opt = NADAM(0.001, (0.9, 0.999));
datasize = 100;
tstep = 0.1;
n_exp_train = 70;
n_exp_test = 30;
n_exp = n_exp_train + n_exp_test;
noise = 5.f-2;
ns = 9;
nr = 8;
alg = Tsit5();
atol = 1e-5;
rtol = 1e-2;

# p = randn(Float32, nr * (2 * ns + 1) + 1) .* 1.f-1;
np = nr * (2 * ns + 1) + 1
p = (rand(Float32, np) .- 0.5) * 2 * sqrt(6 / (ns + nr))
p[end] = 0.1

lb = 1.f-5;
ub = 1.f2;
####################################

function p2vec(p)
    w_b = @view p[1:nr]
    slope = abs(p[end])
    w_in = reshape(@view(p[nr * (ns + 1) + 1:nr * (2 * ns + 1)]), ns, nr)  # .* (1.f1 * slope)

    w_out = reshape(@view(p[nr + 1:nr * (ns + 1)]), ns, nr)
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

    # println("\nw_out")
    # show(stdout, "text/plain", round.(w_out', digits=3))

    println("\nw_out_scale")
    show(stdout, "text/plain", round.(w_out' .* dy_std_' .* exp.(w_b), digits=3))
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
    # ["S"(1), "MAPKKK"(2), "MAPKKK*"(3), "MAPKK"(4), "MAPKK*"(5),
    # "MAPK"(6), "MAPK*"(7), "TF"(8), "TF*"(9)]
    r1 = k[1] * y[1] * y[2]
    r2 = k[2] * y[3] * y[4]
    r3 = k[3] * y[5] * y[6]
    r4 = k[4] * y[7] * y[8]
    r5 = k[5] * y[3]
    r6 = k[6] * y[5]
    r7 = k[7] * y[7]
    r8 = k[8] * y[9]
    dydt[1] = 0
    dydt[2] = -r1 + r5
    dydt[3] = r1 - r5
    dydt[4] = -r2 + r6
    dydt[5] = r2 - r6
    dydt[6] = -r3 + r7
    dydt[7] = r3 - r7
    dydt[8] = -r4 + r8
    dydt[9] = r4 - r8
end

# Generate data sets
u0_list .= 10 .^ (rand(Float32, (n_exp, ns)) * -3);
@. u0_list[[1, 2, end], [3, 5, 7, 9]] .*= 0.f0;
# @. u0_list[:, [3, 5, 7, 9]] .*= 0.f0;
# u0_list = rand(Float32, (n_exp, ns));
# u0_list[:, [1]] .+= 0.01;
# @. u0_list[:, [1, 2, 4, 6, 8]] = 10^(u0_list[:, [1, 2, 4, 6, 8]] * -2);
# @. u0_list[:, [3, 5, 7, 9]] *= 0.0001;
# @. u0_list[:, [2, 4, 6, 8]] = u0_list[:, [2, 4, 6, 8]] * 0.9 + 0.01;
# u0_list .+= 1.f-3;

tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
k = ones(8);
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];
std_list_ = [];

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
    push!(std_list_, max_min(ode_data))
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data))
end

y_std = maximum(hcat(std_list...), dims=2);
dy_std_ = vec(maximum(hcat(std_list...), dims=2))
dy_std_ ./= tspan[2]

log_data =
    reshape(permutedims(log.(ode_data_list .+ lb), [2, 1, 3]), (ns, n_exp * datasize))

logy_mean = vec(mean(log_data, dims=2))
logy_std = vec(std(log_data, dims=2))

show(stdout, "text/plain", round.(y_std', digits=8))

mask = ones((ns, nr));
mask[1, :] .= 0.0f0;

w_in_ = zeros(ns)
w_in_[[1, 2]] .= 1
w_in_ = reshape(w_in_, (9, 1))

function crnn(du, u, p, t)
    w_in_x = w_in' * @. log(clamp(u, lb, ub))
    # w_in_x = w_in' * @. ((log(clamp(u, lb, ub)) - logy_mean) / logy_std);
    du .= w_out * (@. exp(w_in_x + w_b)) .* dy_std_
end

u0 = u0_list[1, :]
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol)
# sol = solve(prob, alg, u0=u0, p=p);

function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p)
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p)), lb, ub)
    return pred
end
predict_neuralode(u0, p);
# @benchmark predict_neuralode(u0, p)

# i_obs = [1];
i_obs = 1:ns;

function loss_neuralode(p, i_exp)
    pred = predict_neuralode(u0_list[i_exp, :], p)
    ode_data = clamp.(ode_data_list[i_exp, :, :], lb, ub)
    # loss = mean(abs.(ode_data_list[i_exp, i_obs, :] .- pred) ./ y_std[i_obs], dims=2)
    # loss = mae( ode_data_list[i_exp, :, :] ./y_std, pred ./y_std)
    loss = mae(log.(ode_data), log.(pred))
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

        list_exp = randperm(n_exp)[1:1]
        println("update plot for ", list_exp)
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss_train, xscale=:log10, yscale=:log10, framestyle=:box, label="Training");
        plot!(plt_loss, list_loss_val, label="Validation");
        plot!(xlabel="Epoch", ylabel="Loss")
        # plot!(size=(400, 350))
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
end

# l2_reg = 1.f-6;
# opt = ADAMW(0.001, (0.9, 0.999), 1.f-5);
# opt = NADAM(0.001, (0.9, 0.999));

i_exp = 1;
epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        # Zygote forward mode AD
        grad = gradient(p) do x
            Zygote.forwarddiff(x) do x
                w_in, w_b, w_out = p2vec(p)
                loss_neuralode(x, i_exp) # + norm(w_in, 2) * l2_reg
            end
        end
        grad = grad[1]
        grad_norm[i_exp] = norm(grad, 2)
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.4e val %.4e", loss_train, loss_val)))
    cb(p, loss_train, loss_val, mean(grad_norm));
end


# for i_exp in 1:n_exp_train
#     # Zygote forward mode AD
#     grad = gradient(p) do x
#         Zygote.forwarddiff(x) do x
#             loss_neuralode(x, i_exp)
#         end
#     end

#     grad = grad[1]

#     @show i_exp
#     display_grad(grad)
# end