using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using DiffEqSensitivity
using Zygote
using ForwardDiff
using Interpolations
using LinearAlgebra
using Random
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DelimitedFiles

is_restart = true;
n_epoch = 100000;
ntotal = 40;
batch_size = 32;
n_plot = 100;
grad_max = 1.e1;
maxiters = 10000;

nr = 10;
opt = ADAMW(0.005, (0.9, 0.999), 1.f-6);

atol = 1.e-8
rtol = 1.e-3
lb = atol;
ode_solver = AutoTsit5(Rosenbrock23(autodiff=false));
# ode_solver = Rosenbrock23(autodiff=false);

rawdata = readdlm("data/10atm_1300K_0.01.txt")';
tsteps = rawdata[1, :];
Tlist = rawdata[2, :];
Plist = rawdata[3, :];
ylabel = rawdata[4:end, :];
t_end = tsteps[end]
tspan = (0.0, tsteps[end]);
ns = size(ylabel)[1];

_tsteps = 10 .^ range(log10(t_end/100), log10(t_end/1.01), length=ntotal);
_tsteps[1] = 0;

itp = LinearInterpolation(tsteps, Tlist);
Tlist = itp.(_tsteps);
itp = LinearInterpolation(tsteps, Plist);
Plist = itp.(_tsteps);
_ylabel = zeros(ns, ntotal);
for i in 1:ns
    itp = LinearInterpolation(tsteps, ylabel[i, :])
    _ylabel[i, :] .= itp.(_tsteps)
end
tsteps = _tsteps;
ylabel = _ylabel
batch_size = minimum([batch_size, ntotal]);

varnames = ["C10H16", "H2", "CH4", "C2H2", "C2H4", "N2", "C4H81",  "H", "CH3"];
l_MW = [136.238, 2.016,  16.043,  26.038,  28.054,  28.014,  56.108, 1.008,  15.035][1:ns];  # kg/mol

E_C = [10, 0, 1, 2, 2, 0, 4, 0, 1];
E_H = [16, 2, 4, 2, 4, 0, 8, 1, 3];
E_N = [0, 0, 0, 0, 0, 2, 0, 0, 0];
E_ = hcat(E_C, E_H, E_N)[1:ns, :];
E_null = nullspace(E_')';
n_null = size(E_null)[1];

ymax = maximum(ylabel, dims=2);
ymin = minimum(ylabel, dims=2);

yscale = clamp.(ymax - ymin, lb, Inf);
normdata = ylabel
u0 = normdata[:, 1];

np = nr * (2 * ns + 3) + 1;
p = randn(Float64, np) .* 0.1;
p[end] = 0.1;  # slope

function p2vec(p)
    slope = p[end] .* 10.0
    w_b = p[1:nr] .* slope
    w_in_b = p[nr + 1:nr * 2]
    w_in_Ea = p[nr * 2 + 1:nr * 3] .* slope

    w_out = reshape(p[nr * 3 + 1:nr * (ns + 3)], ns, nr) 
    w_in = reshape(p[nr * (ns + 3) + 1:nr * (ns * 2 + 3)], ns, nr);
    # w_out = (w_out * E_null)'
    w_out = @. -w_in * (10 ^ w_out)
    w_in = vcat(clamp.(w_in, 0.0, 2.5), w_in_Ea', w_in_b')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p)
    println("\nspecies (column) reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=2))
    println("\nw_out | log(A)")
    show(stdout, "text/plain", round.(hcat(w_out', w_b), digits=2))
    println("\n")
end
display_p(p);

itpT = LinearInterpolation(tsteps, Tlist);
itpP = LinearInterpolation(tsteps, Plist);

R = 1.98720425864083f-3; # kcal⋅K−1⋅mol−1
function Y2density(Y, P, T)
    return P / (8.31446261815324e3 * T * sum(Y ./ l_MW))
end
density = Y2density(u0, Plist[1], Tlist[1])

function Y2C(Y, density)
    return density * (Y ./ l_MW) * 1e3
end
Y2C(u0, density)

Clabel = Y2C(ylabel, density)
Cmax = maximum(Clabel, dims=2)[:, 1]

dydt_scale = (yscale[:, 1] / t_end)
function crnn!(du, u, p, t)
    P = itpP(t)
    T = itpT(t)
    Y = clamp.(u, lb, 10)
    density = Y2density(Y, P, T)
    C = Y2C(Y, density)
    logX = @. log(clamp(C, lb, 10))
    w_in_x = w_in' * vcat(logX, - 1 / R / T, log(T))
    wdot = w_out * (@. exp(w_in_x + w_b))
    @. du = wdot * l_MW / density * dydt_scale
end

prob = ODEProblem(crnn!, u0, tspan)
sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
function predict_n_ode(p, sample=ntotal)
    global w_in, w_b, w_out = p2vec(p)
    _prob = remake(prob, p=p, tspan=[0, tsteps[sample]])
    pred = Array(solve(_prob, ode_solver, saveat=tsteps[1:sample],
                 atol=atol, rtol=rtol, sensalg=sense, maxiters=maxiters))
end
predict_n_ode(p)

function loss_n_ode(p, sample=ntotal)
    pred = predict_n_ode(p, sample)
    loss = mae(pred ./ yscale, normdata[:, 1:size(pred)[2]] ./ yscale)
    return loss
end
loss_n_ode(p)

list_loss = []
list_grad = []
iter = 1
cb = function (p, loss_mean, g_norm; doplot=true)
    global list_loss, list_grad, iter
    push!(list_loss, loss_mean)
    push!(list_grad, g_norm)

    if doplot & (iter % n_plot == 0)
        display_p(p)
        pred = predict_n_ode(p)

        list_plt = []
        for i in 1:ns
            plt = scatter(tsteps[2:end], normdata[i, 2:end], xscale=:log10, label="data")
            plot!(plt, tsteps[2:end], pred[i,2:end], label="pred")
            ylabel!(plt, "$(varnames[i])")
            xlabel!(plt, "Time [s]")
            push!(list_plt, plt)
        end
        plt_all = plot(list_plt..., legend=:false, size=(1000, 600))
        png(plt_all, "figs/pred.png")

        println("update plot")

        plt_loss = plot(list_loss, yscale=:log10, label="loss")
        plt_grad = plot(list_grad, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Grad Norm")
        plt_all = plot([plt_loss, plt_grad]..., legend=:outertopright)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    end
    iter += 1
    return false
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    iter += 1
    # opt = ADAMW(0.0005, (0.9, 0.999), 1.f-6)
end

epochs = ProgressBar(iter:n_epoch);
for epoch in epochs
    global p
    sample = rand(batch_size:ntotal)
    loss = loss_n_ode(p, sample)
    grad = ForwardDiff.gradient(x -> loss_n_ode(x, sample), p);
    grad_norm = norm(grad, 2)

    if grad_norm > grad_max
        grad = grad ./ grad_norm .* grad_max
    end

    update!(opt, p, grad)
    
    set_description(epochs, string(@sprintf("Loss: %.4e grad: %.2e", loss, grad_norm)))
    cb(p, loss, grad_norm)
end