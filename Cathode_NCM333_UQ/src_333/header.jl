using Random, Plots
using ForwardDiff
#using Zygote
using OrdinaryDiffEq
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DelimitedFiles
using YAML
using Distributions
using EllipsisNotation
using Distances
using CurveFit
using StatsBase
using ColorSchemes
using BenchmarkTools



ENV["GKSwstype"] = "100"

cd(dirname(@__DIR__))
conf = YAML.load_file("./config.yaml")

expr_name ="NCM333"
fig_path = string("./results/", expr_name, "/figs")
ckpt_path = string("./results/", expr_name, "/checkpoint")
ckpt_path_det = string("./results/", expr_name, "/checkpoint_det_load")
ckpt_path_plot = string("./results/", expr_name, "/checkpoint_save_plotting")
config_path = "./results/$expr_name/config.yaml"
noise_level = Float64(conf["noise_level_noisy_wang"])


is_restart = Bool(conf["is_restart"])
ns = Int64(conf["ns"])
nr = Int64(conf["nr"])
lb_clamp = Float64(conf["lb_clamp"])
lb_abstol = Float64(conf["lb_abstol"])
n_epoch = Int64(conf["n_epoch"])
n_plot = Int64(conf["n_plot"])
grad_max = Float64(conf["grad_max"])
maxiters = Int64(conf["maxiters"])

lr_max = Float64(conf["lr_max"])
lr_min = Float64(conf["lr_min"])
lr_adam = Float64(conf["adam_lr"])
lr_decay = Float64(conf["lr_decay"])
lr_decay_step = Int64(conf["lr_decay_step"])
w_decay = Float64(conf["w_decay"])
global cathode_num=Int64(conf["cathode"])

num_particles=Int64(conf["num_particles"])
n_iter=Int64(conf["n_iter"])
gap=Int64(conf["gap"])
stepsize=Float64(conf["stepsize"])
stepsize_decay=Float64(conf["stepsize_decay"])
stepsize_decay_epochs=Int64(conf["stepsize_decay_epochs"])
alpha=Float64(conf["alpha"])
#cuda_device=Int64(conf["cuda_device"])

#CUDA.device!(1)


global p_cutoff = -1.0

const l_exp = 1:5
n_exp = length(l_exp)

l_train = []
l_val = []
for i = 1:n_exp
    j = l_exp[i]
    if !(j in [4])
        push!(l_train, i)
    else
        push!(l_val, i)
    end
end

opt = Flux.Optimiser(
    Adam(lr_adam, (0.9, 0.999), w_decay),
);

if !is_restart
    if ispath(fig_path)
        rm(fig_path, recursive = true)
    end
    if ispath(ckpt_path)
        rm(ckpt_path, recursive = true)
    end
end

if ispath("./results") == false
    mkdir("./results")
end

if ispath("./results/$expr_name") == false
    mkdir("./results/$expr_name")
end

if ispath(fig_path) == false
    mkdir(fig_path)
    mkdir(string(fig_path, "/conditions"))
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
end

cp("./config.yaml", config_path; force=true)
