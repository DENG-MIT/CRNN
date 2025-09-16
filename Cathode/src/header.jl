using Random, Plots
using Zygote, ForwardDiff
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

ENV["GKSwstype"] = "100"

cd(dirname(@__DIR__))
conf = YAML.load_file("./config.yaml")

expr_name = conf["expr_name"]
fig_path = string("./results/", expr_name, "/figs")
ckpt_path = string("./results/", expr_name, "/checkpoint")
config_path = "./results/$expr_name/config.yaml"

is_restart = Bool(conf["is_restart"])
ns = Int64(conf["ns"])
nr = Int64(conf["nr"])
lb = Float64(conf["lb"])
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

llb = lb;
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

opt = AdamW(lr_adam, (0.9, 0.999), w_decay);

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
