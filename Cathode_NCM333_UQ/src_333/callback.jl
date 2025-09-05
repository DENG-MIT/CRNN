
function plot_sol(i_exp, HR1,HR2, HR3, exp_data, Tlist, cap, sol0 = nothing)
    beta = l_exp_info[i_exp]
    T0=100+273.15
    sol=zeros(size(HR1))

    ind = length(Tlist)
    plt = plot(
        Tlist,
        exp_data[:, 2],
        seriestype = :scatter,
        label = "Exp",
    )
    for k =1:num_particles
        sol[k, :]=HR1[k, :]+HR2[k, :]+HR3[k, :]
        plot!(
            plt,
            Tlist,
            HR1[k, :],
            lw = 3,
            legend = :left,
            label = "CRNN R1",
        )
        plot!(
            plt,
            Tlist,
            HR2[k, :],
            lw = 3,
            legend = :left,
            label = "CRNN R2",
        )
        plot!(
            plt,
            Tlist,
            HR3[k, :],
            lw = 3,
            legend = :left,
            label = "CRNN R3",
        )
        plot!(
            plt,
            Tlist,
            sol[k, :],
            lw = 3,
            legend = :left,
            label = "CRNN sum",
        )
    end
    if sol0 !== nothing
        plot!(
            plt,
            sol0.t / 60,
            sol0,
            label = "initial model",
        )
    end
    xlabel!(plt, "Time [min]")
    ylabel!(plt, "HRR")
    title!(plt, cap)
    exp_cond = string(
        @sprintf(
            "T0 = %.1f K \n beta = %.2f K/min",
            T0,
            beta,
        )
    )
    annotate!(plt, exp_data[end, 1] / 60.0 * 0.85, 0.4, exp_cond)

    p2 = plot(Tlist, sol[1, :], lw = 2, legend = :right, label = "heat release")
    for k = 2:num_particles
        plot!(Tlist, sol[k, :], lw = 2, legend = :right, label = "heat release")
    end
    xlabel!(p2, "Time [min]")
    ylabel!(p2, "W/g or something")

    plt = plot(plt, p2, framestyle = :box, layout = @layout [a; b])
    plot!(plt, size = (800, 800))
    return plt
end

cbi = function (p, i_exp)
    exp_data = l_exp_data[i_exp]
    times=exp_data[:, 1]
    size_curr=size(exp_data)[1]
    Tlist = similar(times)

    T0=100+273.15
    beta = l_exp_info[i_exp, 1]
    for (i, t) in enumerate(times)
        Tlist[i] = getsampletemp(t, T0, beta)
    end

    HR1=zeros(num_particles, size_curr)
    HR2=zeros(num_particles, size_curr)
    HR3=zeros(num_particles, size_curr)

    for k = 1:num_particles
            
        sol, times, raw_sol = pred_n_ode(p[k, :], i_exp, exp_data)
        w_delH=p[k, 10:12].*p_scales[10:12]
        HR_all=HRR_getter(times, raw_sol[:, :], p[k, :]).*w_delH'
        HR1[k, :]=HR_all[:, 1]
        HR2[k, :]=HR_all[:, 2]
        HR3[k, :]=HR_all[:, 3]
    end
    
    value = l_exp[i_exp]
    plt = plot_sol(i_exp, HR1,HR2, HR3, exp_data, Tlist, "exp_$value")
    png(plt, string(fig_path, "/conditions/pred_exp_$value"))
    return false
end

function plot_loss(l_loss_train, l_loss_val; yscale = :log10)
    plt_loss = plot(l_loss_train, yscale = yscale, label = "train")
    plot!(plt_loss, l_loss_val, yscale = yscale, label = "val")
    plt_grad = plot(norm2(list_grad_data, dims=2), yscale = yscale, label = "grad_norm_data")
    plot!(norm2(list_grad_repul, dims=2), yscale = yscale, label = "grad_norm_repulsion")
    xlabel!(plt_loss, "Epoch")
    ylabel!(plt_loss, "Loss")
    xlabel!(plt_grad, "Epoch")
    ylabel!(plt_grad, "Gradient Norm")
    # ylims!(plt_loss, (-Inf, 1e0))
    # ylims!(plt_grad, (-Inf, 1e3))
    plt_all = plot([plt_loss, plt_grad]..., legend = :top, framestyle=:box)
    plot!(
        plt_all,
        size=(1000, 450),
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
    )
    png(plt_all, string(fig_path, "/loss_grad"))
end

l_loss_train = []
l_loss_val = []
list_grad_data = []
list_grad_repul = []
l_prior=[]
l_post=[]
l_recon=[]
iter = 1
p_opt_printer=deepcopy(p)

cb = function (p, loss_train, loss_val, grad_mean_data, grad_mean_repul, p_his, gap)
    global l_loss_train, l_loss_val, list_grad_data, list_grad_repul, iter

    if !isempty(l_loss_train)
        if loss_train<minimum(l_loss_train)
            global p_opt_printer=deepcopy(p)
        end
    end
    if iter==1
        list_grad_data=grad_mean_data
        list_grad_repul=grad_mean_repul
    else
        list_grad_data=vcat(list_grad_data, grad_mean_data)
        list_grad_repul=vcat(list_grad_repul, grad_mean_repul)
    end
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)

    if iter % n_plot == 0 || iter==1
        display_p(p)
        if @isdefined p_opt_printer
            @printf("parameters of lowest yet loss:")
            display_p(p_opt_printer)
        end
        list_exp = randperm(n_exp)[1]
        @printf(
            "Min Loss train: %.2e val: %.2e",
            minimum(l_loss_train),
            minimum(l_loss_val),
        )
        println("\n update plot ", l_exp[list_exp], "\n")
        for i_exp in list_exp
           #cbi(p, i_exp)
        end

        plot_loss(l_loss_train, l_loss_val; yscale = :log10)
        if @isdefined p_opt_printer

            @save string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val list_grad_data list_grad_repul iter p_opt_printer p_his iter gap
        end
    end
    iter += 1
end
9

