#This could be cleaned up in the future, but works for now.
#Plots the initial solution, before training, based on the initialized kinetic parameters
#Every n_plot, plots the current trained solution as well as the loss profiles.
#All plots are in the /results/figs/ folder. 
#**Note that "plotting the solution" means plotting one randomly picked dataset. Not all datasets are re-plotted every n_plot epochs.**

function plot_sol(i_exp, HR1,HR2, HR3, exp_data, Tlist, cap, sol0 = nothing)
    beta = l_exp_info[i_exp]
    T0=100+273.15
    sol=HR1+HR2+HR3

    ind = length(Tlist)
    plt = plot(
        Tlist,
        exp_data[:, 2],
        seriestype = :scatter,
        label = "Exp",
    )

    plot!(
        plt,
        Tlist,
        HR1,
        lw = 3,
        legend = :left,
        label = "CRNN R1",
    )
    plot!(
        plt,
        Tlist,
        HR2,
        lw = 3,
        legend = :left,
        label = "CRNN R2",
    )
    plot!(
        plt,
        Tlist,
        HR3,
        lw = 3,
        legend = :left,
        label = "CRNN R3",
    )
    plot!(
        plt,
        Tlist,
        sol,
        lw = 3,
        legend = :left,
        label = "CRNN sum",
    )

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

    p2 = plot(Tlist, sol, lw = 2, legend = :right, label = "heat release")
    xlabel!(p2, "Time [min]")
    ylabel!(p2, "W/g")

    plt = plot(plt, p2, framestyle = :box, layout = @layout [a; b])
    plot!(plt, size = (800, 800))
    return plt
end

cbi = function (p, i_exp)
    exp_data = l_exp_data[i_exp]
    sol = pred_n_ode(p, i_exp, exp_data)[1]
    times = pred_n_ode(p, i_exp, exp_data)[2]
    raw_sol=pred_n_ode(p, i_exp, exp_data)[3]
    HR1=HRR_getter(times, raw_sol[:, :])[:, 1]*w_delH[1]
    HR2=HRR_getter(times, raw_sol[:, :])[:, 2]*w_delH[2]
    HR3=HRR_getter(times, raw_sol[:, :])[:, 3]*w_delH[3]
    Tlist = similar(times)
    T0=100+273.15
    beta = l_exp_info[i_exp, 1]
    for (i, t) in enumerate(times)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    value = l_exp[i_exp]
    plt = plot_sol(i_exp, HR1,HR2, HR3, exp_data, Tlist, "exp_$value")
    png(plt, string(fig_path, "/conditions/pred_exp_$value"))
    return false
end

function plot_loss(l_loss_train, l_loss_val; yscale = :log10)
    plt_loss = plot(l_loss_train, yscale = yscale, label = "train")
    plot!(plt_loss, l_loss_val, yscale = yscale, label = "val")
    plt_grad = plot(list_grad, yscale = yscale, label = "grad_norm")
    xlabel!(plt_loss, "Epoch")
    ylabel!(plt_loss, "Loss")
    xlabel!(plt_grad, "Epoch")
    ylabel!(plt_grad, "Gradient Norm")

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
list_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global l_loss_train, l_loss_val, list_grad, iter
    
    if !isempty(l_loss_train)
        if loss_train<minimum(l_loss_train)
            global p_opt=deepcopy(p)
        end
    end
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(list_grad, g_norm)
    

    if iter % n_plot == 0 || iter==1
        display_p(p)
        if @isdefined p_opt
            @printf("parameters of lowest yet loss:")
            display_p(p_opt)
        end
        list_exp = randperm(n_exp)[1]
        @printf(
            "Min Loss train: %.2e val: %.2e",
            minimum(l_loss_train),
            minimum(l_loss_val)
        )
        println("\n update plot ", l_exp[list_exp], "\n")
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plot_loss(l_loss_train, l_loss_val; yscale = :log10)
        if @isdefined p_opt
            @save string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val list_grad iter p_opt
        end
    end
    iter += 1
end

if is_restart
    @load string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val list_grad iter
    iter += 1
end
