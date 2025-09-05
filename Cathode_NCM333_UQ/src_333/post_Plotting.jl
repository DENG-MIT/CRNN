#export CUDA_VISIBLE_DEVICES=""


include("header.jl")
include("dataset.jl")
include("network.jl")
include("callback.jl")
@load string(ckpt_path_plot, "/mymodel.bson") p opt l_loss_train l_loss_val list_grad_data list_grad_repul iter p_opt_printer p_his iter gap 
#p=p_opt_printer

using PlotlyJS
using LaTeXStrings


function plot_sol_post(i_exp, HR1,HR2, HR3, exp_data, Tlist, cap, sol0 = nothing)
    
    sol=HR1+HR2+HR3
    plot!(
        plt,
        Tlist.-273.15,
        HR1,
        lw = 3,
        #legend = :left,
        c=:sienna1,
        alpha=0.025,

        #label = "CRNN R1",
    )
    plot!(
        plt,
        Tlist.-273.15,
        HR2,
        lw = 3,
        #legend = :left,
        c=:blue4,
        alpha=0.025,

        #label = "CRNN R2",
    )
    plot!(
        plt,
        Tlist.-273.15,
        HR3,
        lw = 3,
        #legend = :left,
        c=:yellow,
        alpha=0.025,

        #label = "CRNN R3",
    )
    plot!(
        plt,
        Tlist.-273.15,
        sol,
        lw = 3,
        #legend = :left,
        c=:green,
        label=false,
        alpha=0.1,
       # label = "CRNN sum",
    )

    if sol0 !== nothing
        plot!(
            plt,
            sol0.t / 60,
            sol0,
            #label = "initial model",
        )
    end
    xlabel!(plt, "Temperature (\$ ^ \\circ \$C)")
    ylabel!(plt, "Heat Release Rate (W/g)")
    #title!(plt, cap)
    #exp_cond = string(
    #    @sprintf(
    #        "T0 = %.1f K \n beta = %.2f K/min",
    #        T0,
    #        beta,
    #    )
    #)
    #annotate!(plt, exp_data[end, 1] / 60.0 * 0.85, 0.4, exp_cond)

    plot!(plt)
    return plt
end


#for each of the five, run it 500 times, and plot each one on the same subplot?

function make_plots(i_exp, tot_no)
    beta = l_exp_info[i_exp]
    exp_data = l_exp_data[i_exp]
    T0=100+273.15
    times=exp_data[:, 1]   
    Tlist = similar(times)
    ind = length(Tlist)
    for (i, t) in enumerate(times)
        Tlist[i] = getsampletemp(t, T0, beta)
    end

    value = l_exp[i_exp]
    global plt = Plots.plot(
        Tlist.-273.15,
        exp_data[:, 2:end],
        alpha=0.05,
        ylims = (minimum(exp_data[:, 2:end]),1.25*maximum(exp_data[:,2:end])),
        seriestype = :scatter,
        c=:black,
        dpi=1000,
        #label = "Exp",
        legend=false,
        label=false,

        grid=false,
        size=(700, 400)	

    )


    full_array_to_average=zeros(size(times)[1], tot_no)
    full_soln_arr=zeros(tot_no, 4, size(times)[1])
    for k in 1:tot_no
        sol, trunc_times, raw_sol = pred_n_ode(p[k, :], i_exp, exp_data)
        w_delH=p[k, 10:12].*p_scales[10:12]
        HR_all=HRR_getter_post(trunc_times, raw_sol[:, :], p[k, :]).*w_delH'
        HR1=HR_all[:, 1]
        HR2=HR_all[:, 2]
        HR3=HR_all[:, 3]
        full_array_to_average[1:length(trunc_times), k]=HR1+HR2+HR3
        full_soln_arr[k, 1, 1:length(trunc_times)]=deepcopy(HR1)
        full_soln_arr[k, 2, 1:length(trunc_times)]=deepcopy(HR2)
        full_soln_arr[k, 3, 1:length(trunc_times)]=deepcopy(HR3)
        full_soln_arr[k, 4, 1:length(trunc_times)]=HR1+HR2+HR3
        plot_sol_post(i_exp, HR1,HR2, HR3, exp_data, Tlist[1:length(trunc_times)], "exp_$value")
    end
    plot!(
        Tlist.-273.15,
        exp_data[:, 2:end],
        alpha=0.05,
        seriestype = :scatter,
        label = "Exp. Data",
        c=:black,
        legend=false,
        tickfontsize=10
    )
    plot!(
        Tlist.-273.15,
        mean(full_array_to_average, dims=2),
        c=:limegreen,
        linewidth=5,
        label="SVGD Models",
        #label = "Exp",
        legend=false,
        thickness_scaling=1.25
    )


    png(plt, string(fig_path, "/conditions/pred_exp_$value"))


    ###make another plot that has the 3*sigma UQ range?
    sigmas=std(full_soln_arr, dims=1)
    sigma_1=sigmas[:, 1, :]'
    sigma_2=sigmas[:, 2, :]'
    sigma_3=sigmas[:, 3, :]'
    sigma_4=sigmas[:, 4, :]'
    total_lower=mean(full_array_to_average, dims=2).-4*sigma_4
    total_upper=mean(full_array_to_average, dims=2).+4*sigma_4

    plt = Plots.plot(
        Tlist.-273.15,
        exp_data[:, 2:end],
        alpha=0.05,
        ylims = (minimum(exp_data[:, 2:end]),1.25*maximum(exp_data[:, 2:end])),
        seriestype = :scatter,
        c=:black,
        label = "Exp. Data",
        dpi=1000,
        grid=false,
        size=(700, 400)	

    )
    plot!(
        Tlist.-273.15,
        mean(full_array_to_average, dims=2),
        c=:springgreen2,
        label="SVGD",
        linewidth=5,
        legend=false,
        tickfontsize=10
            )
    plot!(Tlist.-273.15, total_lower, fillrange = total_upper, fillalpha = 0.5, c = :green, label = false, xlabel="Temperature (\$ ^ \\circ \$C)", ylabel="Heat Release Rate (W/g)", thickness_scaling=1.25)
    plot!(Tlist.-273.15, total_upper, c = :green, label=false,
    )

    png(plt, string(fig_path, "/conditions/UQ_pred_exp_$value"))


end

function A_Ea_plots(p, p_scales)
    w_A=p[:,1:3]
    w_Ea=p[:,4:6]
    axis_labels=["A1","A2","A3","E1","E2","E3","b1","b2","b3","H1","H2","H3","n1","n2","n3","v1","v2"]

    
    correlation=corkendall(p)
    heat=Plots.heatmap(
    correlation[1:17, 1:17],
    c = cgrad(:RdBu, rev=true),
    clims=(-1, 1))


    yticks!([1:1:17;], axis_labels)
    xticks!([1:1:17;], axis_labels)
    png(heat, string(fig_path, "/conditions/parameter_correlation"))
#=
    #this is the combined one, then below are the first couple trials:

    xx=w_A[:, 1]
    yy=w_Ea[:, 1]
    lin_fit=curve_fit(LinearFit, xx, yy)
    optimal_points=lin_fit.(xx)
    deviation=optimal_points-yy


    plt_p = scatter(
        w_A[:, 2],
        w_Ea[:, 2],
        zcolor=deviation,
        color=:inferno
    )
    xlabel!(plt_p, "A (rxn 2)")
    ylabel!(plt_p, "Ea (rxn 2)")
    png(plt_p, string(fig_path, "/conditions/Rxn2_vs_Rxn1_A"))
=#

    plt_p = Plots.plot(
        w_A[:, 1]*p_scales[1],
        w_Ea[:, 1]*p_scales[4],
        seriestype = :scatter,
        c=:purple,
        legend=false,
    )
    xlabel!(plt_p, "A (rxn 1)")
    ylabel!(plt_p, "Ea (rxn 1)")
    png(plt_p, string(fig_path, "/conditions/Rxn1_A_Ea"))




    plt_p = Plots.plot(
        w_A[:, 2]*p_scales[2],
        w_Ea[:, 2]*p_scales[5],
        seriestype = :scatter,
        c=:purple,
        legend=false,
    )
    xlabel!(plt_p, "A (rxn 2)")
    ylabel!(plt_p, "Ea (rxn 2)")
    png(plt_p, string(fig_path, "/conditions/Rxn2_A_Ea")) 



end

tot_no=size(p)[1]
display_p(p)

#make the movie 
#movie_maker_p()
#make the comparison plots
A_Ea_plots(p, p_scales)
skip_number=10 #mp4s will plot every skip_number'th set of parameters
history_indexer=Int(floor(iter/gap))
p_his_trunc=p_his[:, :, 1:skip_number:history_indexer]
A2_his=p_his_trunc[:, 2, :]*p_scales[2]
Ea2_his=p_his_trunc[:, 5, :]*p_scales[5]
A1_his=p_his_trunc[:, 1, :]*p_scales[1]
Ea1_his=p_his_trunc[:, 4, :]*p_scales[4]
A3_his=p_his_trunc[:, 3, :]*p_scales[3]
Ea3_his=p_his_trunc[:, 6, :]*p_scales[6]
n1_his=p_his_trunc[:, 13, :]*p_scales[13]
H1_his=p_his_trunc[:, 10, :]*p_scales[10]
H2_his=p_his_trunc[:, 11, :]*p_scales[11]
anim=@animate for i in 1:floor(Int, history_indexer/skip_number)
    Plots.scatter(A2_his[:, i], Ea2_his[:, i],
    xlim=(minimum(A2_his), maximum(A2_his)),
    ylim=(minimum(Ea2_his), maximum(Ea2_his)),
    xlabel="A2",
    ylabel="Ea2",
    label="epoch:"*string(i*skip_number*gap),
    legend=:topleft
    )
end
mp4(anim, string(fig_path, "/conditions/mp4_A_Ea_rxn2.mp4"), fps=10)

anim=@animate for i in 1:floor(Int, history_indexer/skip_number)
    Plots.scatter(A3_his[:, i], Ea3_his[:, i],
    xlim=(minimum(A3_his), maximum(A3_his)),
    ylim=(minimum(Ea3_his), maximum(Ea3_his)),
    xlabel="A3",
    ylabel="Ea3",
    label="epoch:"*string(i*skip_number*gap),
    legend=:topleft
    )
end
mp4(anim, string(fig_path, "/conditions/mp4_A_Ea_rxn3.mp4"), fps=10)

anim=@animate for i in 1:floor(Int, history_indexer/skip_number)
    Plots.scatter(A1_his[:, i], Ea1_his[:, i],
    xlim=(minimum(A1_his), maximum(A1_his)),
    ylim=(minimum(Ea1_his), maximum(Ea1_his)),
    xlabel="A1",
    ylabel="Ea1",
    label="epoch:"*string(i*skip_number),
    legend=:topleft)
end
mp4(anim, string(fig_path, "/conditions/mp4_A_Ea_rxn1.mp4"), fps=10)

anim=@animate for i in 1:floor(Int, history_indexer/skip_number)
    Plots.scatter(H1_his[:, i], H2_his[:, i], 
    xlim=(minimum(H1_his), maximum(H1_his)), 
    ylim=(minimum(H2_his), maximum(H2_his)),
    xlabel="H1",
    ylabel="H2",
    label="epoch:"*string(i*skip_number*gap),
    legend=:topleft)
end
mp4(anim, string(fig_path, "/conditions/mp4_H1_H2.mp4"), fps=10)



#make the individual realization and standard deviation plots
print(1)
make_plots(1, tot_no)
print(2)
make_plots(2, tot_no)
print(3)
make_plots(3, tot_no)
print(4)
make_plots(4, tot_no)
print(5)
make_plots(5, tot_no)



list_grad_data

