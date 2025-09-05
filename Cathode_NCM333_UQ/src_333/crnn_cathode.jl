include("header.jl")
include("dataset.jl")
include("network.jl")
include("callback.jl")

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm_data = zeros(Float64, n_exp, np);
grad_norm_repul = zeros(Float64, n_exp, np);

#saving parameter snapshots for movies 
p_his_insert=zeros((size(p)[1],size(p)[2],Int(trunc(n_iter/gap))+1))
p_his_insert[..,1] = p






svgd = SVGD(p_his_insert,[zeros(2,2,2)])


for iter_curr in epochs
    global p
    for i_exp in randperm(n_exp)
        if i_exp in l_val
            loss_epoch[i_exp] = dlnprob(p, i_exp)[1] #save current loss, before grad update
            continue 
        end

        loss_curr, lnpgrad  = dlnprob(p, i_exp) #returns [grad, soln, loss]
        loss_epoch[i_exp] = loss_curr #save current loss, before grad update
       

        # calculating the kernel matrix
        kxy, dxkxy = svgd_kernel(svgd, p, -1) # kxy is the kernel function between x and y, dxkxy is the derivative of the kernel function w.r.t x
        # the shape of kxy is (num of particles, num of particles)
        # the shape of dxkxy is (num of particles, dim of x)
        # the shape of lnpgrad is (num of particles, dim of x)


        data_term=kxy*lnpgrad
        repulsion=dxkxy
        grad_p_curr=((data_term) + repulsion) / size(p)[1]

        grad_norm_data[i_exp, :] = norm2(data_term, dims=1) #saving the separate data and repulsion gradients, parameter by parameter
        grad_norm_repul[i_exp, :] = norm2(repulsion, dims= 1) 
        grad_norm = norm(grad_p_curr, 2) 

        p = p .+ stepsize * grad_p_curr
    end


    if (iter_curr+1) % gap == 0
        svgd.p_his[..,Int(trunc((iter_curr+1)/gap))] = p #save parameters every so often for movie
        
    end
    

    loss_train = mean(loss_epoch[l_train])
    loss_val = mean(loss_epoch[l_val])
    grad_mean_data = mean(grad_norm_data[l_train, :], dims=1) #keep it with len(p) dimensions, but norm across the datasets 
    grad_mean_repul= mean(grad_norm_repul[l_train, :], dims=1)
    set_description(
        epochs,
        string(
            @sprintf(
                "Loss train: %.2e val: %.2e data grad mean: %.2e repulsion grad mean: %.2e ",
                loss_train,
                loss_val,
                norm(grad_mean_data),
                norm(grad_mean_repul)
                #opt[1].eta
            )
        ),
    )
    cb(p, loss_train, loss_val, grad_mean_data, grad_mean_repul, svgd.p_his, gap)
end

conf["loss_train"] = minimum(l_loss_train)
conf["loss_val"] = minimum(l_loss_val)
YAML.write_file(config_path, conf)

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end

