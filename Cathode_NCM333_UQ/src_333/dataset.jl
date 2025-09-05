#include("header.jl")

noise_level = Float64(conf["noise_level_noisy_wang"])

function load_exp(filename)
    exp_data = readdlm(filename,',', Float64) #[t, HRR]
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    return exp_data
end

l_exp_data = [];
l_exp_info = zeros(Float64, length(l_exp), 1);
heating_rates=[2, 5, 10, 15, 20]
for (i_exp, value) in enumerate(l_exp)
    filename = string("exp_data/UNCERT_cath_", string(cathode_num),"_", string(heating_rates[value]), ".csv")
    exp_data = Float64.(load_exp(filename))
    # turn temp range into time range
    temps=exp_data[:, 1]
    times=(temps.-100).*60/heating_rates[value]
    exp_data[:, 1]=times
    
    push!(l_exp_data, exp_data)
end

#separate normalizer for each rxn based on peak value and noise
Normalizer=zeros(5, 3)
Normalizer[1, :]=[.00538199, 0.00538199, 0.00538199]
Normalizer[2, :]=[0.01230299,0.01230299,0.01230299]
Normalizer[3, :]=[0.02823027,0.02823027,0.02823027]
Normalizer[4, :]=[0.04655562,0.04655562,0.04655562] 
Normalizer[5, :]=[0.05480013,0.05480013,0.05480013]


l_exp_info[:, 1] = heating_rates;
