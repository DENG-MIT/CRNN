include("header.jl")



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
    filename = string("exp_data/cath_", string(cathode_num),"_", string(heating_rates[value]), ".csv")
    exp_data = Float64.(load_exp(filename))
    temps=exp_data[:, 1]
    times=(temps.-100).*60/heating_rates[value] #convert temperatures into times for ODE solving
    exp_data[:, 1]=times
    push!(l_exp_data, exp_data)
end

l_exp_info[:, 1] = heating_rates; 
