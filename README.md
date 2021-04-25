# CRNN (Chemical Reaction Neural Network)

CRNN is an interpretable neural network architecture for autonomously inference chemical reaction pathways in various chemical systems. It is designed based on the following two fundamental physics laws: the Law of Mass Action and Arrhenius Law. It is also possible to incorporate other physics laws to adapt CRNN to a specific domain.

<p align="center">
<img src="./assets/CRNN_TOC.png" width="500">
</p>

You can find the common questions regarding CRNN in the [FAQs](https://github.com/DENG-MIT/CRNN/wiki/FAQs).

# Structure of this repo

This repo provides the case studies presented in the original CRNN paper as well as ongoing preliminary results on other systems. Currently, we are actively working on the following systems:

* [Biomass pyrolysis kinetics](https://github.com/DENG-MIT/CRNN-Pyrolysis)
* [Cell signaling pathways for quantitative modeling drug effects](https://github.com/jiweiqi/CellBox.jl)
* Gene regulatory network (preliminary results)
* Oscillations in yeast glycolysis (preliminary results)
* Systems with strong stiffness (Robertson's problem)
* Gas-phase combustion kinetics (`HyChem`)

Inside each folder, such as case 1/2/3, you will find at least two Julia codes. One for training and the other one for weight pruning. Those two files are identical, except that the weight pruning includes a function to prune the CRNN weights to further encourage sparsity.

# Get Started

Have a look at the code for [case 2](https://github.com/DENG-MIT/CRNN/blob/main/case2/case2.jl). The script consists of the following parts:
* Hyper-parameter settings
* Generate synthesized data
* Define the neural ODE problem
* Train CRNN using ADAM

# Cite
Ji, Weiqi, and Deng, Sili. "Autonomous Discovery of Unknown Reaction Pathways from Data by Chemical Reaction Neural Network." The Journal of Physical Chemistry A, (2021), 125, 4, 1082–1092, [acs](https://pubs.acs.org/doi/full/10.1021/acs.jpca.0c09316)/[arXiv](https://arxiv.org/abs/2002.09062)
