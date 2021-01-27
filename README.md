# CRNN (Chemical Reaction Neural Network)

CRNN is an interpretable neural network architecture for automonously inference chemical reaction pathways in various chemical systems. It is designed based on the following two fundamental physics laws: the Law of Mass Action and Arrhenius Law. It is also possible to incoorperate other physics laws to adapt CRNN to a specific domain.

<p align="center">
<img src="./assets/CRNN_TOC.png" width="500">
</p>

You can find the common questions regarding CRNN in the [FAQs](https://github.com/DENG-MIT/CRNN/wiki/FAQs).

# Structure of this repo

This repo provides the case studies presented in the orginal CRNN paper as well as ongoing prelimanry results on other systems. Currently, we are actively working on the following systems:

* Biomass pyrolysis kinetics for wildland fire modeling (to be announced)
* Cell signaling pathways for quantitative modeling drug effects (to be announced)
* Gene regulatory network (preliminary results in this repo)
* Oscillations in yeast glycolysis (preliminary results in this repo)
* Systems with strong stiffness (preliminary results on the Robertson's problem in this repo)

Inside each folder, such as case 1/2/3, you will find at least two Julia codes. One for traing and the other one for weight pruning. Those two files are basically identical, except that the weight pruning includes a function to prune the CRNN weights to further encourage sparsity.

# Cite
Ji, Weiqi, and Deng, Sili. "Autonomous Discovery of Unknown Reaction Pathways from Data by Chemical Reaction Neural Network." The Journal of Physical Chemistry A, (2021), doi: [10.1021/acs.jpca.0c09316](https://pubs.acs.org/doi/full/10.1021/acs.jpca.0c09316), [arXiv](https://arxiv.org/abs/2002.09062)
