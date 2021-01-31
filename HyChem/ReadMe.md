# Learn HyChem

[HyChem](https://web.stanford.edu/group/haiwanglab/HyChem/) is one of the most important breakthroughs in the past five years in the Combustion Chemistry Community. The key idea is that the fuel pyrolysis process and the oxidation of C1-C4 small hydrocarbons are separated in flames. Therefore, one can lump the breakdown process from large hydrocarbons to C1-C4 small hydrocarbons into a few global steps, instead of building a complex hierarchy model.

The development of HyChem is attributed to the genius experts from Prof. Hai Wang's group at Stanford. We take the initiative to try to learn the HyChem model using CRNN, such that we can rely on the machines to autonomously do such kind of breakthrough research.

We start from a single initial condition demonstration, and the training for a wide range of initial conditions are undergoing.

## How to get started

You can download the required combustion mechanism from the HyChem website.

Use the python script to generate pyrolysis data, which employs Cantera.

Use the julia code to train CRNN.