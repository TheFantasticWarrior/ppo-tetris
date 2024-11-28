import pathlib


path = f"{pathlib.Path().absolute()}/save"
debug = 0
load = 1
render = 1
gpu = True  # not debug
action_space = 10

seed = 63
nenvs = 4 if debug else 16
nsteps = 128
nepoch = 4
nminibatch = 4

samplesperbatch = nenvs*nsteps//nminibatch
iterations = 5000

cliprange = 0.1
ent_coef = 0.02
vf_coef = 0.5
max_grad_norm = 0.5
