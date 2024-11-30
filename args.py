import pathlib


path = f"{pathlib.Path().absolute()}/save"
debug = 0
load = 0
render = 1
gpu = True  # not debug
action_space = 10

seed = 1
nenvs = 4 if debug else 16
nsteps = 256
nepoch = 4
nminibatch = 4

samplesperbatch = nenvs*nsteps//nminibatch
iterations = 5000

cliprange = 0.1
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5

seq_len = 200
ff_size = 64
d_model = 32
