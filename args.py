import pathlib


path = f"{pathlib.Path().absolute()}/save"
debug_acs = 0
load = 1
partial = 0
render = 0
gpu = True  # not debug
action_space = 10

seed = 1576
nenvs = 16
nepoch = 4
nminibatch = 10

nsteps = 320
samplesperbatch = nsteps*nenvs//nminibatch
#samplesperbatch = 256
# nsteps= samplesperbatch*nminibatch//nenvs
# print(f"{nsteps=}")
iterations = 5000
training_steps = iterations*nminibatch*nepoch
warmup_steps = iterations/10*nminibatch*nepoch

lr = 2.5e-4
cliprange = 0.1
ent_coef = 0.01
# ent_coef = 0.00002
vf_coef = 0.5
max_grad_norm = 0.5
gamma = 0.999

seq_len = 200
ff_size = 128
d_model = 64
