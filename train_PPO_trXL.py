import time
from trainer_PPO_trXL import make_train
import jax
import os
import jax.numpy as jnp



config = {
    "LR": 2e-4,
    "NUM_ENVS": 512,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e6,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.002,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 1.,
    "ACTIVATION": "relu",
    "ENV_NAME": "MemoryChain-bsuite",
    "ANNEAL_LR": True,
    "qkv_features":256,
    "EMBED_SIZE":256,
    "num_heads":8,
    "num_layers":2,
    "hidden_layers":256,
    "WINDOW_MEM":128,
    "WINDOW_GRAD":64,
    "gating":True,
    "gating_bias":2.,
    "seed":0
}




seed=config["seed"]

prefix= "results_gymnax/"+config["ENV_NAME"]



try:
    if not os.path.exists(prefix):
                os.makedirs(prefix)
except:
    print("directory creation " + prefix +" failed")

time_a=time.time()
rng = jax.random.PRNGKey(seed)
train_jit = jax.jit(make_train(config))
out = train_jit(rng)
print("training and compilation took " + str(time.time()-time_a))


import matplotlib.pyplot as plt
plt.plot(out["metrics"]["returned_episode_returns"])
plt.xlabel("Updates")
plt.ylabel("Return")
plt.savefig(prefix+"/return_"+str(seed))

plt.clf()


jnp.save(prefix+"/"+str(seed)+"_params", out["runner_state"][0].params)
jnp.save(prefix+"/"+str(seed)+"_config", config)

jnp.save(prefix+"/"+str(seed)+"_metrics",out["metrics"])
