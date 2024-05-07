import time
from trainer_PPO_trXL_pmap import make_train,ActorCriticTransformer
import jax
import optax
import os
import jax.numpy as jnp


from flax.jax_utils import replicate, unreplicate


config = {
    "LR": 2e-4,
    "NUM_ENVS": 1024,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e9,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.999,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.002,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 1.,
    "ACTIVATION": "relu",
    "ENV_NAME": "Craftax",
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
}




seed=int(os.environ["SLURM_ARRAY_TASK_ID"])


prefix= "results/RELSHIFT_pmap2V100_1e9steps_memory128_layer2_gamma999_ent002"




try:
    if not os.path.exists(prefix):
                os.makedirs(prefix)
except:
    print("directory creation " + prefix +" failed")

    
    

time_a=time.time()
rng = jax.random.PRNGKey(seed)

rng,_rng=jax.random.split(rng)
train_fn,train_state  = (make_train(config,_rng))

print(jax.local_devices())


train_states = replicate(train_state, jax.local_devices())
rng=jax.random.split(rng,len(jax.local_devices()))



train_jit_fn= train_fn.lower(rng,train_states).compile()
print("compile took " + str(time_a-time.time()))

time_a=time.time()
out =train_jit_fn(rng,train_states)
a=out["metrics"]["returned_episode_returns"].block_until_ready()
print("training took " + str(time_a-time.time()))

#out=unreplicate(out)

out=jax.tree_util.tree_map(lambda x: x[0],out)

import matplotlib.pyplot as plt
plt.plot(out["metrics"]["returned_episode_returns"])
plt.xlabel("Updates")
plt.ylabel("Return")
plt.savefig(prefix+"/return_"+str(seed))

plt.clf()


jnp.save(prefix+"/"+str(seed)+"_params", out["runner_state"][0].params)
jnp.save(prefix+"/"+str(seed)+"_config", config)

jnp.save(prefix+"/"+str(seed)+"_metrics",out["metrics"])