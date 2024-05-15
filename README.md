# transformerXL_PPO_JAX

This repository provides a JAX implementation of TranformerXL with PPO in a RL setup following :  "Stabilizing Transformers for Reinforcement Learning" from Parisotto et al. (https://arxiv.org/abs/1910.06764). 

The code uses the [PureJaxRL](https://github.com/luchris429/purejaxrl) template for PPO and copied some of the code from [Huggingface transformerXL](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py) transferring it to JAX.

The training handles [Gymnax](https://github.com/RobertTLange/gymnax) environment. 

We also tested it on [Craftax](https://github.com/MichaelTMatthews/Craftax/tree/main/craftax), on which it beat the baseline presented in the paper (https://arxiv.org/abs/2402.16801) including PPO-RNN, training with unsupervised environment design and intrinsic motivation. Notably we reach the 3rd level (the sewer) and obtain several advanced advancements, which was not achieved by the methods presented in the paper. See [Craftax Results](#results-on-craftax) for more informations. 

The training of a 5M transformer on craftax for 1e9 steps takes about 6h30 on a single A100. 

## Installation

```
git clone git@github.com:Reytuag/transformerXL_PPO_JAX.git
cd transformerXL_PPO_JAX
pip install requirements.txt
```
## Training 

You can edit the training config in train_PPO_trXL.py ( or train_PPOtrXL_pmap.py if you want to go multi GPU) including the name of the environment. (you can put any gymnax environment name, or "craftax" which will use the CraftaxSymbolic env)   

To launch the training: 
```
python3 train_PPO_trXL.py
```
Or if you go multi GPU.(it will use all your GPU) 
```
python3 train_PPO_trXL_pmap.py
```

## Results on Craftax 

![enter_sewerb](https://github.com/Reytuag/transformerXL_PPO_JAX/assets/76616547/b517835d-bcfd-4f49-866d-9a6123face18)


Without much parameter search, with a budget of 1e9 timesteps, the normalized return (\% max) achieve 18.3\% compared to 15.3\% for PPO-RNN according to the craftax paper. (with one seed visiting the sewer). 

![craftax_training_transfoXL_PPO](https://github.com/Reytuag/transformerXL_PPO_JAX/assets/76616547/80140a56-a77e-418e-86d7-305a6e43c5ac)

With a budget of 4e9 timesteps, the normalized return is 20.6 \%. Visiting the 3rd floor (the sewer) a decent amount of time and achieve several advanced achievements. Both of this was not reached by the baseline in the craftax paper even PPO-RNN with 10e9 interactions with the environment. 

## Related Works 
* Gymnax: https://github.com/RobertTLange/gymnax
* Craftax: https://github.com/MichaelTMatthews/Craftax
* Xland-Minigrid: https://github.com/corl-team/xland-minigrid
* PureJaxRL: https://github.com/luchris429/purejaxrl
* JaxMARL: https://github.com/FLAIROx/JaxMARL
* Jumanji: https://github.com/instadeepai/jumanji
* Evojax: https://github.com/google/evojax
* Evosax: https://github.com/RobertTLange/evosax
* Brax: https://github.com/google/brax


## Next steps 

* Train it on XLand-MiniGrid (https://github.com/corl-team/xland-minigrid) to test it on an open-ended environment in a meta-RL fashion.
* Add an implementation of Muesli (https://arxiv.org/abs/2104.06159) with transformerXL as in "Human-Timescale Adaptation in an Open-Ended Task Space" (https://arxiv.org/abs/2301.07608)



