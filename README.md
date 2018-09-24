
# Stochastic Weight Averaging (SWA) with TensorFlow

## Implemented features:

- `StochasticWeightAveraging` following the same code structure / usage of `tf.train.ExponentielMovingAverage`
    - we will see how to use correctly these two elements.

- as notices in the original paper, when using the weights from SWA algorithm we need to compute batch norm 
    statistics with theses weights  (instead of moving statistics computed during the training). 
    So I implemented a `MovingFreeBatchNormalization` layer (compatible with `tf.layers.Layer` and 
    `keras.layers.Layers` objects) where there is no moving statistics (mean and variance) but you have to compute 
    these values separately on the training set (or a subset). We will see how to use correctly classical batch 
    normalization and this variant of batch normalization.

- I also used `tf.data.Dataset` to feed samples from train/val/test subsets to the network. I combined `initializable` 
    and `feedable` iterators.

- To compute easily accuracy and loss across a subset, validation for example, I used `tf.metrics.mean` to accumulate the means of theses metrics.

- I used `MomentumOptimizer` and `AdamOptimizer`. In the original paper, weight decay is applied on trainable variables. I used decoupled variants 
    of theses optimizers (`tf.contrib.opt.AdamWOptimizer`, `tf.contrib.opt.MomentumWOptimizer`) which applied correctly weight decay 
    (instead of adding `L2` term to the loss, which is equivalent to weight decay with adaptive gradient descent methods)


## Stochastic Weight Averaging (SWA): model ensembling within the weight space

SWA is new kind of ensembling method. Instead of mixing predictions from the 'prediction' space (like averaging predictions from multiple networks),
SWA averages different models within the 'weight space'. 
If you have to models (same network, same sets of trainable parameters), $\theta_T$ and $\theta_{T+t}$, you can average these weights.
These weights must come from the same optimization trajectory, with $t$ relatively small (like 1 or 2 epochs) without important changing in the learning rate during the period $t$.

SWA combines the idea of averaging models within the weight space and a specific learning rate scheduling. 
The goal is to average models from a region around the same local minimum. (you can't average models from different local minimum,
it will produce an averaged model placed outside of theses regions). This scheduling aims to go down in one local minimum when SWA starts. 


The first part of the training is classical one: you train your model with your preferred optimizer (like `MomentumW`) 
with a constant learning rate ($0.01$ for example) during $B epochs$.
You obtained a pre-trained model with parameters $\theta$.
Now SWA starts: 
- you initialize SWA weights with pre-trained weights $\theta_{SWA}=\theta$ and $n_models=1$ (`n_models` keeps the number of averaged models).
- you train your network during $B epochs* more, with the learning rate described in (add Figure). The weights $\theta$ are optimized, not $\theta_{SWA}$. 
- at the end of each epoch, you add the new weights $\theta$ to $\theta_{SWA}$:

$$
\theta_{SWA} \leftarrow \frac{ \theta_{SWA} \dot n_{models} + \theta }{ n_{models} + 1 } 
n_{models} \leftarrow n_{models} + 1 
$$


At the end you obtained two sets of weights for the same model : $\theta$ and $\theta_{SWA}$. 
You can make inference directly on the model with weights $\theta$ since the batch norm statistics (mean and variance) are already sets for these parameters. 

For $\theta_{SWA}$, it's different. Internal statistics of BN layers are not set properly,because it's a ne model, from a new region within the weight space.
So you need to compute them by making some predictions with training data (full subset ou only some examples). 
Unfortunately, if you built and trained your network with `f.layers.batch_norm` it will be tedious to set the moving averages of 'mean' and 'variance' in each batch norm layers.

So I implemented a variant of Batch Normalization for TensorFlow without moving averages of statistics, which requires a pre-inference step, with some training samples to set the statistics. 
I called it `MovingFreeBatchNormalization`. This variant of batch norm is required for SWA weights, but I used is also the classical weights $\theta$. 

So during the second part of the training, when performin SWA updates, at the end of an epoch, if you want to test the model with trained weights $\theta$ you need to make a `pre inference`
step to set the batch norm statistics. 
Then after the SWA update, if you want the test your model, you need to:
- replace trained weights $\theta$ by $\theta_{SWA}$ within the network (I used backup variables to change easily the weights)
- set (again) the batch norm statistics (it's a new model!) with a pre-inference step. 
- now you can make a complete inference ! 


For more details about moving free batch normalization with TensorFlow, see part xxxx


### How to use correctly Exponential Moving Averages with TensorFlow


### How to use correctly Stochastic Moving Averages with TensorFlow



## Moving Free Batch Normalization: no more `moving_mean` and `moving_variance`


### How to use correctly classical batch normalization layers


### How to use correctly moving free batch normalization layers