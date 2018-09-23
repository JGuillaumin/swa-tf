
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


### How to use correctly Exponential Moving Averages with TensorFlow


### How to use correctly Stochastic Moving Averages with TensorFlow



## Moving Free Batch Normalization: no more `moving_mean` and `moving_variance`


### How to use correctly classical batch normalization layers


### How to use correctly moving free batch normalization layers