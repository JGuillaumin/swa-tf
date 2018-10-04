
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

- To compute easily accuracy and loss across a subset, validation for example, I used `tf.metrics.mean` to accumulate 
    the means of theses metrics.

- I used `MomentumOptimizer` and `AdamOptimizer`. In the original paper, weight decay is applied on trainable variables.
    I used decoupled variants of theses optimizers (`tf.contrib.opt.AdamWOptimizer`, `tf.contrib.opt.MomentumWOptimizer`)
    which applied correctly weight decay (instead of adding `L2` term to the loss, which is equivalent to weight decay
    with adaptive gradient descent methods)


## Stochastic Weight Averaging (SWA): model ensembling within the weight space

SWA is new kind of ensembling method. Instead of mixing predictions from the 'prediction' space (like averaging 
predictions from multiple networks),
SWA averages different models within the 'weight space'. 
If you have to models (same network, same sets of trainable parameters), $\theta_t$ and $\theta_{t+T}$, you can average 
these weights. These weights must come from the same optimization trajectory, with $T$ relatively small
(like 1 or 2 epochs) without important changing in the learning rate during the period $t$.

SWA combines the idea of averaging models within the weight space and a specific learning rate scheduling. 
The goal is to average models from a region around the same local minimum. (you can't average models from different
local minimum, it will produce an averaged model placed outside of theses regions). This scheduling aims to go down 
in one local minimum when SWA starts. 


The first part of the training is classical one: you train your model with your preferred optimizer (like `MomentumW`) 
with a constant learning rate ($0.01$ for example) during $B epochs$.
You obtained a pre-trained model with parameters $\theta$.
Now SWA starts: 
- you initialize SWA weights with pre-trained weights $\theta_{SWA}=\theta$ and $n_models=1$ (`n_models` keeps 
    the number of averaged models).
- you train your network during $B epochs* more, with the learning rate described in (add Figure).
    The weights $\theta$ are optimized, not $\theta_{SWA}$. 
- at the end of each epoch, you add the new weights $\theta$ to $\theta_{SWA}$:

$$
\theta_{SWA} \leftarrow \frac{ \theta_{SWA} \dot n_{models} + \theta }{ n_{models} + 1 } 
n_{models} \leftarrow n_{models} + 1 
$$


At the end you obtained two sets of weights for the same model : $\theta$ and $\theta_{SWA}$. 
You can make inference directly on the model with weights $\theta$ since the batch norm statistics 
(`moving_mean` and `moving_variance`) are already sets for these parameters. 

For $\theta_{SWA}$, it's different. Internal statistics of BN layers are not set properly, because it's a new model,
from a new region within the weight space.
So you need to compute them by making some predictions with training data (full subset ou only some examples). 
Unfortunately, if you built and trained your network with `f.layers.batch_norm` it will be tedious to set the moving 
averages of 'mean' and 'variance' in each batch norm layers.

So I implemented a variant of Batch Normalization for TensorFlow with moving averages of statistics and 
classical averages of the statistics which requires a pre-inference step, with some training samples to set these new statistics. 
I called it `MovingFreeBatchNormalization`. This variant of batch norm is required for SWA weights, but I used is also the classical weights $\theta$.

Here is **a first conclusion** (with CIFAR10 on a ResNet34): do not use the moving statistics ! Fitting `mean` and `variance` 
at the end of the training (or before each validation) is much more efficient ! See Figure xxx. 


So during the second part of the training, when performing SWA updates, at the end of an epoch, if you want to evaluate
 the model with trained weights $\theta$ you need to make a `pre-inference` step to set the batch norm statistics. 

So after the SWA update, if you want the test your model, you need to:
- replace trained weights $\theta$ by $\theta_{SWA}$ within the network (I used backup variables to change easily the weights)
- set the batch norm statistics (it's a new model!) with a pre-inference step. 
- now you can make a complete inference ! 

For more details about moving free batch normalization with TensorFlow, see part xxxx


Stochastic Weight Averaging is not the first training method that averages weights of a model during the training. 
There is also the Exponential Moving Average average, which at each iteration step performs:

$$
\theta_t \leftarrow \text{optimizer(...)}
\theta_{EMA} \leftarrow \theta_{EMA} * \text{decay}_{EMA} + \theta_{t} * (1 - \text{decay}_{EMA})
$$

This formula is used at each iteration. 

EMA is implemented in TensorFlow, but generally not used correctly ! 
In the next section we will see how to use EMA correctly with TensorFlow. 

Since EMA is very close to SWA, I implemented SWA following the same code structure as `tf.train.ExponentialMovingAverage`.
And we will see how to use it properly. 

### How to use correctly Exponential Moving Averages with TensorFlow

```python
import tensorflow as tf

...

# build the model, it contains batch norm layers
loss, logits = build_model(inputs, labels, is_training)
opt = tf.train.AdamOptimizer(...)

# get the trainable variables
model_vars = tf.trainable_variables()

ema = tf.train.ExponentialMovingAverage(decay=0.999)

# use tf.control_dependencies to run the batch norm update, then the weight update and finaly the EMA formula
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = opt.minimize(loss, model_vars)
    with tf.control_dependencies([train_op,]):
        train_op = ema.apply(model_vars)

# now you can train you model, and EMA will be used, but not in your built network ! 
# accumulated weights are stored in ema.average(var) for a specific 'var'
# so you will evaluate your model with the classical weights, not with EMA weights
# trick : create backup variables to store trained weights, and operations to set weights use in the network to weights from EMA

# Make backup variables
with tf.variable_scope('BackupVariables'), tf.device('/cpu:0'):
    # force tensorflow to keep theese new variables on the CPU ! 
    backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                   initializer=var.initialized_value())
                   for var in model_vars]

# operation to assign EMA weights to model
ema_to_weights = tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in model_vars))
# operation to store model into backup variables
save_weight_backups = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))
# operation to get back values from backup variables to model
restore_weight_backups = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

# now you can train your model !
... 
for epoch in range(epochs):
    
    for step in range(steps_per_epoch):
        
        ...
        
        # running this op performs at lot of things: gradient descent, batch norm updates and EMA updates ! 
        sess.run(train_op, feed_dict=...)
        
    
    # now if you evaluate your model directly, trained weights will be used 
    acc, loss = make_inference(....)
    
    # you need to run operations defined above:
    # save weights
    sess.run(save_weight_backups)
    
    # replace weights by EMA ones
    sess.run(ema_to_weights)
    
    # now your network uses EMA weights !
    acc_ema, loss_ema = make_inference(....)
    
    # before the next training steps, you need to restore trained weights 
    sess.run(restore_weight_backups)
```

There are different methods to use EMA weights:
- play with multiple `tf.train.Saver` (use full if your are memory limited, there is no new variables)
- re-build the same network with EMA weights instead of already existing ones ! 


### How to use correctly Stochastic Moving Averages with TensorFlow

```python
import tensorflow as tf

...

# build the model, it contains batch norm layers
loss, logits = build_model(inputs, labels, is_training)
opt = tf.train.AdamOptimizer(...)

# get the trainable variables
model_vars = tf.trainable_variables()

# use tf.control_dependencies to run the batch norm update, then the gradient descent
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = opt.minimize(loss, model_vars)

# create an op that combines the SWA formula for all trainable weights 
swa = StochasticWeightAveraging()
swa_op = swa.apply(var_list=model_vars)

# now you can train you model, and EMA will be used, but not in your built network ! 
# accumulated weights are stored in ema.average(var) for a specific 'var'
# so you will evaluate your model with the classical weights, not with EMA weights
# trick : create backup variables to store trained weights, and operations to set weights use in the network to weights from EMA

# Make backup variables
with tf.variable_scope('BackupVariables'), tf.device('/cpu:0'):
    # force tensorflow to keep theese new variables on the CPU ! 
    backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                   initializer=var.initialized_value())
                   for var in model_vars]

# operation to assign SWA weights to model
swa_to_weights = tf.group(*(tf.assign(var, swa.average(var).read_value()) for var in model_vars))
# operation to store model into backup variables
save_weight_backups = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))
# operation to get back values from backup variables to model
restore_weight_backups = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

# now you can train your model !
... 
for epoch in range(epochs):
    
    for step in range(steps_per_epoch):
        
        ...
        
        # running this op performs at lot of things: gradient descent, batch norm updates 
        sess.run(train_op, feed_dict=...)
        
    
    # now if you evaluate your model directly, trained weights will be used 
    acc, loss = make_inference(....)
    
    # at the end of the epoch, you can run the SWA op which apply the formula defined above
    sess.run(swa_op)
    
    # now to evaluate the model with SWA weights :
    # save weights
    sess.run(save_weight_backups)
    
    # replace weights by SWA ones
    sess.run(swa_to_weights)
    
    # here, normaly you need to fit the batch norm statistics for this new model !!!
    # I will show up how to do this in the next section 
    fit_batch_norm(....)
    
    # now your network uses SWA weights !
    acc_swa, loss_ema = make_inference(....)
    
    # before the next training steps, you need to restore trained weights 
    sess.run(restore_weight_backups)
```



## Moving Free Batch Normalization: no more `moving_mean` and `moving_variance`


### How to use correctly classical batch normalization layers


### How to use correctly moving free batch normalization layers


## Results

- CIFAR10
- ResNet-34 v2 (pre-activation) (from [models/official/resnet](https://github.com/tensorflow/models/tree/master/official/resnet)) 
- 80%/20% for train/validation subsets 
- validation at each epoch (313 optimization steps)
- validation with moving statistics and with estimated statistics in BatchNorm (pre-inference step)
- data augmentation : random translation (8pixels), random flip L/R
- data normalization : per image normalization $x \leftarrow \frac{x - mean(x)}{std(x)}$
- epochs = 200
- batch_size = 128
- 1 GPU (GTX1070)
- TensorFlow 1.10, CUDA9.0, CuDNN 7.1.4 (compiled from source)
- about 1H15 per training


#### Baselines: constant learning rates with Momentum/MomentumW and Adam/AdamW

- learning rates in : {0.1, 0.05, 0.01, 0.005, 0.001, 0.0001}
- weigh decay : 1e-4 (when MomentumW or AdamW)
- test with saved model at highest accuracy 


model | validation | validation ++ | test | test ++
----- | ---------- | ------------- | ---- | -------
Momentum lr=0.1 | **89.16** | **89.49** | 87.33 | 88.88
MomentumW lr=0.1 | 86.86 | 89.89 | 84.42 | 88.17
Momentum lr=0.05 | 88.88 | 89.22 | **87.7** | 88.2
MomentumW lr=0.05 | 86.14 | 89.29 | 83.75 | 88.53
Momentum lr=0.01 | 87.61 | 88.41 | 86.55 | 87.67
MomentumW lr=0.01 | 87.08 | 89.47 | 86.15 | 88.47
Momentum lr=0.005 | 87.17 | 87.60 | 85.94 | 86.85
MomentumW lr=0.005 | 85.89 | 88.78 | 80.88 | 87.94
Momentum lr=0.001 | 83.05 | 83.65 | 81.17 | 83.04
MomentumW lr=0.001 | 85.39 | 88.96 | 84.66 | **89.24**
Momentum lr=0.0001 | 65.14 | 65.18 | 64.02 | 64.20
MomentumW lr=0.0001 | 76.85 | 80.05 | 68.12 | 79.83


model | validation | validation ++ | test | test ++
----- | ---------- | ------------- | ---- | -------
Adam lr=0.05 | 87.03 | 87.99 | 85.87 | 86.91
AdamW lr=0.05 | 85.06 | 87.71 | 79.94 | 86.77
Adam lr=0.01 | **88.81** | **89.76** | **87.95** | **89.15**
AdamW lr=0.01 | 85.96 | 89.33 | 85.23 | 88.94
Adam lr=0.005 | 87.97 | 89.41  86.81 | 88.17
AdamW lr=0.005 | 85.81 | 89.56  82.97 | 89.10
Adam lr=0.001 | 88.33 | 89.08 | 87.25 | 88.32
AdamW lr=0.001 | 86.56 | 89.14 | 80.46 | 88.48
Adam lr=0.0005 | 87.52 | 88.37 | 85.68 | 86.56
AdamW lr=0.0005 | 87.27 | 89.12 | 74.96 | 88.19
Adam lr=0.0001 | 82.15 | 83.52 | 80.95 | 82.67
AdamW lr=0.0001 | 88.12 | 89.34 | xx.xx | xx.xx

Conclusion: 
- When applying weight decay on all trainable variables (or L2 regularization) it seems very important to fit 
    the batch norm statistics instead of using the moving averages of mean and variance. 


#### Results of SWA 

py  
- epochs = 200
- epochs before SWA = 100
- start training with constant learning rate = $\alpha_1$ 
- then decrease linearly the learning rate to $\alpha_2$ in 80 epochs
- and continue training during 20 epochs with constant learning rate $\alpha_2$

- tuple ($\alha_1$, $\alpha_2$) tested: (0.1, 0.001), (0.05, 0.0005), (0.01, 0.0001), (0.001, 0.00001), (0.0005, 0.000005)

model | validation | validation ++ | validation SWA | test | test ++ | test SWA 
----- | ---------- | ------------- | -------------- | ---- | ------- | --------
Momentum (0.1, 0.001) | 90.10 | 90.09 | 90.26 | 89.54 | 89.67 | 89.94
MomentumW (0.1, 0.001) | **92.52** | **92.42** | **92.55** | 91.29 | 91.41 | **92.37**
Momentum (0.05, 0.0005) | 90.02 | 90.04 | 90.46 | 89.44 | 89.53 | 89.65
MomentumW (0.05, 0.0005) | 92.15 | 92.20 | 92.45 | **91.78** | 91.48 | 91.92
Momentum (0.01, 0.0001) | 89.06 | 88.97 | 89.14 | 88.52 | 88.61 | 88.69
MomentumW (0.01, 0.0001) | 92.13 | 92.23 | 92.30 | 91.47 | **91.49** | 91.87
Momentum (0.001, 0.00001) | 83.13 | 83.16 | 83.10 | 82.24 | 82.21 | 82.66
MomentumW (0.001, 0.00001) | 91.85 | 91.78 | 91.81 | 90.75 | 90.81 | 91.06



model | validation | validation ++ | validation SWA | test | test ++ | test SWA 
----- | ---------- | ------------- | -------------- | ---- | ------- | --------
Adam (0.1, 0.001) | 87.51 | 87.54 | 87.80 | 86.74 | 86.70 | 87.23
AdamW (0.1, 0.001) | 89.64 | 89.61 | 90.06 | 88.68 | 88.59 | 88.91
Adam (0.05, 0.0005) | 88.53 | 88.53 | 89.04 | 87.91 | 87.79 | 88.01
AdamW (0.05, 0.0005) | 90.17 | 90.10 | 90.44 | 89.14 | 88.98 | 89.93
Adam (0.01, 0.0001) | 90.58 | 90.61 | 90.64 | 89.05 | 89.06 | 89.36
AdamW (0.01, 0.0001) | 91.62 | 91.50 | 91.97 | 90.98 | 91.08 | **92.10**
Adam (0.001, 0.00001) | 89.87 | 89.87 | 90.00 | 89.05 | 89.02 | 89.01
AdamW (0.001, 0.00001) | **92.21** | **92.10** | **92.26** | **91.28** | **91.28** | 91.52
Adam (0.0005, 0.000005) | 88.18 | 88.21 | 88.52 | 88.06 | 87.96 | 88.22

