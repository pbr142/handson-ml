# Chapter 11 - Training Deep Neural Networks

* Vanishing and exploding gradients
* Limited data, not enough labeled data
* Slow training
* Risk of overfitting

## The Vanishing/Exploding Gradients Problems

Gradients get smaller and smaller going through the network -> lower layers are not updated (but can also explode, especially in recurrent networks)
General problem: Different layers learn at different speeds

Glorot, Bengio (2010): Combination of sigmoid activation and normal random initalization scheme -> Variance increases in each layer

### Glorot and He Initialization

Xavier initialization or Glorot initialization:

Intuition: Make variance equal for each layer
(not possible unless number of inputs = number of outputs)

Compromise: Weights initalized with $fan_{avg} = (fan_{in} + fan_{out})/2$

For logistic activation function:
Initialization with normal distribution with mean 0 and variance $\sigma^2 = \frac{1}{fan_{avg}}$
or with uniform distribution with $r = \sqrt{\frac{3}{fan_{avg}}}$

If $fan_{avg}$ is replaced with $fan_{in}$, then LeCun initialization.

He initialization $\sigma^2 = \frac{2}{fan_{in}}$


### Nonsaturating Activation Functions

Dying ReLU: If weighted sum for all training instances is negative, neuron only puts out 0

* Leaky ReLU $f_a(x) = max(ax,x)$, a typically 0.01
* Randomized leaky ReLU: a picked randomly during training, averaged in testing
* Parametric leaky ReLU: Learn leak parameter
* Exponential linear unit (ELU): $a(e^x-1)$ if $x<0$ and x otherwise
* Scaled ELU -> ELU multiplied by factor



### Batch Normalization



