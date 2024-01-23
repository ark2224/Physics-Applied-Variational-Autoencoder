# Variational Autoencoders

**Variational Autoencoders** (or VAEs) seek to learn some hidden or
“latent” lower-dimensional representation of data points in a training
set. One half of the VAE’s purpose is to encode/transform input
variables into a latent space; the other half is to reverse this
transform, or decode data, by taking a distribution of variables in the
latent space and converting them back into the inputted space and the
distribution of the original inputs. To get a better idea, take a high
dimensional input such as an image with thousands of pixels. Each
image’s representation will translate to a vector with thousands of
dimensions. It is possible to use this vector for calculations and
storage, however, it is also possible to reduce this vector into a much
more condensed representation. The manifold hypothesis states that real
world high dimensional data such as this consists of low dimensional
data that is embedded in the high dimensional space. In other words,
high dimensional data can be accurately represented by lower dimensional
data, providing more efficient calculations and storage. The main goal
of the VAE is to find an accurate transformation of data from high to
low dimensions as well as low to high. The advantage of transforming
data in both directions is that it ensures the transformation to lower
dimensions still captures the unique features of the original data as
well as maintaining differences between separate datum. The other
advantage is that reversing the transformation (i.e. going from lower
dimensional latent spaces to original high dimensions) makes the VAE a
form of unsupervised learning. The VAE calculates its own accuracy and
“learns” by comparing the data transformed to and from the latent space
to the original inputs. It uses this difference between the
transformed-retransformed data and untransformed data to update the
encoding and decoding parameters accordingly. The unique feature about
VAEs and what distinguishes them from typical autoencoders is that the
encoded data is not represented as a vector but as a distribution: each
data point in the latent space is represented by a mean and standard
deviation.

## Objective

The objective of the VAE is to create a distribution of input data p(x)
that represents the entire set of input data as well as a distribution
of the original data transformed into the latent space p(z\|x). In terms
of this study, our objective is to make an input distribution that
represents the space of possible wavelet scattering transform (WST)
coefficients used in the modeling of polarized dust emission from
interstellar medium. Our latent space, or the focus of our study, is to
create a distribution of reduced wavelet scattering transform (RWST)
coefficients. Past papers have attempted to do this through other
methods\[add reference\] and have returned decent embeddings. However,
we believe that the use of deep learning and neural nets such as the VAE
produce a more accurate embedding of RWSTs. To do this, two VAE models
are used to provide RWST and WST coefficients: one that produces RWSTs
of slightly higher dimension than past work, and another model that
produces RWSTs of much lower dimensional space. We believe that both
models produce more accurate embeddings of WSTs than in past work and
serve as a new approach toward better representations of galactic dust
emission.

## Our Models

Our models are based on a convolutional neural net design that was
originally used by Thorne et al. for a similar purpose. The
architectures of the first VAE’s encoder and decoder are tabulated below
in Tables 1-2.

| Table 1. VAE-1 Encoder for Large-Latent Space |                  |                           |
|:----------------------------------------------|:-----------------|:--------------------------|
| Layers                                        | Output Dimension | Hyperparameters           |
| Input                                         | (1, 1360, 1)     |                           |
| Conv2D                                        | (1, 1360, 1)     | Kernel = 28, Dilation = 2 |
| ReLU                                          | (1, 1360, 1)     |                           |
| BatchNorm                                     | (1, 1360, 1)     | Momentum = 0.9            |
| Conv2D                                        | (1, 680, 128)    | Kernel = 14, Dilation = 2 |
| ReLU                                          | (1, 680, 128)    |                           |
| BatchNorm                                     | (1, 680, 128)    | Momentum = 0.9            |
| Conv2D                                        | (1, 170, 32)     | Kernel = 7, Dilation = 2  |
| ReLU                                          | (1, 170, 32)     |                           |
| BatchNorm                                     | (1, 170, 32)     | Momentum = 0.9            |
| Dense                                         | 340              |                           |
| Dense                                         | 170              |                           |

| Table 2. VAE-1 Decoder for Large-Latent Space |                  |                          |
|:----------------------------------------------|:-----------------|:-------------------------|
| Layers                                        | Output Dimension | Hyperparameters          |
| Input                                         | (1, 170, 1)      |                          |
| Dense                                         | 5440             |                          |
| Reshape                                       | (1, 170, 32)     |                          |
| BatchNorm                                     | (1, 170, 32)     | Momentum = 0.9           |
| TransposeConv2D                               | (1, 340, 128)    | Kernel = 14, Strides = 2 |
| ReLU                                          | (1, 340, 128)    |                          |
| BatchNorm                                     | (1, 340, 128)    | Momentum = 0.9           |
| TransposeConv2D                               | (1, 340, 64)     | Kernel = 14, Strides = 1 |
| ReLU                                          | (1, 340, 64)     |                          |
| BatchNorm                                     | (1, 340, 64)     | Momentum = 0.9           |
| TransposeConv2D                               | (1, 680, 32)     | Kernel = 28, Strides = 2 |
| ReLU                                          | (1, 680, 32)     |                          |
| BatchNorm                                     | (1, 680, 32)     | Momentum = 0.9           |
| TransposeConv2D                               | (1, 1360, 1)     | Kernel = 56, Strides = 2 |
| ReLU                                          | (1, 1360, 1)     |                          |
| BatchNorm                                     | (1, 1360, 1)     | Momentum = 0.9           |
| TransposeConv2D                               | (1, 1360, 1)     | Kernel = 56, Strides = 1 |

The distinguishing characteristic of this first VAE is that it produces
a latent space (sets of RWST coefficients) with a slightly higher
dimension than that of Thorne et al. The purpose of this VAE is to
demonstrate that using neural nets and deep learning provide more
accurate and efficient means to find an embedding of WST coefficients.
The second VAE within this study is meant to demonstrate that a deep
learning methodology makes it possible to find a lower dimensional
embedding while maintaining similar accuracy to Thorne et al and the
first VAE.

| Table 3. VAE-2 Encoder for Smaller-Latent Space |                  |                           |
|:------------------------------------------------|:-----------------|:--------------------------|
| Layers                                          | Output Dimension | Hyperparameters           |
| Input                                           | (1, 1360, 1)     |                           |
| Conv2D                                          | (1, 1360, 1)     | Kernel = 28, Dilation = 2 |
| ReLU                                            | (1, 1360, 1)     |                           |
| BatchNorm                                       | (1, 1360, 1)     | Momentum = 0.9            |
| Conv2D                                          | (1, 680, 128)    | Kernel = 14, Dilation = 2 |
| ReLU                                            | (1, 680, 128)    |                           |
| BatchNorm                                       | (1, 680, 128)    | Momentum = 0.9            |
| Conv2D                                          | (1, 170, 32)     | Kernel = 7, Dilation = 2  |
| ReLU                                            | (1, 170, 32)     |                           |
| BatchNorm                                       | (1, 170, 32)     | Momentum = 0.9            |
| Conv2D                                          | (1, 85, 16)      | Kernel = 14, Dilation = 2 |
| ReLU                                            | (1, 85, 16)      |                           |
| BatchNorm                                       | (1, 85, 16)      | Momentum = 0.9            |
| Dense                                           | 170              |                           |
| Dense                                           | 85               |                           |

| Table 4. VAE-2 Decoder for Smaller-Latent Space |                  |                          |
|:------------------------------------------------|:-----------------|:-------------------------|
| Layers                                          | Output Dimension | Hyperparameters          |
| Input                                           | (1, 85, 1)       |                          |
| Dense                                           | 2720             |                          |
| Reshape                                         | (1, 85, 32)      |                          |
| BatchNorm                                       | (1, 85, 32)      | Momentum = 0.9           |
| TransposeConv2D                                 | (1, 85, 128)     | Kernel = 14, Strides = 2 |
| ReLU                                            | (1, 85, 128)     |                          |
| BatchNorm                                       | (1, 85, 128)     | Momentum = 0.9           |
| TransposeConv2D                                 | (1, 340, 64)     | Kernel = 14, Strides = 2 |
| ReLU                                            | (1, 340, 64)     |                          |
| BatchNorm                                       | (1, 340, 64)     | Momentum = 0.9           |
| TransposeConv2D                                 | (1, 680, 32)     | Kernel = 28, Strides = 2 |
| ReLU                                            | (1, 680, 32)     |                          |
| BatchNorm                                       | (1, 680, 32)     | Momentum = 0.9           |
| TransposeConv2D                                 | (1, 1360, 1)     | Kernel = 56, Strides = 2 |
| ReLU                                            | (1, 1360, 1)     |                          |
| BatchNorm                                       | (1, 1360, 1)     | Momentum = 0.9           |
| TransposeConv2D                                 | (1, 1360, 1)     | Kernel = 56, Strides = 1 |

Part of both VAE classes are four functions for operating the neural
net. First is the "sample" function that takes a vector from a normal
distribution in the input space. This vector can then be fed into the
neural net to transform it into the latent space using the next function
"encode". The "encode" function takes in one vector that should be of
shape (1,1360) and feeds this vector into the neural net’s encoder
attribute that returns the vector’s transformation into the latent space
distribution with a mean and standard deviation. The function
"reparameterize" calculates a specific vector z in the latent space from
the calculated distribution returned by the "encode" function. The
purpose of "reparameterize" is to take a sample vector from the latent
space distribution to feed into the "decode" function and decode
attribute to later backpropagate into the encoding and decoding
functions. The last function is "decode" which is meant to take a sample
vector from distributions in the latent space and decode them into the
original input space. The purpose of this function is to transform
vectors from latent to input space and calculate the loss between these
decoded vectors and where they fall in the input data’s distribution.

Below the models independent functions for calculating the loss and
updating the neural net parameters with a given optimizer. The first
function log-normal-pdf takes three parameters: a sample input vector, a
mean, and log of the distribution’s variance. Its purpose is to
calculate the logarithmic probability of a vector under a certain
distribution. The function compute-loss uses log-normal-pdf for
calculating the total loss of the encoding and decoding process to
update the neural net’s parameters. For calculating the loss, this
function calculates the ELBO loss: a sum of the log probability of input
vector x given latent vector z, the log probability of latent vector z
in the latent space normal distribution, and the log probability of
latent vector z given input vector x under the input data’s
distribution. Lastly, the train-step function conducts one training run
through the VAE. It takes one input vector and inputs it into the VAE to
get one returned encoded-decoded vector. The function then computes the
ELBO loss using compute-loss and uses the gradient of each of the VAE’s
parameters with a Tensorflow package optimizer Adam to update the
parameters themselves.

Both VAE’s are trained with five epochs each through a training dataset
of 1310 sets of WST’s. The loss calculated by the first VAE is around
10<sup>−404</sup> whereas the second VAE’s loss is around
10<sup>−943</sup>.

## References

-   [TDS guide on
    VAEs](https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed)

-   [Somenes github on VAEs](https://avandekleut.github.io/vae/)

-   [tensorflow VAE
    tutorial](https://www.tensorflow.org/tutorials/generative/cvae)
