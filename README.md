# MMD-VAE
Pytorch implementation of Maximum Mean Discrepancy Variational Autoencoder, a member of the InfoVAE family that maximizes Mutual Information between the Isotropic Gaussian Prior (as the latent space) and the Data Distribution.

Short explanation: The traditional VAE is known as the ELBO-VAE, named after the **E**vidence **L**ower **Bo**und used in its objective. The ELBO suffers from two problems: overestimation of latent variance, and uninformative latent information. 

The latter is because one of the objective's terms is the KL-Divergence between the Gaussian parameterized by the encoder and the Standard Isotropic Gaussian. This dissuades usage of the latent code, so that the KL-Divergence term is allowed to fall even further. It is important to note that the KL-Divergence should never truly reach zero, as that means the encoder is not learning useful features and cannot find feature locality, and the decoder is just randomly sampling from Standard Gaussian noise. 

The overestimation of variance results from the KL-Divergence term not being strong enough to balance against the Reconstruction Error, and thus the Encoder prefers to learn a multimodal latent distribution with spread apart means, leading to low training error as it overfits, but low quality samples as well, as the sampling distribution is assumed to be a Standard Isotropic Gaussian. One effort to mitigate this effect is the Disentangled Variational Autoencoder, which simply raises the weight on the KL-Divergence term. However, this increases the problem stated in the paragraph above since it further penalizes using the latent code.

For more detailed explanations, I used these resources to learn, in order of usefulness to me:
- https://arxiv.org/pdf/1706.02262.pdf
- http://ruishu.io/2018/03/14/vae/
- http://approximateinference.org/accepted/HoffmanJohnson2016.pdf
- https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
- http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/
- https://ermongroup.github.io/cs228-notes/inference/variational/
