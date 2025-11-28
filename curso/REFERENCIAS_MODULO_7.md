# 📚 Referências Bibliográficas - Módulo 7: GANs e VAEs

> Este documento complementa o [Módulo 7: GANs e VAEs - Geração Sintética de Imagens](07_gans_vaes_geracao_sintetica.ipynb)

---

## Papers Fundamentais

### GANs (Generative Adversarial Networks)

1. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014)**
   *Generative adversarial nets*
   Advances in neural information processing systems, 27.
   📄 [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
   💡 **Paper original que introduziu GANs** - Revolucionou geração de imagens com framework adversarial

2. **Radford, A., Metz, L., & Chintala, S. (2015)**
   *Unsupervised representation learning with deep convolutional generative adversarial networks*
   arXiv preprint arXiv:1511.06434.
   📄 [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
   💡 **DCGAN** - Arquitetura convolucional que estabilizou treinamento de GANs

3. **Arjovsky, M., Chintala, S., & Bottou, L. (2017)**
   *Wasserstein generative adversarial networks*
   International conference on machine learning (pp. 214-223). PMLR.
   📄 [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
   💡 **WGAN** - Solução para instabilidade usando distância de Wasserstein

4. **Karras, T., Laine, S., & Aila, T. (2019)**
   *A style-based generator architecture for generative adversarial networks*
   Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4401-4410).
   📄 [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
   💡 **StyleGAN** - Controle de estilo hierárquico para geração de faces realistas

### VAEs (Variational Autoencoders)

5. **Kingma, D. P., & Welling, M. (2013)**
   *Auto-encoding variational bayes*
   arXiv preprint arXiv:1312.6114.
   📄 [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
   💡 **Paper original de VAEs** - Introduziu reparameterization trick e ELBO

6. **Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017)**
   *beta-vae: Learning basic visual concepts with a constrained variational framework*
   ICLR.
   📄 [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
   💡 **β-VAE** - Aprendizado de representações disentangled

7. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017)**
   *Neural discrete representation learning*
   Advances in neural information processing systems, 30.
   📄 [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
   💡 **VQ-VAE** - Quantização vetorial para representações discretas

---

## Tutoriais e Surveys

8. **Goodfellow, I. (2016)**
   *NIPS 2016 tutorial: Generative adversarial networks*
   arXiv preprint arXiv:1701.00160.
   📄 [arXiv:1701.00160](https://arxiv.org/abs/1701.00160)
   💡 Tutorial oficial do criador das GANs

9. **Doersch, C. (2016)**
   *Tutorial on variational autoencoders*
   arXiv preprint arXiv:1606.05908.
   📄 [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)
   💡 Tutorial didático e completo sobre VAEs

10. **Creswell, A., White, T., Dumoulin, V., Arulkumaran, K., Sengupta, B., & Bharath, A. A. (2018)**
    *Generative adversarial networks: An overview*
    IEEE signal processing magazine, 35(1), 53-65.
    📄 [arXiv:1710.07035](https://arxiv.org/abs/1710.07035)
    💡 Survey abrangente sobre GANs

---

## Variantes e Melhorias

### GANs Avançadas

11. **Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017)**
    *Improved training of wasserstein gans*
    Advances in neural information processing systems, 30.
    📄 [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
    💡 **WGAN-GP** - Gradient penalty para estabilidade

12. **Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017)**
    *Progressive growing of gans for improved quality, stability, and variation*
    arXiv preprint arXiv:1710.10196.
    📄 [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
    💡 **Progressive GAN** - Crescimento progressivo para alta resolução

13. **Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020)**
    *Analyzing and improving the image quality of stylegan*
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8110-8119).
    📄 [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
    💡 **StyleGAN2** - Melhorias de qualidade e remoção de artifacts

### VAEs Avançados

14. **Razavi, A., Van den Oord, A., & Vinyals, O. (2019)**
    *Generating diverse high-fidelity images with vq-vae-2*
    Advances in neural information processing systems, 32.
    📄 [arXiv:1906.00446](https://arxiv.org/abs/1906.00446)
    💡 **VQ-VAE-2** - Alta fidelidade com arquitetura hierárquica

15. **Kingma, D. P., & Welling, M. (2019)**
    *An introduction to variational autoencoders*
    Foundations and Trends in Machine Learning, 12(4), 307-392.
    📄 [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
    💡 Tutorial completo e atualizado sobre VAEs

---

## Métricas de Avaliação

16. **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017)**
    *GANs trained by a two time-scale update rule converge to a local nash equilibrium*
    Advances in neural information processing systems, 30.
    📄 [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
    💡 **FID (Fréchet Inception Distance)** - Métrica padrão para avaliar GANs

17. **Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016)**
    *Improved techniques for training gans*
    Advances in neural information processing systems, 29.
    📄 [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
    💡 **Inception Score** e técnicas de treinamento

---

## Livros Recomendados

📖 **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**
*Deep learning*
MIT press.
- Capítulo 20: Deep Generative Models
- [Livro Online](https://www.deeplearningbook.org/)

📖 **Murphy, K. P. (2022)**
*Probabilistic machine learning: An introduction*
MIT press.
- Capítulo 20: Variational Inference
- [Livro Online](https://probml.github.io/pml-book/book1.html)

📖 **Murphy, K. P. (2023)**
*Probabilistic machine learning: Advanced topics*
MIT press.
- Capítulo 25: Deep Generative Models
- [Livro Online](https://probml.github.io/pml-book/book2.html)

---

## Recursos Online

### Artigos Distill.pub
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/)

### PyTorch Tutorials
- [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [VAE Tutorial](https://github.com/pytorch/examples/tree/main/vae)

### Implementações de Referência
- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) - Implementações de diversas arquiteturas GAN
- [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) - Implementação oficial StyleGAN2
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Estado da arte em geração

---

## Estado da Arte (2023-2024)

18. **Karras, T., Aittala, M., Laine, S., Härkönen, E., Hellsten, J., Lehtinen, J., & Aila, T. (2021)**
    *Alias-free generative adversarial networks*
    Advances in Neural Information Processing Systems, 34, 852-863.
    📄 [arXiv:2106.12423](https://arxiv.org/abs/2106.12423)
    💡 **StyleGAN3** - Rotação e translação equivariantes

19. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022)**
    *High-resolution image synthesis with latent diffusion models*
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10684-10695).
    📄 [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
    💡 **Stable Diffusion** - Diffusion models em espaço latente

20. **Ho, J., Jain, A., & Abbeel, P. (2020)**
    *Denoising diffusion probabilistic models*
    Advances in neural information processing systems, 33, 6840-6851.
    📄 [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
    💡 **DDPM** - Fundamentos de diffusion models

---

## Exercícios Propostos para Mestrado

### Nível Teórico

1. **Teoria dos Jogos**: Demonstre formalmente que o equilíbrio de Nash da GAN corresponde a p_g = p_data

2. **ELBO Derivation**: Derive completamente o ELBO para VAE, incluindo a forma fechada da KL divergence para distribuições Gaussianas

3. **Wasserstein Distance**: Prove que a distância de Wasserstein fornece gradientes úteis mesmo quando suportes de p_data e p_g não se sobrepõem

### Nível Implementação

4. **DCGAN**: Implemente uma DCGAN completa para CIFAR-10 e compare com a implementação MLP

5. **Conditional GAN**: Estenda o GAN do módulo para geração condicional (escolha da classe)

6. **β-VAE**: Implemente um β-VAE e experimente com diferentes valores de β. Visualize o efeito no espaço latente

### Nível Pesquisa

7. **Métricas de Avaliação**: Implemente FID (Fréchet Inception Distance) e IS (Inception Score) para avaliar seus modelos

8. **Mode Collapse**: Experimente técnicas para mitigar mode collapse (Unrolled GAN, Minibatch Discrimination) e compare quantitativamente

9. **Ablation Study**: Realize um estudo de ablação removendo componentes (BatchNorm, Dropout, LeakyReLU) e analise o impacto

---

## Links Úteis

- [GAN Lab - Visualização Interativa](https://poloclub.github.io/ganlab/)
- [This Person Does Not Exist](https://thispersondoesnotexist.com/) - Demonstração de StyleGAN
- [Two Minute Papers - GANs](https://www.youtube.com/watch?v=kSLJriaOumA)

---

**Última atualização**: Novembro 2024
**Curso**: Visão Computacional - Mestrado
**Professor**: Rodrigo Fapo
