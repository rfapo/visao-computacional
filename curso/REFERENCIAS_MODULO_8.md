# 📚 Referências Bibliográficas - Módulo 8: Vision Transformers e Atenção

> Este documento complementa o [Módulo 8: Vision Transformers e Mecanismos de Atenção](08_vision_transformers_atencao.ipynb)

---

## Papers Fundamentais

### Transformer e Attention

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017)**
   *Attention is all you need*
   Advances in neural information processing systems, 30.
   📄 [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   💡 **Paper original do Transformer** - Revolucionou NLP e posteriormente visão computacional

2. **Bahdanau, D., Cho, K., & Bengio, Y. (2014)**
   *Neural machine translation by jointly learning to align and translate*
   arXiv preprint arXiv:1409.0473.
   📄 [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
   💡 Primeira aplicação influente de attention mechanism

### Vision Transformers

3. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020)**
   *An image is worth 16x16 words: Transformers for image recognition at scale*
   arXiv preprint arXiv:2010.11929.
   📄 [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
   💡 **Vision Transformer (ViT)** - Transformers puros para visão computacional

4. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021)**
   *Swin transformer: Hierarchical vision transformer using shifted windows*
   Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).
   📄 [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
   💡 **Swin Transformer** - Hierarquia e shifted windows para eficiência

5. **Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021)**
   *Training data-efficient image transformers & distillation through attention*
   International conference on machine learning (pp. 10347-10357). PMLR.
   📄 [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
   💡 **DeiT** - Data-efficient image transformers

---

## Aplicações de Transformers em Visão

6. **Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020)**
   *End-to-end object detection with transformers*
   European conference on computer vision (pp. 213-229). Springer, Cham.
   📄 [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
   💡 **DETR** - Detection com transformers end-to-end

7. **Zheng, S., Lu, J., Zhao, H., Zhu, X., Luo, Z., Wang, Y., ... & Zhang, L. (2021)**
   *Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers*
   Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 6881-6890).
   📄 [arXiv:2012.15840](https://arxiv.org/abs/2012.15840)
   💡 **SETR** - Segmentation com transformers

8. **Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021)**
   *Vivit: A video vision transformer*
   Proceedings of the IEEE/CVF international conference on computer vision (pp. 6836-6846).
   📄 [arXiv:2103.15691](https://arxiv.org/abs/2103.15691)
   💡 **ViViT** - Video understanding com transformers

---

## Mecanismos de Atenção Avançados

9. **Wang, X., Girshick, R., Gupta, A., & He, K. (2018)**
   *Non-local neural networks*
   Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7794-7803).
   📄 [arXiv:1711.07971](https://arxiv.org/abs/1711.07971)
   💡 **Non-local blocks** - Self-attention para vídeos

10. **Hu, J., Shen, L., & Sun, G. (2018)**
    *Squeeze-and-excitation networks*
    Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    📄 [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
    💡 **SE-Net** - Channel attention mechanism

11. **Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018)**
    *Cbam: Convolutional block attention module*
    Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).
    📄 [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
    💡 **CBAM** - Spatial + Channel attention

---

## Variantes e Otimizações

12. **Yuan, L., Chen, Y., Wang, T., Yu, W., Shi, Y., Jiang, Z. H., ... & Yan, S. (2021)**
    *Tokens-to-token vit: Training vision transformers from scratch on imagenet*
    Proceedings of the IEEE/CVF international conference on computer vision (pp. 558-567).
    📄 [arXiv:2101.11986](https://arxiv.org/abs/2101.11986)
    💡 **T2T-ViT** - Tokenização progressiva

13. **Wang, W., Xie, E., Li, X., Fan, D. P., Song, K., Liang, D., ... & Shao, L. (2021)**
    *Pyramid vision transformer: A versatile backbone for dense prediction without convolutions*
    Proceedings of the IEEE/CVF international conference on computer vision (pp. 568-578).
    📄 [arXiv:2102.12122](https://arxiv.org/abs/2102.12122)
    💡 **PVT** - Pyramid Vision Transformer

14. **Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022)**
    *A convnet for the 2020s*
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11976-11986).
    📄 [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
    💡 **ConvNeXt** - Modernização de CNNs inspirada em ViTs

---

## Análise Teórica

15. **Cordonnier, J. B., Loukas, A., & Jaggi, M. (2019)**
    *On the relationship between self-attention and convolutional layers*
    arXiv preprint arXiv:1911.03584.
    📄 [arXiv:1911.03584](https://arxiv.org/abs/1911.03584)
    💡 Análise teórica: self-attention vs convolution

16. **Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. (2021)**
    *Do vision transformers see like convolutional neural networks?*
    Advances in Neural Information Processing Systems, 34, 12116-12128.
    📄 [arXiv:2108.08810](https://arxiv.org/abs/2108.08810)
    💡 Análise comparativa: ViT vs CNN

---

## Surveys e Tutoriais

17. **Khan, S., Naseer, M., Hayat, M., Zamir, S. W., Khan, F. S., & Shah, M. (2022)**
    *Transformers in vision: A survey*
    ACM computing surveys (CSUR), 54(10s), 1-41.
    📄 [arXiv:2101.01169](https://arxiv.org/abs/2101.01169)
    💡 **Survey completo** sobre transformers em visão

18. **Han, K., Wang, Y., Chen, H., Chen, X., Guo, J., Liu, Z., ... & Tao, D. (2022)**
    *A survey on vision transformer*
    IEEE transactions on pattern analysis and machine intelligence, 45(1), 87-110.
    📄 [arXiv:2012.12556](https://arxiv.org/abs/2012.12556)
    💡 Survey técnico sobre ViTs

19. **Lin, T., Wang, Y., Liu, X., & Qiu, X. (2022)**
    *A survey of transformers*
    AI Open, 3, 111-132.
    📄 [arXiv:2106.04554](https://arxiv.org/abs/2106.04554)
    💡 Survey geral sobre arquiteturas transformer

---

## Livros Recomendados

📖 **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**
*Deep learning*
MIT press.
- Capítulo 10: Sequence Modeling (RNNs e Attention)
- [Livro Online](https://www.deeplearningbook.org/)

📖 **Tunstall, L., von Werra, L., & Wolf, T. (2022)**
*Natural Language Processing with Transformers*
O'Reilly Media.
- Aplicável também para visão
- [GitHub](https://github.com/nlp-with-transformers/notebooks)

📖 **Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023)**
*Dive into Deep Learning*
Cambridge University Press.
- Capítulo 11: Attention Mechanisms and Transformers
- [Livro Online](https://d2l.ai/)

---

## Recursos Online

### Artigos Distill.pub
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- [Visualizing Attention](https://distill.pub/2020/circuits/visualizing-weights/)

### PyTorch/Hugging Face
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Timm - PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

### Implementações de Referência
- [Google Research - Vision Transformer](https://github.com/google-research/vision_transformer)
- [Microsoft - Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [Facebook Research - DeiT](https://github.com/facebookresearch/deit)

---

## Papers Recentes (2023-2024)

20. **Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., ... & Houlsby, N. (2023)**
    *Scaling vision transformers to 22 billion parameters*
    ICML 2023.
    📄 [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
    💡 Escalabilidade de ViTs

21. **Bai, J., Yuan, L., Xia, S., Yan, S., Li, Z., & Liu, W. (2023)**
    *Improving Vision Transformers by Revisiting High-frequency Components*
    ECCV 2024.
    📄 [arXiv:2403.18234](https://arxiv.org/abs/2403.18234)
    💡 Análise de componentes de frequência em ViTs

---

## Exercícios Propostos para Mestrado

### Nível Teórico

1. **Complexidade Computacional**: Derive a complexidade de self-attention O(n²) em termos do comprimento da sequência. Compare com convoluções O(k²·n²)

2. **Positional Encoding**: Prove que positional encodings sinusoidais permitem que o modelo aprenda a atender a posições relativas

3. **Multi-Head Attention**: Analise matematicamente por que múltiplas cabeças de atenção aumentam a capacidade expressiva

### Nível Implementação

4. **ViT from Scratch**: Implemente um Vision Transformer completo do zero para classificação em CIFAR-10

5. **Positional Embeddings**: Experimente com diferentes tipos de positional embeddings (learned, sinusoidal, relative)

6. **Visualization**: Implemente visualização de attention maps para diferentes cabeças e camadas

### Nível Pesquisa

7. **Data Efficiency**: Compare ViT e ResNet em regimes de poucos dados. Implemente técnicas de data augmentation

8. **Attention Analysis**: Analise quantitativamente o que diferentes cabeças de atenção aprendem (local vs global patterns)

9. **Transfer Learning**: Avalie transferibilidade de features de ViT pré-treinado para downstream tasks

---

## Links Úteis

- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Implementação comentada linha por linha
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Explicação visual excelente
- [Attention Is All You Need - Video](https://www.youtube.com/watch?v=iDulhoQ2pro) - Explicação do paper

---

**Última atualização**: Novembro 2024
**Curso**: Visão Computacional - Mestrado
**Professor**: Rodrigo Fapo
