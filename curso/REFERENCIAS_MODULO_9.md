# 📚 Referências Bibliográficas - Módulo 9: Foundation Models para Visão Computacional

> Este documento complementa o [Módulo 9: Foundation Models para Visão Computacional](09_foundation_models_visao_computacional.ipynb)

---

## Foundation Models - Conceitos Fundamentais

1. **Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... & Liang, P. (2021)**
   *On the opportunities and risks of foundation models*
   arXiv preprint arXiv:2108.07258.
   📄 [arXiv:2108.07258](https://arxiv.org/abs/2108.07258)
   💡 **Paper seminal** definindo Foundation Models e suas implicações

2. **Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020)**
   *Language models are few-shot learners*
   Advances in neural information processing systems, 33, 1877-1901.
   📄 [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
   💡 **GPT-3** - Demonstrou poder de large language models e few-shot learning

---

## CLIP - Vision-Language Models

3. **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021)**
   *Learning transferable visual models from natural language supervision*
   International conference on machine learning (pp. 8748-8763). PMLR.
   📄 [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
   💡 **CLIP** - Alinhamento visão-linguagem em larga escala, zero-shot transfer

4. **Jia, C., Yang, Y., Xia, Y., Chen, Y. T., Parekh, Z., Pham, H., ... & Chen, H. (2021)**
   *Scaling up visual and vision-language representation learning with noisy text supervision*
   International Conference on Machine Learning (pp. 4904-4916). PMLR.
   📄 [arXiv:2102.05918](https://arxiv.org/abs/2102.05918)
   💡 **ALIGN** - Alternativa ao CLIP com 1.8B pares imagem-texto

5. **Li, J., Li, D., Savarese, S., & Hoi, S. (2023)**
   *Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models*
   arXiv preprint arXiv:2301.12597.
   📄 [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
   💡 **BLIP-2** - Q-Former para eficiência com modelos congelados

---

## DALL-E e Text-to-Image Generation

6. **Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., ... & Sutskever, I. (2021)**
   *Zero-shot text-to-image generation*
   International Conference on Machine Learning (pp. 8821-8831). PMLR.
   📄 [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)
   💡 **DALL-E** - Primeira versão, VQ-VAE + Transformer autoregressive

7. **Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022)**
   *Hierarchical text-conditional image generation with clip latents*
   arXiv preprint arXiv:2204.06125.
   📄 [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)
   💡 **DALL-E 2** - Unificar conceitos, prior diffusion + decoder diffusion

8. **Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., ... & Norouzi, M. (2022)**
   *Photorealistic text-to-image diffusion models with deep language understanding*
   Advances in Neural Information Processing Systems, 35, 36479-36494.
   📄 [arXiv:2205.11487](https://arxiv.org/abs/2205.11487)
   💡 **Imagen** - Diffusion com T5 encoder, fotorrealismo extremo

9. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022)**
   *High-resolution image synthesis with latent diffusion models*
   Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10684-10695).
   📄 [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
   💡 **Stable Diffusion** - Latent diffusion, open-source

---

## GPT-4V e Multimodal LLMs

10. **OpenAI. (2023)**
    *GPT-4 technical report*
    arXiv preprint arXiv:2303.08774.
    📄 [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
    💡 **GPT-4** incluindo capacidades visuais (GPT-4V)

11. **Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2024)**
    *Visual instruction tuning*
    Advances in neural information processing systems, 36.
    📄 [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
    💡 **LLaVA** - Visual instruction tuning, multimodal conversations

12. **Alayrac, J. B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., ... & Simonyan, K. (2022)**
    *Flamingo: a visual language model for few-shot learning*
    Advances in Neural Information Processing Systems, 35, 23716-23736.
    📄 [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
    💡 **Flamingo** - Few-shot VLM com cross-attention

---

## Segment Anything e Universal Models

13. **Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023)**
    *Segment anything*
    Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4015-4026).
    📄 [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
    💡 **SAM** - Segmentação universal promptable, 1B+ masks, 11M imagens

14. **Zou, X., Yang, J., Zhang, H., Li, F., Li, L., Wang, J., ... & Gao, J. (2023)**
    *Segment everything everywhere all at once*
    arXiv preprint arXiv:2304.06718.
    📄 [arXiv:2304.06718](https://arxiv.org/abs/2304.06718)
    💡 **SEEM** - Extensão do SAM com diferentes tipos de prompts

---

## Diffusion Models

15. **Ho, J., Jain, A., & Abbeel, P. (2020)**
    *Denoising diffusion probabilistic models*
    Advances in neural information processing systems, 33, 6840-6851.
    📄 [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
    💡 **DDPM** - Fundamentos de diffusion models

16. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020)**
    *Score-based generative modeling through stochastic differential equations*
    arXiv preprint arXiv:2011.13456.
    📄 [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
    💡 **Score-based models** - Framework unificado para diffusion

17. **Dhariwal, P., & Nichol, A. (2021)**
    *Diffusion models beat gans on image synthesis*
    Advances in neural information processing systems, 34, 8780-8794.
    📄 [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
    💡 Demonstrou que diffusion models > GANs em qualidade

---

## Contrastive Learning

18. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020)**
    *A simple framework for contrastive learning of visual representations*
    International conference on machine learning (pp. 1597-1607). PMLR.
    📄 [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
    💡 **SimCLR** - Self-supervised contrastive learning

19. **He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020)**
    *Momentum contrast for unsupervised visual representation learning*
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729-9738).
    📄 [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
    💡 **MoCo** - Momentum contrastive learning

---

## Datasets em Larga Escala

20. **Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., ... & Jitsev, J. (2022)**
    *Laion-5b: An open large-scale dataset for training next generation image-text models*
    Advances in Neural Information Processing Systems, 35, 25278-25294.
    📄 [arXiv:2210.08402](https://arxiv.org/abs/2210.08402)
    💡 **LAION-5B** - 5.85 bilhões de pares imagem-texto, open-source

---

## Controle e Aplicações

21. **Zhang, L., Rao, A., & Agrawala, M. (2023)**
    *Adding conditional control to text-to-image diffusion models*
    Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3836-3847).
    📄 [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
    💡 **ControlNet** - Controle fino espacial em diffusion models

22. **Mou, C., Wang, X., Xie, L., Zhang, J., Qi, Z., Shan, Y., & Qie, X. (2023)**
    *T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models*
    arXiv preprint arXiv:2302.08453.
    📄 [arXiv:2302.08453](https://arxiv.org/abs/2302.08453)
    💡 **T2I-Adapter** - Lightweight adapters para controle

---

## Ética e Segurança

23. **Bianchi, F., Kalluri, P., Durmus, E., Ladhak, F., Cheng, M., Nozza, D., ... & Zou, J. (2023)**
    *Easily accessible text-to-image generation amplifies demographic stereotypes at large scale*
    Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (pp. 1493-1504).
    📄 [arXiv:2211.03759](https://arxiv.org/abs/2211.03759)
    💡 Análise de viés em modelos text-to-image

24. **Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., ... & Wallace, E. (2023)**
    *Extracting training data from diffusion models*
    32nd USENIX Security Symposium (USENIX Security 23) (pp. 5253-5270).
    📄 [arXiv:2301.13188](https://arxiv.org/abs/2301.13188)
    💡 Privacy concerns em diffusion models

25. **Tolosana, R., Vera-Rodriguez, R., Fierrez, J., Morales, A., & Ortega-Garcia, J. (2020)**
    *Deepfakes and beyond: A survey of face manipulation and fake detection*
    Information Fusion, 64, 131-148.
    📄 [arXiv:2001.00179](https://arxiv.org/abs/2001.00179)
    💡 Survey sobre deepfakes e detecção

---

## Surveys

26. **Zhou, C., Li, Q., Li, C., Yu, J., Liu, Y., Wang, G., ... & Sun, L. (2023)**
    *A comprehensive survey on pretrained foundation models: A history from bert to chatgpt*
    arXiv preprint arXiv:2302.09419.
    📄 [arXiv:2302.09419](https://arxiv.org/abs/2302.09419)
    💡 Survey completo sobre foundation models

27. **Yang, J., Jin, H., Tang, R., Han, X., Feng, Q., Jiang, H., ... & Hu, X. (2024)**
    *Harnessing the power of llms in practice: A survey on chatgpt and beyond*
    ACM Transactions on Knowledge Discovery from Data, 18(6), 1-32.
    📄 [arXiv:2304.13712](https://arxiv.org/abs/2304.13712)
    💡 Survey sobre LLMs práticos

---

## Livros Recomendados

📖 **Murphy, K. P. (2023)**
*Probabilistic machine learning: Advanced topics*
MIT press.
- Capítulo sobre Diffusion Models
- [Livro Online](https://probml.github.io/pml-book/book2.html)

📖 **Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., ... & Wen, J. R. (2023)**
*A survey of large language models*
arXiv preprint arXiv:2303.18223.
- Capítulos sobre multimodalidade
- [Paper](https://arxiv.org/abs/2303.18223)

---

## Recursos Online

### OpenAI
- [CLIP: Connecting Text and Images](https://openai.com/blog/clip/)
- [DALL-E 2: Extending Creativity](https://openai.com/dall-e-2)
- [GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- [API Documentation](https://platform.openai.com/docs)

### Hugging Face
- [Diffusers Library](https://github.com/huggingface/diffusers)
- [Transformers - Multimodal Models](https://huggingface.co/docs/transformers/main/en/model_doc/clip)
- [CLIP Space](https://huggingface.co/spaces/openai/clip)
- [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1)

### Stability AI
- [Stable Diffusion](https://stability.ai/stable-diffusion)
- [Technical Report](https://stability.ai/research/stable-diffusion-3-medium)
- [DreamStudio](https://beta.dreamstudio.ai/)

### Meta AI
- [Segment Anything](https://segment-anything.com/)
- [Demo Interactive](https://segment-anything.com/demo)
- [GitHub](https://github.com/facebookresearch/segment-anything)

### Google Research
- [Imagen](https://imagen.research.google/)
- [Parti](https://parti.research.google/)

---

## Código e Implementações

### Principais Bibliotecas
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Implementação open-source de CLIP
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

### Ferramentas
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Interface para Stable Diffusion
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - WebUI popular
- [InvokeAI](https://github.com/invoke-ai/InvokeAI) - Stable Diffusion toolkit

---

## Exercícios Propostos para Mestrado

### Nível Teórico

1. **Contrastive Learning**: Derive matematicamente a InfoNCE loss usada em CLIP e prove suas propriedades

2. **Diffusion Process**: Derive o processo de difusão forward e reverse em diffusion models

3. **Zero-Shot Transfer**: Analise teoricamente por que CLIP permite zero-shot transfer eficaz

### Nível Implementação

4. **CLIP Fine-tuning**: Fine-tune CLIP em um dataset específico de domínio e avalie transferência

5. **Text-to-Image**: Implemente pipeline completo de geração usando Stable Diffusion via Diffusers

6. **Prompt Engineering**: Experimente com diferentes estratégias de prompting para DALL-E/Midjourney

### Nível Pesquisa

7. **Bias Analysis**: Conduza análise sistemática de viés em modelos text-to-image (gênero, raça, profissões)

8. **Few-Shot Learning**: Compare few-shot performance de CLIP vs modelos supervisionados tradicionais

9. **Multimodal Alignment**: Investigue alinhamento entre modalidades em diferentes foundation models

---

## Links Úteis

- [LAION](https://laion.ai/) - Large-scale AI Open Network
- [Papers With Code - Foundation Models](https://paperswithcode.com/task/foundation-models)
- [The AI Index Report](https://aiindex.stanford.edu/)
- [Segment Anything Demo](https://segment-anything.com/demo)

---

**Última atualização**: Novembro 2024
**Curso**: Visão Computacional - Mestrado
**Professor**: Rodrigo Fapo
