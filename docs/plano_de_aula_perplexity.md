Agora vou criar os planos de aula detalhados para os dois dias, incorporando todo o conteúdo pesquisado.

## Plano de Aula - Introdução à Visão Computacional

### **Aula 01 - Fundamentos e Técnicas Clássicas (7 horas)**

#### **Módulo 1: Introdução e História da Visão Computacional (1h30)**
*09:00 - 10:30*

**Objetivos:**
- Compreender o contexto histórico e evolução da visão computacional
- Entender a importância da visão para inteligência artificial
- Conhecer os principais marcos da área

**Conteúdo:**

**1.1 O que é Visão Computacional?**[1][2]
- Definição: capacidade de máquinas "verem" e compreenderem imagens/vídeos
- Relação entre visão biológica (540 milhões de anos) e visão computacional
- Visão como pedra angular da inteligência artificial[2]

**1.2 História e Evolução**[3][4][5]
- Anos 1960-1980: Primeiros sistemas de reconhecimento de padrões
- Década de 1990: Métodos tradicionais (SIFT, HOG, filtros Gabor)
- 2009: ImageNet - o ponto de virada[6][4][7]
  - Criação por Fei-Fei Li
  - 14+ milhões de imagens anotadas manualmente
  - Hierarquia baseada em WordNet
  - Mais de 20.000 categorias
- 2012: AlexNet e a Revolução do Deep Learning[8][5]
  - Vitória no ILSVRC com margem de 9,8%
  - Primeira aplicação prática bem-sucedida de Deep Learning
  - Taxa de erro < 25%
  - CNN de 8 camadas

**1.3 O Papel do CS231n de Stanford**[9][10][1][3][2]
- Curso ministrado por Fei-Fei Li, Jiajun Wu, Andrei Karpathy
- Referência mundial em visão computacional com deep learning
- Foco em CNNs e aplicações práticas
- Conceitos que moldaram a indústria atual

**Atividade Prática (15 min):**
- Discussão: Como a visão computacional impacta o dia a dia dos alunos?
- Exemplos de aplicações: smartphones, redes sociais, carros autônomos

***

#### **Intervalo (20 min)**
*10:30 - 10:50*

***

#### **Módulo 2: Processamento Digital de Imagem - Fundamentos (2h10)**
*10:50 - 13:00*

**Objetivos:**
- Compreender os fundamentos matemáticos de imagens digitais
- Dominar operações básicas de processamento de imagem
- Aplicar filtros e transformações essenciais

**Conteúdo:**

**2.1 Fundamentos de Gonzalez & Woods**[11][12][13][14][15]
- Representação de imagens digitais
  - Matrizes de pixels
  - Profundidade de bits (8-bit, 16-bit, RGB)
  - Resolução espacial vs. resolução de intensidade
  
**2.2 Operações Fundamentais**[13][14][11]
- Transformações de intensidade
  - Negativo de imagem
  - Ajuste de contraste
  - Equalização de histograma
  - Correção gama
- Filtragem espacial
  - Filtros de suavização (média, Gaussiano)
  - Filtros de aguçamento (Laplaciano, Sobel)
  - Detecção de bordas (Canny, Prewitt)
- Operações morfológicas
  - Erosão e dilatação
  - Abertura e fechamento
  - Hit-or-miss

**2.3 Transformações de Domínio**[14][13]
- Transformada de Fourier
  - Domínio de frequência
  - Filtragem passa-baixa e passa-alta
- Aplicações práticas
  - Remoção de ruído
  - Detecção de padrões periódicos

**Demonstração Prática (30 min):**
- Utilização de OpenCV/Python para aplicar filtros
- Visualização de resultados com diferentes operações
- Comparação entre técnicas clássicas

***

#### **Almoço/Intervalo (1h)**
*13:00 - 14:00*

***

#### **Módulo 3: Deep Learning para Visão Computacional (2h)**
*14:00 - 16:00*

**Objetivos:**
- Compreender arquiteturas de CNNs
- Entender o funcionamento de redes neurais convolucionais
- Conhecer arquiteturas clássicas e sua evolução

**Conteúdo:**

**3.1 Redes Neurais Convolucionais (CNNs)**[16][17][18][19][20]

**Arquitetura Básica:**
- **Camadas Convolucionais**[17][16]
  - Filtros/kernels aprendíveis
  - Campo receptivo local
  - Compartilhamento de parâmetros
  - Operação de convolução matemática
  - Feature maps e detecção hierárquica de padrões
  
- **Camadas de Pooling**[18][19][17]
  - Max pooling vs. Average pooling
  - Redução de dimensionalidade espacial
  - Invariância a translação
  - Stride e receptive field

- **Camadas Totalmente Conectadas**[19][17][18]
  - Transformação de features em predições
  - Vetorização das feature maps
  - Classificação final

- **Funções de Ativação**[19]
  - ReLU (Rectified Linear Unit)
  - Introdução de não-linearidade
  - Problemas de dying ReLU
  - Alternativas: Leaky ReLU, PReLU, ELU

**3.2 Arquiteturas Clássicas**[21][22][23][24][17]

**AlexNet (2012)**[5][8]
- 8 camadas (5 conv + 3 FC)
- ReLU activation
- Dropout
- Data augmentation
- GPU training
- 60 milhões de parâmetros

**VGG (2014)**[23][24][25][17]
- Arquitetura uniforme com blocos 3×3
- VGG16: 16 camadas com pesos
- VGG19: 19 camadas
- 138 milhões de parâmetros
- Simples e fácil de entender
- Convergência rápida para fine-tuning[25]
- Excelente para transfer learning[24]

**ResNet (2015)**[22][23][24][25]
- Introdução de conexões residuais (skip connections)
- Solução para o problema de degradação
- Identity mapping
- ResNet-18, ResNet-50, ResNet-101, ResNet-152
- Até 152 camadas de profundidade
- ResNet-50: ~25M parâmetros (apenas 36% do VGG14)[25]
- Melhor eficiência computacional[23][25]
- Estado da arte em múltiplas tarefas[24]

**Comparação VGG vs. ResNet:**[24][25]
- ResNet é mais eficiente em parâmetros
- VGG converge mais rápido em fine-tuning[25]
- ResNet tem melhor desempenho geral[24]
- VGG é mais simples para implementação[24]
- Escolha depende da aplicação específica

**Demonstração Prática (20 min):**
- Visualização de arquiteturas com Netron ou TensorBoard
- Análise de feature maps em diferentes camadas
- Comparação de complexidade computacional

***

#### **Intervalo (20 min)**
*16:00 - 16:20*

***

#### **Módulo 4: Transfer Learning e Aplicações Práticas (40 min)**
*16:20 - 17:00*

**Objetivos:**
- Compreender o conceito de transfer learning
- Aplicar modelos pré-treinados
- Entender quando usar fine-tuning

**Conteúdo:**

**4.1 Transfer Learning em Visão Computacional**[26][27][28][29]

**Conceitos Fundamentais:**[28][29][26]
- Utilização de conhecimento de tarefas similares
- Modelos pré-treinados em ImageNet
- Feature extraction vs. Fine-tuning
- Quando usar transfer learning[29]
  - Datasets pequenos
  - Problemas de domínio similar
  - Recursos computacionais limitados
  - Economia de tempo de treinamento

**Estratégias de Transfer Learning:**[27][26][28]

**1. Feature Extraction**[26][27]
- Congelar camadas convolucionais
- Treinar apenas camadas finais
- Útil com datasets muito pequenos
- Reduz tempo de treinamento

**2. Fine-tuning**[27][28][26]
- Ajustar pesos de camadas profundas
- Manter camadas iniciais congeladas ou com learning rate menor
- Melhor performance com dados suficientes
- Adaptar representações ao novo domínio

**3. Pre-training e Initialization**[27]
- Usar pesos do ImageNet como inicialização
- Treinar completamente no novo dataset
- Melhor ponto de partida que inicialização aleatória

**Benefícios do Transfer Learning:**[28]
- Economia de tempo (até 10x mais rápido)
- Requer menos dados (5-10x menos)
- Melhora acurácia (especialmente em datasets pequenos)
- Facilita adaptação a novos domínios[29]

**Aplicações Práticas no Mercado:**[30][31][32][33][34]

**Saúde:**[31][34]
- Detecção de câncer em imagens médicas
- Análise de raio-X, CT e MRI
- Identificação de células cancerígenas
- Segmentação de órgãos e tumores

**Indústria 4.0:**[32][33][30]
- Inspeção de qualidade automatizada
- Detecção de defeitos em linhas de produção
- Controle de qualidade 24/7
- Monitoramento de processos industriais
- Robôs colaborativos com visão

**Varejo:**[33][31]
- Análise de comportamento do consumidor
- Sistemas de checkout automático
- Gestão de inventário
- Prevenção de perdas

**Agricultura:**[34][31]
- Monitoramento de saúde das culturas
- Detecção de pragas e doenças
- Colheita automatizada
- Otimização de irrigação

**Transporte:**[31][33][34]
- Veículos autônomos
- Sistemas de assistência ao motorista
- Detecção de sinais de trânsito
- Monitoramento de infraestrutura

**Demonstração (10 min):**
- Exemplo de transfer learning com modelo pré-treinado
- Comparação: treino do zero vs. transfer learning
- Discussão sobre casos de uso da turma

***

**Encerramento do Dia 1 (10 min)**
*17:00 - 17:10*
- Revisão dos conceitos principais
- Preparação para o Dia 2
- Perguntas e respostas

---

### **Aula 02 - Tarefas de Visão Computacional e Foundation Models (8 horas)**

#### **Módulo 5: Tarefas Fundamentais em Visão Computacional (2h)**
*09:00 - 11:00*

**Objetivos:**
- Dominar as principais tarefas de visão computacional
- Compreender diferenças entre classificação, detecção e segmentação
- Conhecer arquiteturas específicas para cada tarefa

**Conteúdo:**

**5.1 Classificação de Imagens**[35][36][9]
- **Definição:** Atribuir uma ou mais labels a uma imagem completa
- **Arquiteturas:** VGG, ResNet, EfficientNet
- **Métricas:** Accuracy, Precision, Recall, F1-Score
- **Aplicações:** 
  - Reconhecimento de objetos
  - Classificação de documentos
  - Diagnóstico médico por imagem

**5.2 Detecção de Objetos**[37][38][39][40][41][35][21]

**Conceito:**[35]
- Identificar e localizar múltiplos objetos em uma imagem
- Bounding boxes + class labels
- Mais complexo que classificação

**Arquiteturas Principais:**

**YOLO (You Only Look Once)**[39][40][41][42][21]
- **Arquitetura:**[40][41][21]
  - Single-stage detector
  - Grid-based approach (S×S grid)
  - Cada célula prediz B bounding boxes
  - Predição de class probabilities por célula
  - Backbone CNN para extração de features
  
- **Funcionamento:**[21][39][40]
  - Grid Creation: imagem dividida em SxS grid
  - Bounding Box Prediction: cada célula prediz boxes com confidence scores
  - Class Probability Prediction: probabilidades condicionais de classe
  - Non-Maximum Suppression (NMS): elimina boxes redundantes
  
- **Evolução:**[41][40][21]
  - YOLOv1 (2015): Conceito original
  - YOLOv2/v3: Melhorias em accuracy
  - YOLOv4/v5: State-of-the-art em speed/accuracy
  - YOLOv8: Arquitetura moderna otimizada
  - YOLOv11: Últimas inovações[41]

- **Vantagens:**[42][21]
  - Extremamente rápido (real-time)
  - Unified detection pipeline
  - Bom para aplicações práticas
  - Trade-off ajustável entre velocidade e precisão

**Outras Arquiteturas:**
- Faster R-CNN: Two-stage detector, maior precisão
- SSD (Single Shot Detector): Balance speed/accuracy
- RetinaNet: Focal loss para imbalanced datasets

**Métricas de Avaliação:**[35]
- IoU (Intersection over Union)
- mAP (mean Average Precision)
- Precision-Recall curves
- FPS (Frames Per Second) para tempo real

**5.3 Segmentação de Imagens**[38][43][44][45][46][47][37][35]

**Tipos de Segmentação:**[43][38]

**Semantic Segmentation:**[38][43]
- Classificação pixel-a-pixel
- Mesma label para objetos da mesma classe
- Não diferencia instâncias individuais
- Aplicações: cenas de rua, imagens médicas

**Instance Segmentation:**[43][38]
- Diferencia objetos individuais
- Única label por instância
- Mais granular que semantic
- Aplicações: contagem de objetos, rastreamento

**Panoptic Segmentation:**[43]
- Combinação de semantic + instance
- Máxima granularidade
- Estado da arte em compreensão de cena

**Arquitetura U-Net**[44][45][46][47][48]

**Estrutura:**[45][46][47][44]
- **Formato em U:** nome deriva da estrutura visual
- **Contracting Path (Encoder):**[44][45]
  - Convolutional blocks (3×3 conv + ReLU)
  - Max pooling (2×2) para downsampling
  - Dobra número de feature channels a cada pooling
  - Captura contexto e features abstratas
  
- **Expanding Path (Decoder):**[46][45][44]
  - Upsampling (transposed convolutions)
  - Concatenação com features do encoder (skip connections)
  - Convolutional blocks
  - Restaura resolução espacial original
  
- **Skip Connections:**[48][46][44]
  - Copiam feature maps do encoder para decoder
  - Preservam informação espacial de alta resolução
  - Facilitam aprendizado de segmentação precisa
  - Diferencial chave da arquitetura

**Características:**[47][46][44]
- Fully Convolutional Network (sem FC layers)
- 1×1 convolution final para mapear classes
- Pixel-wise softmax para predição
- Eficiente com poucos dados de treinamento
- Originalmente para segmentação biomédica

**Aplicações:**[45][46][47]
- Segmentação médica (órgãos, tumores, células)
- Difusão models (DALL-E, Stable Diffusion)
- Colorização de imagens
- Super-resolution
- Image inpainting

**Métricas de Avaliação:**[49][50][51]
- Pixel Accuracy
- Mean IoU (mIoU)
- Dice Coefficient
- Pixel-wise Cross Entropy

**Demonstração Prática (30 min):**
- Comparação visual: classificação vs. detecção vs. segmentação
- Análise de resultados de diferentes arquiteturas
- Discussão sobre quando usar cada abordagem

***

#### **Intervalo (20 min)**
*11:00 - 11:20*

***

#### **Módulo 6: OCR e Reconhecimento de Texto (1h10)**
*11:20 - 12:30*

**Objetivos:**
- Compreender técnicas de OCR
- Entender a contribuição do deep learning para OCR
- Aplicar OCR em cenários práticos

**Conteúdo:**

**6.1 OCR com Deep Learning**[52][53][54][55][56][57]

**Fundamentos:**[53][52]
- Optical Character Recognition
- Tradução de dados de imagem em texto
- Evolução: métodos tradicionais → deep learning

**Pipeline de OCR Moderno:**[54][55][56][53]

**1. Preprocessing:**[56]
- Tratamento de ruído, blur, skewness
- Binarização e normalização
- Correção de orientação
- Essencial para imagens não ideais

**2. Text Detection/Localization:**[57][56]
- Modelos: Mask-RCNN, EAST Text Detector, YOLOv5
- Criação de bounding boxes ao redor do texto
- Localização precisa de regiões de texto
- Separação de texto do background

**3. Text Recognition:**[55][53][56]
- **CRNN (Convolutional Recurrent Neural Network):**[53][55]
  - **CNN:**Feature extraction visual
    - Detecção de bordas e padrões
    - Eficiência em classificação
    - Extração hierárquica de features
  
  - **RNN:**Predição sequencial
    - LSTM cells para sequências variáveis
    - Evita vanishing gradient
    - Captura relações entre caracteres
    - Útil para handwriting e textos não estruturados

**Arquiteturas e Ferramentas:**[52][56][57][53]
- **Tesseract OCR Engine:**[52]
  - Open-source
  - Alta precisão em texto impresso
  - Integração com OpenCV
  
- **Deep Learning Models:**[57]
  - DBNet para detecção
  - LinkNet para segmentação
  - SAR (Show, Attend and Read) para reconhecimento

**Desafios:**[56][53]
- Texto irregular e "wild text"
- Condições de iluminação variáveis
- Diferentes fontes e tamanhos
- Orientações arbitrárias
- Oclusões parciais

**Aplicações Práticas:**[52]
- Digitalização de documentos
- Leitura de placas veiculares
- Extração de dados de faturas/recibos
- Tradução de texto em imagens
- Acessibilidade (leitura de texto para deficientes visuais)

**Demonstração Prática (20 min):**
- Exemplo com Tesseract/EasyOCR
- OCR em diferentes tipos de documento
- Análise de performance em cenários desafiadores

***

#### **Almoço (1h30)**
*12:30 - 14:00*

---

#### **Módulo 7: Vision Transformers e Mecanismos de Atenção (1h30)**
*14:00 - 15:30*

**Objetivos:**
- Compreender a arquitetura de transformers aplicada à visão
- Entender mecanismos de atenção
- Conhecer o estado da arte em arquiteturas

**Conteúdo:**

**7.1 Fundamentos dos Transformers**[58][59]

**Origem:**[59][58]
- "Attention Is All You Need" (2017)
- Originalmente para NLP
- Self-attention mechanism
- Abandono de recorrência e convoluções

**Mecanismo de Atenção:**[58][59]
- **Query (Q), Key (K), Value (V)**[58]
  - Q: o que estamos procurando
  - K: o que temos disponível
  - V: informação a ser recuperada
  - Inner product para calcular attention weights
  
- **Multi-Head Attention:**[59]
  - Múltiplos conjuntos de (W^Q, W^K, W^V)
  - Cada head aprende diferentes "relevâncias"
  - Captura múltiplos tipos de relações
  - Processamento paralelo
  - Concatenação dos outputs

**7.2 Vision Transformers (ViT)**[60][61][62][63][64][65][58]

**Arquitetura ViT:**[61][62][63][60]

**1. Image Patching:**[62][60][61]
- Imagem dividida em patches fixos (ex: 16×16)
- Cada patch flattened em vetor 1D
- Transformação 2D → sequência 1D
- Similar a tokens em NLP

**2. Linear Embedding:**[60][61]
- Projeção linear de cada patch
- Criação de patch embeddings
- Dimensão maior para representações ricas
- Sequência de embeddings como entrada

**3. Positional Encoding:**[61][62][60]
- Adição de informação posicional
- Transformers não entendem ordem naturalmente
- Codificação da localização espacial
- Preserva estrutura da imagem

**4. Transformer Encoder:**[62][60][61]
- Multi-head self-attention layers
- Feed-forward neural networks
- Layer normalization
- Residual connections
- Múltiplas camadas empilhadas

**5. Classification Token [CLS]:**[60][61]
- Token especial prepended à sequência
- Representa a imagem inteira
- Output usado para classificação
- Aprendido durante treinamento

**Diferenças: ViT vs. CNN:**[63][64][62]

**CNNs:**
- Inductive bias local (convolução)
- Hierarquia de features (baixo → alto nível)
- Eficiente com dados limitados
- Melhor para edge devices
- Interpretação mais intuitiva

**ViTs:**[64][63]
- Atenção global desde início
- Captura long-range dependencies
- Requer mais dados de treinamento
- Melhor com datasets grandes (ImageNet-21k)
- Escalabilidade superior

**Performance:**[63][64][62]
- ViT inferior a CNNs em datasets pequenos/médios
- ViT supera CNNs em datasets massivos
- Trade-off: dados vs. capacidade
- Modelos híbridos combinam vantagens

**7.3 Aplicações de Atenção em Visão**[66][67][68][58]

**Self-Attention Effects:**[68][66]
- Grouping baseado em similaridade visual
- Perceptual grouping, não apenas atenção
- Feed-forward (diferente de atenção humana)
- Captura relações entre todas as partes da imagem

**Convolutional Self-Attention (CSA):**[67]
- Emulação de atenção com convoluções
- Eficiência em GPUs otimizadas
- Receptive field global
- Relational encoding through rotations

**Modelos Híbridos:**[64][63]
- Swin Transformer: sliding windows
- DeiT: Data-efficient image transformers
- RT-DETR: CNN backbone + Transformer
- Best of both worlds

**Demonstração (15 min):**
- Visualização de attention maps
- Comparação ViT vs. CNN em mesma tarefa
- Análise de interpretabilidade

***

#### **Intervalo (20 min)**
*15:30 - 15:50*

***

#### **Módulo 8: Foundation Models para Visão Computacional (2h)**
*15:50 - 17:50*

**Objetivos:**
- Compreender o conceito de foundation models
- Utilizar APIs de visão (OpenAI, Gemini)
- Aplicar modelos multimodais em projetos práticos

**Conteúdo:**

**8.1 Foundation Models: Conceito e Evolução**[69][70][71][72][73]

**Definição:**[73]
- Modelos pré-treinados em datasets massivos
- Capacidade multimodal (texto, imagem, áudio, vídeo)
- Fine-tuning para tarefas específicas
- Self-supervised learning

**Características:**[73]
- Treinamento em dados diversos
- Transferência para múltiplos domínios
- Emergência de capacidades não explicitamente treinadas
- Escalabilidade

**8.2 GPT-4V (GPT-4 Vision)**[70][71][74][75][76][77][69]

**Capacidades:**[76][77]
- Análise visual de imagens
- Responder perguntas sobre imagens
- Detecção e análise de objetos
- Interpretação de dados visuais (gráficos, charts)
- Leitura de texto em imagens (OCR integrado)
- Raciocínio visual complexo

**Arquitetura:**[77]
- Baseado em GPT-4 + capacidades visuais
- Não é um modelo diferente, mas GPT-4 aumentado
- Mesmo desempenho em tarefas de texto
- Capabilities adicionadas, não substituídas

**Uso via API:**[74][75][76]

**Setup Básico:**
```python
import openai
from openai import OpenAI

client = OpenAI(api_key="sua-chave-api")

response = client.chat.completions.create(
    model="gpt-4-vision-preview",  # ou gpt-4o
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "O que você vê nesta imagem?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://exemplo.com/imagem.jpg",
                        "detail": "high"  # ou "low", "auto"
                    }
                }
            ]
        }
    ],
    max_tokens=512
)

print(response.choices[0].message.content)
```

**Parâmetros Importantes:**[75][74]
- `model`: gpt-4-vision-preview ou gpt-4o
- `detail`: controla resolução de processamento ("low", "high", "auto")
- `max_tokens`: limite de resposta (default baixo pela OpenAI)
- `temperature`: controle de aleatoriedade
- Suporte a múltiplas imagens no mesmo prompt

**Casos de Uso Práticos:**[76]
- Análise de documentos e faturas
- Descrição automática de imagens
- Detecção de anomalias visuais
- Suporte visual para aplicações
- Acessibilidade (descrição para deficientes visuais)

**8.3 Google Gemini Vision**[71][78][79][80][81][82][69][73]

**Modelos Disponíveis:**[82][73]
- **Gemini 1.5 Pro:** Análise complexa, contexto longo (2M tokens)
- **Gemini 1.5 Flash / Flash-8B:** Execução rápida, menor latência
- **Gemini Ultra:** Tarefas avançadas

**Capacidades Multimodais:**[82][73]
- Processamento de texto, imagens, áudio, vídeo, código
- Context window massivo (até 2 milhões de tokens)
- Suporte a PDFs longos (1000+ páginas)[80][82]
- Vídeos até 90 minutos[80]
- Cross-modal tasks (legendagem, VQA)

**Características Distintivas:**[69][71]
- Respostas detalhadas com imagens e links[71]
- Melhor em scene understanding[70]
- Integração com ecossistema Google

**Setup e Uso:**[78][79][80][82]

**1. Configuração Inicial:**[79][78]
```python
# Instalação
pip install google-generativeai

# Importação e autenticação
import google.generativeai as genai

genai.configure(api_key="SUA_API_KEY")
```

**2. Criação de API Key:**[79]
- Acessar Google AI Studio
- Criar/importar projeto Google Cloud
- Habilitar Vertex AI API
- Gerar API key no dashboard

**3. Uso Básico:**[80][82]
```python
# Configurar modelo
model = genai.GenerativeModel('gemini-1.5-pro')

# Com imagem local
import PIL.Image
img = PIL.Image.open('imagem.jpg')

response = model.generate_content([
    "Descreva esta imagem em detalhes",
    img
])

print(response.text)

# Com PDF
import pathlib
pdf_file = genai.upload_file(
    path="documento.pdf",
    display_name="Meu PDF"
)

response = model.generate_content([
    "Analise este documento e resuma os pontos principais",
    pdf_file
])
```

**Uso Avançado:**[82][80]
```python
# Loop em múltiplas imagens no Google Drive
from google.colab import drive
import os

drive.mount('/content/drive')

image_folder = '/content/drive/MyDrive/images/'

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_folder, filename)
        img = PIL.Image.open(img_path)
        
        response = model.generate_content([
            "Gere um texto ALT para esta imagem",
            img
        ])
        
        print(f"{filename}: {response.text}\n")
```

**Integração com Vertex AI:**[78]
- Projetos empresariais
- Billing configurável
- Quotas e limites gerenciáveis
- Integração com outros serviços GCP

**8.4 Comparação: GPT-4V vs. Gemini**[69][70][71]

**GPT-4V:**[71][69]
- Respostas precisas e sucintas
- Excelente em non-visual questions
- Feature extraction desafiadora
- Forte em interpretação contextual

**Gemini:**[69][71]
- Respostas detalhadas e expansivas
- Incluí imagens e links relevantes
- Melhor em scene understanding
- Excelente para análise de documentos longos

**Performance:**[72][70]
- Ambos competitivos em tarefas standard
- Trade-offs entre precisão e detalhe
- Escolha depende da aplicação
- Possibilidade de combinação dos modelos[71]

**8.5 Aplicações Práticas com Foundation Models**[72][73]

**Tarefas Standard de CV:**[72]
- Semantic segmentation
- Object detection
- Image classification
- Depth estimation
- Surface normal prediction

**Técnicas:**[72]
- Prompt chaining para tarefas complexas
- Adaptação via prompts (sem fine-tuning)
- Generalização zero-shot
- Few-shot learning com exemplos

**Limitações Atuais:**[72]
- Performance inferior a modelos especializados
- Tarefas semânticas > tarefas geométricas
- Mas são "generalistas respeitáveis"
- Não treinados explicitamente para essas tarefas

**Demonstração Prática (40 min):**

**Exercício 1: GPT-4V**
- Análise de imagem de produto
- Extração de informações de invoice
- Descrição automática para e-commerce

**Exercício 2: Gemini**
- Processamento de PDF multipage
- Análise de vídeo curto
- Geração de alt-text para conjunto de imagens

**Exercício 3: Comparação**
- Mesma tarefa em ambas APIs
- Análise de diferenças nas respostas
- Discussão sobre casos de uso ideais

***

#### **Intervalo (20 min)**
*17:50 - 18:10*

***

#### **Módulo 9: Atividade Final Prática (1h50)**
*18:10 - 20:00*

**Objetivo:**
Aplicar os conhecimentos adquiridos em um projeto prático integrando múltiplas técnicas de visão computacional e foundation models.

**Atividade Proposta:**

**Projeto: Sistema de Análise Inteligente de Documentos**

**Descrição:**
Desenvolver um pipeline que processa imagens/PDFs de documentos (faturas, recibos, relatórios), realiza análise visual e extrai informações estruturadas usando foundation models.

**Requisitos:**

**Parte 1 - Preprocessing (20 min)**
- Aplicar técnicas de processamento de imagem
- Correção de orientação e qualidade
- Preparação para análise

**Parte 2 - Análise com Foundation Model (30 min)**
- Escolher entre GPT-4V ou Gemini
- Criar prompts efetivos para extração
- Processar conjunto de documentos
- Gerar outputs estruturados (JSON)

**Parte 3 - Comparação e Otimização (30 min)**
- Testar com ambos os modelos
- Comparar resultados
- Otimizar prompts
- Análise de custo-benefício

**Parte 4 - Apresentação (30 min)**
- Grupos apresentam soluções
- Discussão de desafios encontrados
- Compartilhamento de melhores práticas
- Feedback coletivo

**Recursos Fornecidos:**
- Dataset de exemplo com documentos variados
- Código base em Python/Colab
- Guias de API para OpenAI e Gemini
- Métricas de avaliação

**Entregáveis:**
- Notebook Jupyter funcional
- Documentação do approach
- Análise comparativa dos modelos
- Insights e aprendizados

**Critérios de Avaliação:**
- Qualidade do preprocessing (20%)
- Efetividade dos prompts (30%)
- Precisão das extrações (30%)
- Análise crítica e insights (20%)

***

#### **Encerramento e Discussão Final (10 min)**
*20:00 - 20:10*

**Conteúdo:**
- Revisão dos principais conceitos abordados
- Discussão sobre próximos passos
- Recursos para aprendizado contínuo
- Tendências futuras em visão computacional
- Perguntas finais
- Feedback sobre o curso

***

### **Recursos Complementares**

**Bibliografia Fundamental:**

1. **Gonzalez & Woods**[12][11][13][14]
   - "Digital Image Processing" (4th Edition)
   - Referência completa em processamento de imagem
   - Teoria matemática sólida

2. **Stanford CS231n**[10][1][3][9][2]
   - Slides e vídeos disponíveis online
   - Assignments práticos
   - Comunidade ativa

3. **Papers Fundamentais:**
   - AlexNet (Krizhevsky et al., 2012)
   - VGGNet (Simonyan & Zisserman, 2014)
   - ResNet (He et al., 2015)
   - U-Net (Ronneberger et al., 2015)
   - Vision Transformer (Dosovitskiy et al., 2020)
   - "Attention Is All You Need" (Vaswani et al., 2017)

**Ferramentas e Frameworks:**
- PyTorch / TensorFlow
- OpenCV
- Hugging Face Transformers
- Ultralytics YOLO
- OpenAI API
- Google Gemini API

**Datasets:**
- ImageNet
- COCO (Common Objects in Context)
- Pascal VOC
- CityScapes (segmentação)
- Open Images Dataset

**Plataformas de Aprendizado:**
- Google Colab (GPUs gratuitas)
- Kaggle (datasets e competitions)
- Papers with Code
- ArXiv (papers recentes)

**Comunidades:**
- r/computervision
- r/MachineLearning
- Computer Vision Foundation
- OpenCV Community

***

### **Métricas de Avaliação de Modelos**[50][51][49]

**Classificação:**[49]
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

**Detecção:**[50][49]
- IoU (Intersection over Union)
- mAP (mean Average Precision)
- Precision-Recall curves
- FPS para tempo real

**Segmentação:**[51]
- Pixel Accuracy
- Mean IoU (mIoU)
- Dice Coefficient
- Pixel-wise Cross Entropy

---

### **Data Augmentation**[83][84][85][86][87][88]

**Técnicas Geométricas:**[86][83]
- Rotation
- Flipping (horizontal/vertical)
- Cropping
- Scaling
- Translation

**Técnicas de Cor:**[84][83]
- Brightness adjustment
- Contrast modification
- Saturation changes
- Color jittering

**Técnicas Avançadas:**[87][84]
- Adding noise (Gaussian, salt-and-pepper)
- Blur
- Neural style transfer
- Adversarial training
- Mixup / CutMix

**Benefícios:**[85][88]
- Previne overfitting
- Aumenta diversidade do dataset
- Melhora generalização
- Pode aumentar dataset 10x-100x ou infinitamente

***

### **Tendências e Futuro**[89][30][33]

**Mercado:**[30][89]
- Mercado global: $17.84B (2024) → $58.33B (2032)
- CAGR: 16-19%
- Crescimento acelerado em múltiplas indústrias

**Tecnologias Emergentes:**
- Foundation models cada vez maiores
- Modelos multimodais universais
- Edge AI e modelos eficientes
- Self-supervised learning
- Neural architecture search

**Aplicações Futuras:**[33][30][31]
- Cidades inteligentes
- Saúde preditiva
- Realidade aumentada onipresente
- Robótica doméstica
- Indústria 5.0

Este plano de aula fornece uma jornada completa desde fundamentos clássicos até as tecnologias mais recentes em visão computacional, equilibrando teoria acadêmica robusta com aplicações práticas relevantes para o mercado, ideal para o perfil diversificado da turma de mestrado.

[1](https://en.wikipedia.org/wiki/Fei-Fei_Li)
[2](https://www.youtube.com/watch?v=2fq9wYslV0A)
[3](https://www.youtube.com/watch?v=vT1JzLTH4G4)
[4](https://www.historyofdatascience.com/imagenet-a-pioneering-vision-for-computers/)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10049745/)
[6](https://docs.ultralytics.com/datasets/classify/imagenet/)
[7](https://en.wikipedia.org/wiki/ImageNet)
[8](https://www.pinecone.io/learn/series/image-search/imagenet/)
[9](https://cs231n.stanford.edu/slides/2024/lecture_1_part_2.pdf)
[10](https://csdiy.wiki/en/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/CS231/)
[11](https://studylib.net/doc/25705174/digital-image-processing-4th-ed.---r.-gonzalez--r.-woods-...)
[12](https://api.pageplace.de/preview/DT0400.9781292223070_A37747583/preview-9781292223070_A37747583.pdf)
[13](https://www.cl72.org/090imagePLib/books/Gonzales,Woods-Digital.Image.Processing.4th.Edition.pdf)
[14](http://www.r-5.org/files/books/computers/algo-list/image-processing/flat/Rafael_Gonzalez_Richard_Woods-Digital_Image_Processing-EN.pdf)
[15](https://books.google.com/books/about/Digital_Image_Processing.html?hl=pt-BR&id=738oAQAAMAAJ)
[16](https://en.wikipedia.org/wiki/Convolutional_neural_network)
[17](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/)
[18](https://www.upgrad.com/blog/basic-cnn-architecture/)
[19](https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-machine-learning/)
[20](https://www.v7labs.com/blog/convolutional-neural-networks-guide)
[21](https://www.v7labs.com/blog/yolo-object-detection)
[22](https://pdfs.semanticscholar.org/9d76/4a97861dfa241416313aaa1c956ecfeb7d87.pdf)
[23](https://www.digitalocean.com/community/tutorials/popular-deep-learning-architectures-resnet-inceptionv3-squeezenet)
[24](https://www.reddit.com/r/MachineLearning/comments/6e6mlf/d_is_vgg_common_in_newer_research_or_is_resnet/)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC8024101/)
[26](https://www.geeksforgeeks.org/computer-vision/transfer-learning-for-computer-vision/)
[27](https://ijsred.com/volume6/issue4/IJSRED-V6I4P20.pdf)
[28](https://machinelearningmastery.com/leveraging-transfer-learning-in-computer-vision-for-quick-wins/)
[29](https://blog.roboflow.com/what-is-transfer-learning/)
[30](https://www.grandviewresearch.com/industry-analysis/computer-vision-market)
[31](https://dac.digital/what-are-the-practical-applications-of-modern-computer-vision-technology/)
[32](https://pragmile.com/top-12-practical-applications-of-computer-vision-how-ai-is-revolutionizing-industries/)
[33](https://intellias.com/top-computer-vision-applications-for-industries/)
[34](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025)
[35](https://www.picsellia.com/post/segmentation-vs-detection-vs-classification-in-computer-vision-a-comparative-analysis)
[36](https://viso.ai/deep-learning/imagenet/)
[37](https://eines.com/deep-learning-in-computer-vision-segmentation/)
[38](https://flypix.ai/blog/deep-learning-segmentation/)
[39](https://encord.com/blog/yolo-object-detection-guide/)
[40](https://www.geeksforgeeks.org/computer-vision/how-does-yolo-work-for-object-detection/)
[41](https://arxiv.org/html/2410.17725v1)
[42](https://addaxis.ai/unveiling-the-power-of-yolo-transforming-object-detection-with-the-yolo-series-models/)
[43](https://encord.com/blog/image-segmentation-for-computer-vision-best-practice-guide/)
[44](https://developers.arcgis.com/python/latest/guide/how-unet-works/)
[45](https://viso.ai/deep-learning/u-net-a-comprehensive-guide-to-its-architecture-and-applications/)
[46](https://glassboxmedicine.com/2020/01/21/segmentation-u-net-mask-r-cnn-and-medical-applications/)
[47](https://en.wikipedia.org/wiki/U-Net)
[48](https://www.reddit.com/r/learnmachinelearning/comments/voce98/the_unet_neural_network_architecture_for_semantic/)
[49](https://viso.ai/computer-vision/model-performance/)
[50](https://www.geeksforgeeks.org/computer-vision/evaluation-of-computer-vision-model/)
[51](https://encord.com/blog/measure-model-performance-computer-vision/)
[52](https://packagex.io/blog/ocr-machine-learning)
[53](https://repositorio.unesp.br/items/fd14f909-e0a4-4cc6-8adf-5b63c3838eb7)
[54](https://www.affinda.com/blog/machine-learning-ocr)
[55](https://labelyourdata.com/articles/ocr-with-deep-learning)
[56](https://neptune.ai/blog/building-deep-learning-based-ocr-model)
[57](https://github.com/das-projects/deepOCR)
[58](https://www.v7labs.com/blog/vision-transformer-guide)
[59](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))
[60](https://blog.roboflow.com/vision-transformers/)
[61](https://www.geeksforgeeks.org/deep-learning/vision-transformer-vit-architecture/)
[62](https://viso.ai/deep-learning/vision-transformer-vit/)
[63](https://en.wikipedia.org/wiki/Vision_transformer)
[64](https://www.ultralytics.com/glossary/vision-transformer-vit)
[65](https://pmc.ncbi.nlm.nih.gov/articles/PMC10333157/)
[66](https://arxiv.org/abs/2303.13731)
[67](https://developer.nvidia.com/blog/emulating-the-attention-mechanism-in-transformer-models-with-a-fully-convolutional-network/)
[68](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1178450/full)
[69](https://www.emergentmind.com/papers/2312.15011)
[70](https://arxiv.org/html/2312.10637v1)
[71](https://arxiv.org/abs/2312.15011)
[72](https://openreview.net/forum?id=h3unlS2VWz)
[73](https://blog.roboflow.com/foundation-model/)
[74](https://microsoft.github.io/promptflow/reference/tools-reference/openai-gpt-4v-tool.html)
[75](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/tools-reference/openai-gpt-4v-tool?view=azureml-api-2)
[76](https://www.datacamp.com/tutorial/gpt-4-vision-comprehensive-guide)
[77](https://www.youtube.com/watch?v=ZNgQCQO0tH8)
[78](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart)
[79](https://ai.google.dev/gemini-api/docs/api-key)
[80](https://www.youtube.com/watch?v=YI_Cku18ssM)
[81](https://docs.cloud.google.com/vision/docs)
[82](https://www.videosdk.live/developer-hub/ai/gemini-vision-api)
[83](https://research.aimultiple.com/data-augmentation-techniques/)
[84](https://aws.amazon.com/what-is/data-augmentation/)
[85](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025)
[86](https://blog.roboflow.com/data-augmentation/)
[87](https://viso.ai/computer-vision/image-data-augmentation-for-computer-vision/)
[88](https://encord.com/blog/data-augmentation-guide/)
[89](https://www.fortunebusinessinsights.com/computer-vision-market-108827)
[90](https://stackoverflow.com/questions/1718903/what-do-square-brackets-mean-in-function-class-documentation)
[91](https://en.wikipedia.org/wiki/Quotation_mark)
[92](https://en.wikipedia.org/wiki/F-test)
[93](https://en.wikipedia.org/wiki/E_(mathematical_constant))
[94](https://en.wikipedia.org/wiki/I)
[95](https://pt.wikipedia.org/wiki/L_(Death_Note))
[96](https://pt.wikipedia.org/wiki/C_(linguagem_de_programa%C3%A7%C3%A3o))
[97](https://pt.wikipedia.org/wiki/O)
[98](https://discuss.python.org/t/meaning-of-square-brackets/20294)
[99](https://learn.microsoft.com/en-us/style-guide/punctuation/quotation-marks)
[100](https://www.britannica.com/dictionary/f)
[101](https://pt.wikipedia.org/wiki/E)
[102](https://www.merriam-webster.com/dictionary/i)
[103](https://pt.wikipedia.org/wiki/L)
[104](https://www.merriam-webster.com/dictionary/c)
[105](https://en.wikipedia.org/wiki/O)
[106](https://en.wikipedia.org/wiki/Bracket)
[107](https://www.reddit.com/r/writing/comments/15xfgvj/can_someone_explain_quotation_marks_and_why/)
[108](https://www.britannica.com/topic/F-letter)
[109](https://pt.wikipedia.org/wiki/%C3%89)
[110](https://www.scribd.com/document/363154742/Gonzalez-Rc-Woods-Re-Digital-Image-Processing)
[111](https://meta-quantum.today/?p=7985)
[112](https://www.scribd.com/document/740518312/lecture-1-2-ruohan)
[113](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)
[114](https://arxiv.org/html/2312.10637v2)
[115](https://www.youtube.com/watch?v=tgipd6F6KZc)
[116](https://www.youtube.com/watch?v=taC5pMCm70U)
[117](https://opencv.org/blog/deep-learning-with-computer-vision/)
[118](https://cloud.google.com/use-cases/ocr)
[119](https://toloka.ai/blog/transfer-learning/)
[120](https://learn.microsoft.com/pt-br/azure/machine-learning/prompt-flow/tools-reference/openai-gpt-4v-tool?view=azureml-api-2)
[121](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns)
[122](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
[123](https://www.reddit.com/r/OpenAIDev/comments/1hjmj39/gpt4_vision_capabilities/)
[124](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/)
[125](https://arxiv.org/abs/2409.07736)
[126](https://www.sciencedirect.com/science/article/pii/S2666285X21000558)
[127](https://cloud.google.com/vision)
[128](https://www.datacamp.com/blog/yolo-object-detection-explained)
[129](https://dev.to/saaransh_gupta_1903/resnet-vs-efficientnet-vs-vgg-vs-nn-2hf5)
[130](https://cloud.google.com/vision/docs/tutorials)
[131](https://en.wikipedia.org/wiki/You_Only_Look_Once)
[132](https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/)
[133](https://www.reddit.com/r/MachineLearning/comments/qidpqx/d_how_to_truly_understand_attention_mechanism_in/)
[134](https://viso.ai/applications/computer-vision-applications/)
[135](https://www.digitalocean.com/community/tutorials/unet-architecture-image-segmentation)
[136](https://labelyourdata.com/articles/object-detection-metrics)
[137](https://www.nature.com/articles/s41598-024-56706-x)
[138](https://arxiv.org/abs/2301.02830)
[139](https://www.youtube.com/watch?v=nbCY93BMw0U)