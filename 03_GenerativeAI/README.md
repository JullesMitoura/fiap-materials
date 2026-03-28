<div align="center">

<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>


<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg"
     alt="Python"
     width="48"
     height="48"/>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg"
     alt="Jupyter"
     width="48"
     height="48"/>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg"
     alt="PyTorch"
     width="48"
     height="48"/>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg"
     alt="TensorFlow"
     width="48"
     height="48"/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)


<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# IA Generativa e Redes Avançadas
> PhD. Julles Mitoura

Módulo prático de Inteligência Artificial Generativa. Os notebooks percorrem uma trilha progressiva: da implementação manual do mecanismo de atenção até pipelines RAG com bancos de dados vetoriais — passando por fine-tuning eficiente com LoRA, classificação de texto, embeddings semânticos e recuperação vetorial com FAISS.

---

## Estrutura do Módulo

```
03_GenerativeAI/
├── genai_aula1_attention_torch.ipynb               # Aula 1a — Transformer com PyTorch
├── genai_aula1_model_test.ipynb                    # Aula 1b — Inferência e teste do modelo
├── genai_aula2_attention_scratch.ipynb             # Aula 2  — Atenção do zero (só NumPy)
├── genai_aula3_1_models_lstm.ipynb                 # Aula 3a — Classificação de sentimentos: BiLSTM
├── genai_aula3_2_models_fcnn.ipynb                 # Aula 3b — Classificação de sentimentos: FCNN + TF-IDF
├── genai_aula3_3_models_transformer.ipynb          # Aula 3c — Fine-tuning de DistilBERT
├── genai_aula3_4_models_evaluate.ipynb             # Aula 3d — Avaliação comparativa dos três modelos
├── genai_aula4_lora_vision.ipynb                   # Aula 4  — LoRA para Visão (ViT + Beans dataset)
├── genai_aula5_qlora_gpt2.ipynb                    # Aula 5  — QLoRA com GPT-2 (quantização NF4 + LoRA)
├── genai_aula6_llm_api_call.ipynb                  # Aula 6  — Consumo de LLMs via API
├── genai_aula7_concepts_rag.ipynb                  # Aula 7  — Conceitos de RAG: embeddings, similaridade e pipeline do zero
├── genai_aula8_vdbs.ipynb                          # Aula 8  — Bancos de dados vetoriais com FAISS
├── docs/
│   └── article.pdf                                 # Corpus RAG: artigo científico (21 p.) — Aulas 7 e 8
├── requirements.txt
├── .env.example                                    # OPENAI_API_KEY (não versionar)
└── suporte/                                        # Documentação de bibliotecas por aula
```

---

## Aula 1 — O Mecanismo de Atenção com PyTorch

### `genai_aula1_attention_torch.ipynb` — Construção e treino

Apresenta a arquitetura Transformer proposta no paper *"Attention is All You Need"* (2017) e implementa um modelo completo com PyTorch, do zero.

**O que é abordado:**
- Visão geral da arquitetura Transformer: embeddings, single-head e multi-head attention, feed-forward, conexões residuais e projeção sobre o vocabulário
- Implementação da classe `Transformer` com `torch.nn`: `Embedding`, `MultiheadAttention`, `LayerNorm`, `Linear`, `GELU`, `Sequential`
- Dataset sintético com `torch.utils.data.Dataset` e `DataLoader` para treino de predição do próximo token (vocab_size=1000, seq_len=16)
- Loop de treino com `CrossEntropyLoss` e otimizador `AdamW` ao longo de 1.000 épocas
- Avaliação: loss média, perplexidade, acurácia top-1 (99,31%) e top-5 (100%)
- Salvamento do checkpoint em `models/transformer_attention.pt`

**Bibliotecas:** `torch`, `matplotlib`

---

### `genai_aula1_model_test.ipynb` — Inferência e teste

Consome o checkpoint gerado no notebook anterior e demonstra o fluxo completo de inferência.

**O que é abordado:**
- Reconstrução da arquitetura e restauração dos pesos com `torch.load` e `load_state_dict`
- Inferência sem gradientes com `torch.no_grad`
- Ranking top-k: `softmax` sobre os logits da última posição e `topk` para os tokens mais prováveis
- Interpretação dos resultados: acerto top-1 e presença do token esperado no top-5

**Bibliotecas:** `torch`

---

## Aula 2 — O Mecanismo de Atenção do Zero

### `genai_aula2_attention_scratch.ipynb`

Reimplementa os componentes internos do Transformer manualmente — apenas NumPy, sem nenhum framework de deep learning.

**O que é abordado:**
- Camada de embedding como indexação em uma matriz aleatória
- `softmax` com estabilidade numérica (subtração do máximo)
- `scaled_dot_product_attention`: produto escalar QKᵀ, escala por √dₖ, softmax e ponderação de V
- Camada `linear_softmax`: projeção para o espaço do vocabulário
- Encoder e Decoder do zero; predição sobre sequências sintéticas (1–10 tokens)
- Modelo completo em 5 etapas: embedding → self-attention → projeção linear → argmax → token predito

**Bibliotecas:** `numpy`

> Material de suporte: `suporte/genai_aula2_mecanismo_atencao_qkv.md` — detalhamento matemático de Q, K e V

---

## Aula 3 — Modelos de NLP para Classificação de Texto

Quatro notebooks que treinam e comparam três arquiteturas para a mesma tarefa: classificação de sentimentos em 6 classes (anger, fear, joy, love, sadness, surprise) sobre 18.000 amostras.

---

### `genai_aula3_1_models_lstm.ipynb` — BiLSTM

Rede neural recorrente bidirecional para classificação de texto.

**O que é abordado:**
- Pré-processamento com spaCy: lematização, remoção de stopwords e pontuação
- Tokenização e padding de sequências com `Tokenizer` e `pad_sequences` do Keras (MAX_LEN=100)
- `LabelEncoder` e balanceamento com `compute_class_weight`
- Arquitetura: `Embedding(128)` → `Bidirectional(LSTM(64))` → `Dropout` → `Dense(64, ReLU)` → `Dense(6, Softmax)`
- Treino com `EarlyStopping` e cross-entropy ponderada
- Avaliação: `classification_report`, `confusion_matrix`, curvas de treino — F1-Macro ~86%

**Bibliotecas:** `tensorflow`, `scikit-learn`, `spacy`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_2_models_fcnn.ipynb` — FCNN + TF-IDF

Rede feed-forward sobre vetores TF-IDF — abordagem mais simples, sem recorrência.

**O que é abordado:**
- Mesmo pipeline de pré-processamento com spaCy
- `TfidfVectorizer` (unigramas + bigramas): 11.504 features esparsas
- Arquitetura: `Dense(256, SELU)` → `Dropout` → `Dense(128, SELU)` → `Dense(64, SELU)` → `Dense(6, Softmax)`
- Comparação implícita com BiLSTM: entrada estruturada, sem embedding
- F1-Macro ~86%, salvamento do vetorizador TF-IDF e do modelo

**Bibliotecas:** `tensorflow`, `scikit-learn`, `spacy`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_3_models_transformer.ipynb` — Fine-tuning de DistilBERT

Fine-tuning de DistilBERT pré-treinado com a biblioteca HuggingFace Transformers.

**O que é abordado:**
- DistilBERT: 40% menos parâmetros que BERT-base, 60% mais rápido, 97% da capacidade — via knowledge distillation
- Tokenização WordPiece com `AutoTokenizer`; sem pré-processamento manual de texto
- Conversão do DataFrame em `Dataset` HuggingFace e `DataCollatorWithPadding` para padding dinâmico
- Fine-tuning com `TrainingArguments` (lr=2e-5, weight_decay=0.01, 4 épocas) e `Trainer`
- F1-Macro ~87%, salvamento em `models/distilbert_sentiment/`

**Bibliotecas:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_4_models_evaluate.ipynb` — Avaliação Comparativa

Carrega os três modelos treinados (BiLSTM, FCNN e DistilBERT) e avalia sobre o mesmo conjunto de teste.

**O que é abordado:**
- Reaplicação do pipeline spaCy para BiLSTM e FCNN; inferência direta com `pipeline` do HuggingFace para DistilBERT
- Métricas unificadas: accuracy, F1 macro, `classification_report` e `confusion_matrix` por modelo
- Análise de confiança por classe e comparação de padrões de erro
- Gráfico comparativo: FCNN 88,9%, BiLSTM 94,4%, DistilBERT 94,4%

**Bibliotecas:** `tensorflow`, `transformers`, `torch`, `spacy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## Aula 4 — LoRA para Visão

### `genai_aula4_lora_vision.ipynb`

Demonstra fine-tuning eficiente de um Vision Transformer (ViT) usando LoRA — atualizando apenas ~0,2% dos parâmetros do modelo base.

**O que é abordado:**
- **Vision Transformer (ViT):** divisão de imagens em patches e processamento como sequência via atenção
- **LoRA (Low-Rank Adaptation):** injeção de matrizes de baixo rank nas projeções de atenção (query, value); congelamento do restante
- Configuração com `LoraConfig` da biblioteca `peft`: rank=8, alpha=16, target modules = query e value
- Dataset **Beans** (HuggingFace): 3 classes de doenças em plantas (angular_leaf_spot, bean_rust, healthy), ~1.000 imagens de treino
- Pré-processamento com `AutoImageProcessor` (redimensionamento para 224×224)
- Fine-tuning com `Trainer` (lr=2e-4, batch=16, 5 épocas); pode rodar em CPU (~15 min)
- Avaliação: `classification_report` e `confusion_matrix`

**Bibliotecas:** `transformers`, `peft`, `datasets`, `torch`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`

---

## Aula 5 — QLoRA com GPT-2

### `genai_aula5_qlora_gpt2.ipynb`

Apresenta o **QLoRA** (*Quantized LoRA*) — a técnica que viabiliza o fine-tuning de LLMs em hardware limitado ao combinar quantização NF4 4-bit com adaptadores LoRA de rank baixo. Fine-tuning do GPT-2 para análise de sentimentos em avaliações de filmes.

**O que é abordado:**
- **Por que PEFT:** custo de memória do full fine-tuning (pesos + gradientes + Adam states) — de GPT-2 a Llama-3-70B
- **LoRA:** fatorização de baixo rank $\Delta W = BA$; por que apenas ~0,x% dos parâmetros precisam ser treinados
- **Quantização:** escada FP32 → FP16 → INT8 → NF4; NF4 vs INT4 (bins não-uniformes calibrados para distribuição normal); double quantization
- **QLoRA:** modelo base congelado em NF4 4-bit + adaptadores LoRA em FP16 treináveis
- **GPT-2:** arquitetura decoder-only; módulos `c_attn` (QKV fusionados) e `c_proj` como targets LoRA
- Dataset de avaliações de filmes (`dataset.csv`) formatado como causal LM: `Review: [...]\nSentiment: [label]`
- `BitsAndBytesConfig` com NF4 (CUDA) e fallback FP32 automático para CPU/MPS
- `LoraConfig` + `get_peft_model` + `DataCollatorForLanguageModeling(mlm=False)`
- Comparação base GPT-2 vs fine-tuned; salvamento do adaptador (~1–5 MB vs ~500 MB do modelo completo)

**Bibliotecas:** `transformers`, `peft`, `datasets`, `torch`, `pandas`, `matplotlib`, `seaborn`

> Requer `bitsandbytes` para quantização NF4 (CUDA). Em CPU/MPS o notebook executa com LoRA em FP32.

---

## Aula 6 — Consumo de LLMs via API

### `genai_aula6_llm_api_call.ipynb`

Muda o foco de construção/treino de modelos para **consumo de LLMs em produção** via OpenAI API. Cada parâmetro de geração é explicado teoricamente e demonstrado com experimentos comparativos.

**O que é abordado:**

**Fundamentos:**
- Estrutura das mensagens: roles `system`, `user` e `assistant`
- Anatomia da resposta: `choices`, `finish_reason`, `usage` (tokens de entrada, saída e total)

**Parâmetros de geração:**

| Parâmetro | O que controla |
|---|---|
| `model` | Escolha do modelo: trade-off entre capacidade e custo |
| `temperature` | Aleatoriedade na seleção de tokens (0 = determinístico, 2 = máxima criatividade) |
| `max_tokens` | Limite de tokens na saída; impacto no `finish_reason` |
| `top_p` | Nucleus sampling: filtra os tokens de menor probabilidade após o softmax |
| `frequency_penalty` | Penaliza tokens proporcionalmente à frequência de aparição |
| `presence_penalty` | Penaliza tokens pela simples presença (encoraja novos temas) |
| `stop` | Strings de parada — interrompe a geração ao encontrá-las |
| `n` | Número de respostas independentes geradas por chamada |

**Boas práticas:**
- System prompt fraco vs. forte: comparação de impacto na qualidade
- Estimativa de custo por chamada a partir de `response.usage`
- Tratamento de erros: `RateLimitError`, `AuthenticationError`, `BadRequestError`

**Bibliotecas:** `openai`, `python-dotenv`

---

## Aula 7 — Conceitos de RAG

### `genai_aula7_concepts_rag.ipynb`

Introduz **Retrieval-Augmented Generation (RAG)** construindo cada componente do zero — sem frameworks externos. O pipeline completo é implementado com Python puro e NumPy, usando um artigo científico real como corpus.

**O que é abordado:**

**Embeddings:**
- Conceito: $\vec{v} = \text{Embed}(\text{texto}) \in \mathbb{R}^{1536}$ — texto mapeado para vetor de alta dimensão
- Propriedade semântica: textos similares → vetores próximos
- Tabela de modelos OpenAI: `text-embedding-3-small`, `text-embedding-3-large`, `ada-002`
- Função `embed(text)` via `client.embeddings.create(model="text-embedding-3-small")`

**Similaridade de Vetores:**
- Similaridade de cosseno: $\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|}$
- Implementação NumPy pura; comparação de frases contra query com ranking

**Banco de Dados Vetorial Local:**
- Classe `VectorDatabase` em Python puro (listas + NumPy), busca linear O(n)
- Métodos `add(id, text, embedding)` e `search(query_embedding, top_k)`

**Extração de PDF e Chunking:**
- `pypdf.PdfReader` para extração de texto; `clean_text()` com regex para artefatos
- `chunk_text(text, chunk_size=1000, overlap=200)` — janela deslizante com sobreposição

**Pipeline RAG Completo:**
- Fase 1 — Indexação: PDF → limpeza → chunking → embed → VectorDatabase
- Fase 2 — Consulta: embed(pergunta) → search → top-k chunks → prompt aumentado → `gpt-4o-mini`
- Demo com 3 perguntas sobre o artigo de termodinâmica computacional

**Bibliotecas:** `openai`, `python-dotenv`, `numpy`, `pypdf`

---

## Aula 8 — Bancos de Dados Vetoriais com FAISS

### `genai_aula8_vdbs.ipynb`

Substitui o `VectorDatabase` artesanal da Aula 7 pelo **FAISS** (Facebook AI Similarity Search), a biblioteca de referência para busca vetorial em escala. Cobre três tipos de índice, benchmark comparativo, persistência em disco e RAG com FAISS.

**O que é abordado:**

**Motivação para VDBs reais:**
- Problema de escala: busca linear O(n·d) inviável para corpora de milhões de documentos
- Approximate Nearest Neighbor (ANN): trade-off precisão ↔ velocidade

**IndexFlatL2 — Busca Exata:**
- Distância Euclidiana: $\|\vec{u} - \vec{v}\|^2 = \sum_i (u_i - v_i)^2$
- Sem treinamento; retorna sempre os k vizinhos corretos
- Mesma acurácia que VectorDatabase da Aula 7, implementação C++ muito mais rápida

**Distâncias e Normalização:**
- Equivalência L2 ↔ cosseno para vetores unitários: $\|\vec{u} - \vec{v}\|^2 = 2(1 - \cos(\vec{u}, \vec{v}))$
- `faiss.normalize_L2(array)` — normalização in-place; `IndexFlatIP` com vetores normalizados = busca por cosseno

**IndexIVFFlat — Busca Aproximada:**
- Particionamento em células de Voronoi via k-means; parâmetros `nlist` (clusters) e `nprobe` (clusters visitados)
- Etapa de `train()` obrigatória; `nprobe=nlist` = resultado exato

**Benchmark:**
- 50.000 vetores sintéticos d=128; 100 queries cronometradas com `time.perf_counter()`
- Speedup IVFFlat vs FlatL2 + recall (% de resultados corretos)

**Persistência:**
- `faiss.write_index(index, path)` / `faiss.read_index(path)` — salvar e recarregar sem re-embedar

**RAG com FAISS:**
- `IndexFlatIP` com vetores normalizados como backend de busca
- `rag_query_faiss(question, index, texts, top_k, verbose)` — mesma interface da Aula 7
- Índice salvo em `dataset/rag_faiss.index`; demo com 3 perguntas sobre o artigo

**Bibliotecas:** `faiss-cpu`, `openai`, `python-dotenv`, `numpy`, `pypdf`

---

## Pré-requisitos

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
python -m spacy download pt_core_news_sm   # modelo spaCy em português
```

Crie um arquivo `.env` na raiz com sua chave de API (necessário para as Aulas 2, 6, 7 e 8):

```
OPENAI_API_KEY=sk-...
```

---

## Bibliotecas por Aula

### `torch` — PyTorch

| Aulas | Papel |
|---|---|
| 1a, 1b | Arquitetura completa: `nn.Module`, `nn.Embedding`, `nn.MultiheadAttention`, `nn.Linear`, `AdamW`, `DataLoader`. Avaliação com perplexidade, acurácia top-1/top-5 e salvamento de checkpoint |
| 3c, 3d | Base de inferência para DistilBERT via HuggingFace |
| 4 | Backend do ViT; detecção de device (CUDA/MPS/CPU) |
| 5 | Backend do GPT-2; `BitsAndBytesConfig`; geração de texto com `.generate()` |

### `tensorflow` / `tf_keras`

Usado nas Aulas 3a, 3b e 3d. Constrói e treina os modelos BiLSTM e FCNN via `tensorflow.keras`: camadas `Embedding`, `Bidirectional(LSTM)`, `Dense`, `Dropout`, `EarlyStopping`, `Tokenizer`, `pad_sequences`. `tf_keras` é um pacote standalone de compatibilidade exigido pelo TF em alguns ambientes.

### `transformers` — HuggingFace

| Aula | Componentes |
|---|---|
| 3c | `AutoTokenizer`, `AutoModelForSequenceClassification`, `TrainingArguments`, `Trainer`, `DataCollatorWithPadding` |
| 4 | `AutoImageProcessor`, `ViTForImageClassification` |
| 5 | `AutoTokenizer`, `AutoModelForCausalLM`, `BitsAndBytesConfig`, `DataCollatorForLanguageModeling` |

### `peft` — Parameter-Efficient Fine-Tuning

Usado nas Aulas 4 e 5. `LoraConfig` define rank `r`, `lora_alpha` (escala = α/r), `target_modules` e `lora_dropout`. `get_peft_model` congela os pesos base e injeta as matrizes A e B treináveis. Em vez de atualizar $W$, aprende $\Delta W = B \cdot A$ com $r \ll \min(d,k)$ — redução de dezenas de vezes no número de parâmetros treináveis. `prepare_model_for_kbit_training` ativa gradient checkpointing quando combinado com bitsandbytes (Aula 5).

### `bitsandbytes`

Usado na Aula 5 (**requer CUDA**). Viabiliza o QLoRA carregando os pesos do modelo em NF4 4-bit: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_use_double_quant=True`. Os pesos são dequantizados para FP16 no forward pass. Em CPU/MPS o notebook cai automaticamente para FP32.

### `datasets` — HuggingFace

Usado nas Aulas 3c, 4 e 5. Converte DataFrames pandas em objetos `Dataset` para integração com `Trainer`/`SFTTrainer`. Suporta `map()` em batch, caching e `train_test_split`. Na Aula 4 carrega o dataset Beans diretamente do HuggingFace Hub (`load_dataset("beans")`).

### `numpy`

Backbone da Aula 2: implementa toda a arquitetura Transformer manualmente — embedding como indexação de matriz, `softmax` com estabilidade numérica, `scaled_dot_product_attention` (`np.dot(Q, K.T) / sqrt(dk)`), projeção linear e `np.argmax`. Na Aula 7 implementa `cosine_similarity` e é a estrutura interna do `VectorDatabase`. Na Aula 8 é a interface entre os embeddings OpenAI e o FAISS (`np.array(..., dtype=np.float32)`). Presente como utilitário numérico em todas as aulas.

### `scikit-learn`

Usado nas Aulas 3 e 4. Fornece `train_test_split`, `LabelEncoder`, `TfidfVectorizer` (11.504 features, Aula 3b), `compute_class_weight` e métricas unificadas: `classification_report`, `confusion_matrix`, `f1_score`, `accuracy_score`.

### `spacy`

Usado nas Aulas 3a, 3b e 3d. Pipeline de pré-processamento de texto: tokenização, remoção de stopwords e lematização antes de alimentar BiLSTM e FCNN. O mesmo pipeline é reaplicado na Aula 3d para garantir consistência na avaliação comparativa.

### `openai`

| Aula | Uso |
|---|---|
| 2 | `client.embeddings.create(model="text-embedding-3-small")` — substitui embeddings aleatórios por vetores semânticos reais para o classificador |
| 6 | `client.chat.completions.create` — Chat Completions com controle de `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `n`, streaming e saída estruturada JSON |
| 7 | `client.embeddings.create(model="text-embedding-3-small")` para gerar vetores de chunks e queries; `client.chat.completions.create(model="gpt-4o-mini")` para geração de resposta RAG |
| 8 | Mesmas funções da Aula 7; embeddings usados como entrada para índices FAISS |

### `python-dotenv`

Usado nas Aulas 2 e 6. `load_dotenv(override=True)` carrega `OPENAI_API_KEY` do arquivo `.env` sem expor credenciais no código.

### `pypdf`

Usado nas Aulas 7 e 8. `PdfReader` extrai texto página a página de PDFs em Python puro, sem dependências de ferramentas externas como Poppler. O texto bruto passa por `clean_text()` com regex para remover cabeçalhos de página, numeração e hifenização de linha antes do chunking.

### `faiss-cpu`

Usado na Aula 8. Biblioteca do Meta AI para busca vetorial eficiente — núcleo C++ com Python bindings. Fornece `IndexFlatL2` (busca exata por distância L2), `IndexFlatIP` (inner product — cosine com vetores normalizados via `faiss.normalize_L2()`) e `IndexIVFFlat` (busca aproximada com particionamento Voronoi, requer `train()`). Persistência via `faiss.write_index` / `faiss.read_index`. A variante `faiss-gpu` suporta CUDA; `faiss-cpu` cobre todos os ambientes.

### `matplotlib` / `seaborn` / `pandas`

Presentes em todos os notebooks. `matplotlib` plota curvas de loss, distribuições e comparações. `seaborn` gera heatmaps para matrizes de confusão. `pandas` realiza leitura de CSV, exploração e filtragem dos datasets.

---

## Documentação de bibliotecas

Cada aula possui um arquivo de suporte detalhando a função de cada biblioteca no contexto dos notebooks:

| Arquivo | Aulas |
|---|---|
| `suporte/genai_aula1_libs.md` | Aula 1a e 1b |
| `suporte/genai_aula2_libs.md` | Aula 2 |
| `suporte/genai_aula2_mecanismo_atencao_qkv.md` | Detalhamento matemático de Q, K e V |
| `suporte/genai_aula3_libs.md` | Aulas 3a, 3b, 3c e 3d |
| `suporte/genai_aula4_libs.md` | Aula 4 |
| `suporte/genai_aula5_libs.md` | Aula 5 |
| `suporte/genai_aula6_libs.md` | Aula 6 |
| `suporte/genai_aula7_libs.md` | Aula 7 |
| `suporte/genai_aula8_libs.md` | Aula 8 |

---

## Trilha de aprendizado

```
Aula 1  →  Transformer com PyTorch (implementação + treino + inferência)
   ↓
Aula 2  →  Mesma arquitetura do zero, só com NumPy (abre a "caixa preta")
   ↓
Aula 3  →  NLP aplicado: BiLSTM → FCNN+TF-IDF → DistilBERT fine-tuning → comparação
   ↓
Aula 4  →  LoRA: fine-tuning eficiente de Vision Transformer em imagens
   ↓
Aula 5  →  QLoRA: quantização NF4 + LoRA para fine-tuning de GPT-2 em texto
   ↓
Aula 6  →  Consumo de LLMs em produção: parâmetros, boas práticas, OpenAI API
   ↓
Aula 7  →  RAG do zero: embeddings, similaridade de cosseno, VectorDatabase, chunking de PDF, pipeline RAG completo
   ↓
Aula 8  →  Bancos de dados vetoriais com FAISS: IndexFlatL2, IndexFlatIP, IndexIVFFlat, benchmark, persistência e RAG com FAISS
```
