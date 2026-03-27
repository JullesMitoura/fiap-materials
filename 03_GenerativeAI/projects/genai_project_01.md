# Projeto 01 - Fine-Tuning de Modelos Pre-Treinados

**Disciplina:** Generative AI & Advanced Networks
**Instituição:** FIAP
**Professor:** PhD. Julles Mitoura

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-green?style=flat-square)

---

## Descrição Geral

Neste projeto, você irá aplicar técnicas de **fine-tuning eficiente em parâmetros (PEFT)** para adaptar um modelo pré-treinado a uma tarefa de sua escolha. O objetivo é consolidar os conceitos trabalhados nas Aulas 4 e 5 (LoRA com Vision Transformer/ViT e QLoRA com GPT-2) e demonstrar autonomia na seleção, configuração e avaliação de um pipeline de fine-tuning completo.

O projeto deve ser entregue em formato de **Jupyter Notebook** (`.ipynb`), com o código funcional, células de texto explicativas e todos os resultados reproduzíveis.

---

## Objetivos

Ao final do projeto, o aluno será capaz de:

- Selecionar e justificar um modelo pré-treinado adequado à tarefa proposta
- Preparar e analisar um dataset adequado para fine-tuning supervisionado
- Configurar e executar um pipeline de fine-tuning com LoRA ou QLoRA via biblioteca PEFT
- Avaliar o desempenho do modelo com métricas apropriadas
- Testar o modelo com dados externos ao dataset de treinamento

---

## Etapas do Projeto

### 1. Seleção do Modelo Pré-Treinado

Escolha um modelo pré-treinado disponível no [HuggingFace Hub](https://huggingface.co/models) que seja adequado à sua tarefa. O modelo **não pode ser** o mesmo utilizado nas aulas (GPT-2 para NLP ou ViT + Beans para visão).

**O notebook deve conter:**

- Nome e identificador do modelo (ex: `distilbert-base-uncased`)
- Arquitetura (BERT, GPT, ViT, T5, etc.) e número de parâmetros
- Justificativa da escolha: por que esse modelo é adequado para a tarefa?
- Domínio de aplicação: NLP, Visão, Áudio, Multimodal, etc.

**Modelos sugeridos (mas não limitados a):**

| Modelo | Domínio | Tarefa sugerida |
|--------|---------|-----------------|
| `distilbert-base-uncased` | NLP | Classificação de texto |
| `roberta-base` | NLP | Análise de sentimentos |
| `facebook/bart-base` | NLP | Sumarização / Geração |
| `google/vit-base-patch16-224` | Visão | Classificação de imagens |
| `microsoft/resnet-50` | Visão | Classificação de imagens |
| `openai/whisper-small` | Áudio | Transcrição / Classificação |

---

### 2. Seleção do Dataset

Escolha um dataset para treinar e avaliar seu modelo. O dataset **não pode ser** o Beans Dataset nem o dataset de críticas de filmes utilizados nas aulas.

**O notebook deve conter:**

- Nome e fonte do dataset (HuggingFace Hub, Kaggle, repositório próprio, etc.)
- Descrição: o que os dados representam?
- Análise exploratória:
  - Número de amostras por split (treino, validação, teste)
  - Distribuição de classes ou rótulos (gráfico ou tabela)
  - Exemplos representativos de cada classe
  - Estatísticas relevantes (comprimento dos textos, resolução das imagens, etc.)
- Justificativa: por que esse dataset é adequado para o modelo escolhido?

**Requisitos mínimos do dataset:**

- Mínimo de **2 classes** ou rótulos distintos
- Mínimo de **500 amostras** de treinamento
- Deve possuir split de teste separado (ou ser dividido manualmente)

---

### 3. Fine-Tuning com LoRA ou QLoRA

Aplique a técnica de fine-tuning eficiente utilizando a biblioteca [PEFT (HuggingFace)](https://huggingface.co/docs/peft). Utilize **LoRA** ou **QLoRA** conforme a disponibilidade de hardware e a arquitetura do modelo.

**O notebook deve conter:**

#### 3.1 Configuração do LoRA / QLoRA

Documente e justifique cada hiperparâmetro utilizado:

| Hiperparâmetro | Valor utilizado | Justificativa |
|----------------|----------------|---------------|
| `r` (rank) | ? | ... |
| `lora_alpha` | ? | ... |
| `lora_dropout` | ? | ... |
| `target_modules` | ? | ... |
| `learning_rate` | ? | ... |
| `num_train_epochs` | ? | ... |
| `per_device_train_batch_size` | ? | ... |

> **Dica:** Valores utilizados nas aulas (r=8 ou r=16, alpha=16, lr=2e-4) podem servir como ponto de partida, mas justifique se mantiver ou alterar esses valores.

#### 3.2 Eficiência de Parâmetros

Reporte a quantidade de parâmetros treináveis versus o total de parâmetros do modelo, destacando a porcentagem de parâmetros efetivamente atualizados. Utilize a função `print_trainable_parameters()` da PEFT ou equivalente.

#### 3.3 Treinamento

Execute o treinamento e registre:

- Perda (*loss*) de treino e validação por época
- Tempo total de treinamento e hardware utilizado (CPU / GPU / MPS)
- Qualquer dificuldade encontrada e como foi resolvida

---

### 4. Análise e Resultados

#### 4.1 Métricas de Avaliação

Escolha e reporte as métricas mais adequadas à sua tarefa:

| Tarefa | Métricas recomendadas |
|--------|-----------------------|
| Classificação de texto/imagem | Accuracy, F1-Score (macro/weighted), Precision, Recall |
| Geração de linguagem (LM causal) | Perplexidade (exp(loss)), exemplos gerados |
| Sumarização | ROUGE-1, ROUGE-2, ROUGE-L |
| Regressão | MSE, MAE, R² |

Apresente os resultados no **conjunto de teste**, não apenas no de validação.

#### 4.2 Curvas de Treinamento

Apresente gráficos das curvas de *loss* (e/ou métrica principal) por época para os conjuntos de treino e validação. Analise o comportamento:

- O modelo convergiu? Em quantas épocas?
- Há sinais de *overfitting* ou *underfitting*?

#### 4.3 Testes com Dados Externos

Teste o modelo com **pelo menos 5 exemplos** que **não fazem parte do dataset** utilizado no treinamento. Esses exemplos podem ser:

- Coletados manualmente da internet
- Criados pelo próprio aluno
- Pertencentes a um dataset diferente do mesmo domínio

Para cada exemplo, apresente:
- A entrada fornecida ao modelo
- A predição gerada
- A análise qualitativa do resultado (o modelo acertou? errou? por quê?)

#### 4.4 Análise Crítica

Inclua uma seção de conclusão com:

- Principais aprendizados do projeto
- Limitações identificadas no modelo ou no pipeline
- O que poderia ser feito para melhorar os resultados?

---

## Entregáveis

| Entregável | Formato | Obrigatório |
|-----------|---------|-------------|
| Jupyter Notebook executado | `.ipynb` com todas as saídas | Sim |
| Notebook comentado (células markdown) | Integrado ao `.ipynb` | Sim |
| Relatório de análise em PDF | `.pdf` | Não (bônus) |

O notebook deve ser **autocontido**: um avaliador deve conseguir executá-lo do início ao fim em um ambiente com as dependências instaladas.

---

## Critérios de Avaliação

| Critério | Descrição | Peso |
|----------|-----------|------|
| **Seleção e justificativa do modelo** | Escolha adequada e bem fundamentada do modelo | 15% |
| **Seleção e análise do dataset** | Qualidade da análise exploratória e justificativa | 10% |
| **Implementação do fine-tuning** | Correta configuração e execução do pipeline PEFT | 25% |
| **Métricas e curvas de treinamento** | Métricas adequadas à tarefa, análise das curvas | 20% |
| **Testes com dados externos** | Qualidade e análise dos exemplos fora do dataset | 15% |
| **Qualidade da escrita e análise crítica** | Clareza, organização e profundidade da análise | 15% |

**Pontuação total: 100 pontos**

---

## Dicas e Boas Práticas

- **Hardware:** Se não tiver GPU disponível, prefira modelos menores (DistilBERT ~66M params, GPT-2 ~124M params) com LoRA em CPU. Evite QLoRA (NF4) sem CUDA.
- **Overfitting:** Se o *loss* de validação aumentar enquanto o de treino diminui, reduza o número de épocas ou aumente o `lora_dropout`.
- **Datasets do HuggingFace:** Use `datasets.load_dataset("nome_do_dataset")` para carregar diretamente. Consulte a documentação em [huggingface.co/datasets](https://huggingface.co/datasets).
- **Reprodutibilidade:** Fixe as sementes aleatórias no início do notebook (`torch.manual_seed(42)`, `np.random.seed(42)`).
- **Comentários:** Explique suas decisões nas células markdown *antes* do código, não apenas depois.

---

## Referências

- Hu, E. et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. arXiv:2106.09685
- Dettmers, T. et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. arXiv:2305.14314
- HuggingFace PEFT Documentation: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- HuggingFace Transformers: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Material das Aulas 4 e 5 - Disciplina Generative AI & Advanced Networks (FIAP)

---

<div align="center">

*Generative AI & Advanced Networks - FIAP*
*PhD. Julles Mitoura*

</div>
