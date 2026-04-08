<div align="center">

<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg"
     alt="Python"
     width="80"
     height="80"/>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)

<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# Projeto 01 - Aplicação de Redes Neurais
     **Disciplina:** Redes Neurais e Aplicacoes
     **Instituicao:** FIAP
     **Professor:** PhD. Julles Mitoura

## Descricao Geral

Neste projeto, voce ira selecionar uma arquitetura de rede neural, um conjunto de dados e aplicar o pipeline completo de treinamento e avaliacao. O objetivo e consolidar os conceitos vistos ao longo das aulas (implementacao manual do gradiente descendente, redes totalmente conectadas (FNN), redes recorrentes (RNN) e memoria de longo e curto prazo (LSTM)), demonstrando capacidade de escolha, justificativa e analise critica dos resultados.

A entrega consiste em um **Jupyter Notebook** (`.ipynb`) executavel. Um relatorio descrevendo as decisoes e resultados e opcional, podendo ser entregue em **PDF** ou **Markdown** (`.md`).

## Objetivos

Ao final do projeto, o aluno sera capaz de:

- Selecionar e justificar uma arquitetura de rede neural adequada ao problema
- Preparar e analisar um conjunto de dados para treinamento supervisionado
- Implementar o pipeline de treinamento com PyTorch
- Avaliar o modelo com metricas apropriadas ao tipo de tarefa
- Interpretar e comunicar os resultados de forma clara e critica

## Etapas do Projeto

### 1. Selecao da Arquitetura

Escolha **uma** das tres arquiteturas estudadas em aula:

| Arquitetura | Quando usar |
|-------------|-------------|
| **FNN** (Rede Totalmente Conectada) | Dados tabulares sem dependencia temporal |
| **RNN** (Rede Recorrente) | Sequencias curtas ou dados com dependencia temporal simples |
| **LSTM** (Long Short-Term Memory) | Series temporais longas ou com dependencias de longo prazo |

**O notebook deve conter:**

- Nome da arquitetura escolhida
- Descricao da estrutura: numero de camadas, neuronios por camada, funcoes de ativacao
- Justificativa da escolha em relacao ao problema e ao dataset

> **Atencao:** Escolha a arquitetura que faz sentido para o seu problema, nao apenas a mais complexa. Uma FNN bem configurada pode superar uma LSTM em dados tabulares.

### 2. Selecao do Dataset

Escolha um conjunto de dados para sua tarefa. O dataset **nao pode ser** nenhum dos utilizados em aula (regression_example.csv, temperatura.db, jena_climate.csv ou os dados sinteticos gerados nos notebooks).

**A tarefa pode ser de regressao ou classificacao:**

| Tipo de Tarefa | Exemplos de problemas |
|----------------|-----------------------|
| **Regressao** | Previsao de preco, estimativa de temperatura, previsao de demanda |
| **Classificacao** | Diagnostico medico, detecao de fraude, reconhecimento de atividade |

**O notebook deve conter:**

- Nome, fonte e link do dataset
- Descricao: o que os dados representam? Qual o contexto do problema?
- Analise exploratoria:
  - Numero de amostras e de features
  - Distribuicao das classes ou da variavel alvo (grafico)
  - Verificacao de valores ausentes e outliers
  - Estatisticas descritivas (media, desvio padrao, min/max)
- Pre-processamento aplicado: normalizacao, divisao treino/validacao/teste, criacao de janelas deslizantes (para RNN/LSTM)
- Justificativa: por que esse dataset e adequado para a arquitetura escolhida?

**Requisitos minimos do dataset:**

- Minimo de **300 amostras**
- Minimo de **2 features** de entrada
- Deve possuir split de teste separado (ou ser dividido manualmente com proporcao sugerida de 70/15/15)

### 3. Implementacao com PyTorch

Implemente o modelo utilizando PyTorch (`torch.nn.Module`), seguindo o padrao visto em aula.

**O notebook deve conter:**

#### 3.1 Definicao do Modelo

- Classe do modelo herdando de `nn.Module` com metodo `forward()` implementado
- Descricao de cada camada e sua funcao no modelo

#### 3.2 Configuracao do Treinamento

Documente e justifique cada escolha:

| Configuracao | Valor utilizado | Justificativa |
|--------------|----------------|---------------|
| Funcao de perda | ? | ... |
| Otimizador | ? | ... |
| Learning rate | ? | ... |
| Batch size | ? | ... |
| Numero de epocas | ? | ... |
| Dropout (se aplicado) | ? | ... |
| Gradient clipping (se aplicado) | ? | ... |

> **Referencia:** Valores vistos em aula - lr entre 0.001 e 0.005, batch size 32 ou 64, Adam como otimizador padrao. Justifique se mantiver ou alterar esses valores.

#### 3.3 Loop de Treinamento

O loop de treinamento deve registrar a perda de treino e validacao a cada epoca.

### 4. Analise e Resultados

#### 4.1 Metricas de Avaliacao

Reporte as metricas no **conjunto de teste**:

| Tipo de tarefa | Metricas obrigatorias | Metricas opcionais |
|----------------|-----------------------|--------------------|
| **Regressao** | MSE, RMSE, MAE | R², MAPE |
| **Classificacao** | Accuracy, F1-Score | Precision, Recall, AUC-ROC |

#### 4.2 Curvas de Treinamento

Apresente o grafico da perda (*loss*) de treino e validacao por epoca. Analise:

- O modelo convergiu? Em quantas epocas?
- Ha sinais de *overfitting* (loss de validacao sobe enquanto a de treino cai)?
- Ha sinais de *underfitting* (ambas as losses permanecem altas)?

#### 4.3 Visualizacao das Predicoes

- **Regressao:** Grafico de valores preditos versus valores reais (scatter plot ou serie temporal)
- **Classificacao:** Matriz de confusao e relatorio de classificacao por classe

## Relatorio (Opcional)

O relatorio e opcional e pode ser entregue em **PDF** ou **Markdown** (`.md`). Se entregue, deve ter entre **3 e 5 paginas** e cobrir:

1. **Introducao:** Descricao do problema e motivacao da escolha do dataset
2. **Arquitetura escolhida:** Justificativa tecnica da escolha (FNN, RNN ou LSTM)
3. **Analise exploratoria:** Principais descobertas sobre os dados
4. **Decisoes de modelagem:** Hiperparametros escolhidos e justificativas
5. **Resultados:** Metricas, curvas de treinamento e visualizacoes
6. **Conclusao:** O que funcionou, o que nao funcionou e possiveis melhorias

O relatorio deve ser escrito com suas proprias palavras. Nao copie trechos de codigo no relatorio - use graficos, tabelas e descricoes textuais.

## Entregaveis

| Entregavel | Formato | Obrigatorio |
|-----------|---------|-------------|
| Jupyter Notebook executado com saidas | `.ipynb` | Sim |
| Relatorio de analise | `.pdf` ou `.md` | Nao |

O notebook deve ser **autocontido**: todas as celulas devem poder ser executadas do inicio ao fim em um ambiente com as dependencias instaladas. Fixe a semente aleatoria no inicio (`torch.manual_seed(42)`).

## Criterios de Avaliacao

| Criterio | Descricao | Peso |
|----------|-----------|------|
| **Selecao e justificativa da arquitetura** | Adequacao da escolha ao problema e clareza da justificativa | 15% |
| **Analise exploratoria do dataset** | Qualidade e profundidade da analise dos dados | 15% |
| **Implementacao com PyTorch** | Correcao do modelo, loop de treinamento e pre-processamento | 25% |
| **Metricas e curvas de treinamento** | Metricas adequadas a tarefa e analise do comportamento | 20% |
| **Relatorio** | Clareza, organizacao e analise critica dos resultados | 25% |

**Pontuacao total: 100 pontos**

## Dicas e Boas Praticas

- **Normalizacao e essencial:** Use `StandardScaler` ou normalizacao manual. Dados sem normalizar causam instabilidade no treinamento.
- **Comece simples:** Uma FNN com 2 camadas ocultas pode ser um bom ponto de partida antes de tentar RNN/LSTM.
- **Para RNN e LSTM:** Utilize janela deslizante para criar sequencias. O tamanho da janela (`SEQ_LEN`) e um hiperparametro importante.
- **Gradient clipping:** Para RNN, use `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` para evitar gradientes explodindo.
- **Overfitting:** Se o modelo overfita, experimente aumentar o `dropout`, reduzir o numero de camadas ou utilizar mais dados de treinamento.
- **Reproducibilidade:** Fixe a semente: `torch.manual_seed(42)` e `np.random.seed(42)` no inicio do notebook.


<div align="center">

*Redes Neurais e Aplicacoes - FIAP*
*PhD. Julles Mitoura*

</div>