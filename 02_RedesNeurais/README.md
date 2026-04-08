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

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)


<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# Redes Neurais Artificiais, Deep Learning e Algoritmos Genéticos
> PhD. Julles Mitoura

Módulo prático que conduz o aluno desde os fundamentos matemáticos de uma rede neural até a implementação com PyTorch  construindo cada componente do zero antes de utilizar frameworks, para que a abstração faça sentido.

---

## Estrutura do Módulo

| Aula | Notebook | Tema | Conteúdo Principal |
|------|----------|------|--------------------|
| 00 | [aula0_redes_neurais.ipynb](./aula0_redes_neurais.ipynb) | Regressão Linear com Gradiente Descendente | Forward pass, MSE, backpropagation manual, loop de treino com NumPy |
| 01 | [aula1_redes_neurais.ipynb](./aula1_redes_neurais.ipynb) | O que é uma Rede Neural? | Neurônio artificial, redes multicamada do zero, normalização Z-score, aplicação em regressão real |
| 03 | [aula3_redes_neurais_fnn.ipynb](./aula3_redes_neurais_fnn.ipynb) | Fully Connected NN com PyTorch | `nn.Module`, autograd, Adam, regressão com CSV real e classificação multiclasse (bullseye) |
| 04 | [aula4_redes_neurais_rnn_01.ipynb](./aula4_redes_neurais_rnn_01.ipynb) | RNN com PyTorch | `nn.RNN`, estado oculto, BPTT, clipping de gradiente, previsão de série temporal e classificação de sequências |
| 05 | [aula5_redes_neurais_lstm_01.ipynb](./aula5_redes_neurais_lstm_01.ipynb) | LSTM com PyTorch | Gates, estado de célula, janela deslizante, previsão de série temporal e classificação de sequências |
| 06 | [aula6_redes_neurais_gru.ipynb](./aula6_redes_neurais_gru.ipynb) | GRU com PyTorch | Reset e update gates, comparação direta com LSTM, regressão de série temporal e classificação de sequências |
| 07a | [aula7_redes_neurais_cnn_1.ipynb](./aula7_redes_neurais_cnn_1.ipynb) | CNN do zero com NumPy | Convolução, ReLU, max pooling, forward e backpropagation manual, filtros treináveis, classificação O×X |
| 07b | [aula7_redes_neurais_cnn_2.ipynb](./aula7_redes_neurais_cnn_2.ipynb) | CNN com Dataset Sintético | Geração programática de dígitos 0–9 com ruído, treino completo com PyTorch |
| 07c | [aula7_redes_neurais_cnn_3.ipynb](./aula7_redes_neurais_cnn_3.ipynb) | CNN com Datasets Reais | `nn.Conv2d`, BatchNorm, data augmentation, classificação de imagens (MNIST e CIFAR-10) |

---

## Progressão Pedagógica

```
Aula 00 → Um neurônio, gradiente descendente na forma mais simples
Aula 01 → Rede multicamada completa: forward, loss, backward, update (só NumPy)
Aula 03 → Mesma rede, agora com PyTorch: autograd, DataLoader, treinamento com GPU
Aula 04 → RNN: recorrência, estado oculto, BPTT e o problema do gradiente que desvanece
Aula 05 → LSTM: memória de longo prazo, gates e solução para o gradiente que desvanece
Aula 06 → GRU: versão simplificada da LSTM, menos parâmetros, convergência mais rápida
Aula 07a → CNN do zero: convolução, pooling e backpropagation com NumPy puro
Aula 07b → CNN com dataset sintético: geração programática de dígitos, pipeline completo
Aula 07c → CNN com datasets reais: Conv2d, BatchNorm, data augmentation, MNIST e CIFAR-10
```

A progressão é intencional: você implementa tudo manualmente antes de usar o framework. Quando o PyTorch é introduzido, cada componente já é familiar. As aulas 04–06 cobrem dados **sequenciais**; as aulas 07a e 07b abrem o domínio de dados **espaciais (imagens)**.

---

## Datasets

| Aula | Dataset | Descrição |
|------|---------|-----------|
| 00 | Sintético | Relação linear $y = wx + b + \varepsilon$ com ruído gaussiano |
| 01 | Sintético + CSV | Dataset gerado e `data/regression_example.csv` (400 amostras, 4 features) |
| 03 | CSV + Sintético | `data/regression_example.csv` para regressão; bullseye sintético (3 classes por faixa de raio) para classificação |
| 04 | Sintético | Série temporal $\sin(t) + 0.5\sin(3t) + \varepsilon$ para regressão; sequências senoidais, lineares e ruído para classificação |
| 05 | Jena Climate + UCI HAR | Jena Climate 2009–2016 (previsão de temperatura); UCI HAR (reconhecimento de atividade humana) |
| 06 | Jena Climate + UCI HAR | Mesmos datasets da Aula 05, comparação direta LSTM × GRU |
| 07a | Sintético | Padrões O×X em matrizes 6×6 com ruído gaussiano |
| 07b | Sintético | Dígitos 0–9 gerados com matplotlib (28×28, cinza) com ruído e variação de posição |
| 07c | MNIST + CIFAR-10 | MNIST (70k dígitos manuscritos, 28×28, cinza); CIFAR-10 (60k imagens coloridas, 32×32, RGB) |

---

## Pré-requisitos

```bash
pip install -r requirements.txt
```

| Biblioteca | Versão mínima | Uso |
|------------|--------------|-----|
| `numpy` | 1.20.0 | Álgebra linear, geração de datasets |
| `matplotlib` | 3.5.0 | Visualizações e gráficos |
| `pandas` | 1.3.0 | Leitura de CSV (aulas 01 e 03) |
| `scikit-learn` | 1.0 | Normalização com `StandardScaler` (aulas 04–06) |
| `torch` |  | Framework de deep learning (aulas 03–07) |
| `torchvision` |  | Datasets e transformações de imagem (aula 07) |

---

## Como executar

Abra qualquer notebook no Jupyter e execute todas as células em sequência (`Run All`). Cada notebook é independente e autocontido.

```bash
jupyter notebook
```
