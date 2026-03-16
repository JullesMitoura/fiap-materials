# Docker: Machine Learning para Produção

> Foco: Engenharia de ML, Reprodutibilidade, Produção, Workflow real

Curso prático que conduz o aluno desde a organização de um projeto de ML até o deploy completo de um modelo em produção usando Docker — com CI/CD, versionamento de artefatos e serving via API.

---

## Estrutura do Curso

| Aula | Tema | Entrega |
|------|------|---------|
| 01 | [Setup do projeto de ML](./modulo1/) | Repositório base funcional |
| 02 | Containerizando o projeto | Projeto rodando em container |
| 03 | Separando treino e inferência | Dois containers com responsabilidades claras |
| 04 | Otimização de imagens para ML | Imagens menores e mais rápidas |
| 05 | Gerando e versionando artefatos de modelo | Modelo versionado fora da imagem |
| 06 | Pipeline de build com GitHub Actions | CI funcionando |
| 07 | Publicação da imagem | Imagem publicada e reutilizável |
| 08 | Serving do modelo | API de inferência rodando em Docker |
| 09 | Ambiente de produção | Pipeline completo local |
| 10 | Workflow completo de ML em produção | Projeto finalizado |

---

## Case do Curso

O projeto de ML usado ao longo do curso é um **classificador de qualidade de vinho** (Wine Quality) com `scikit-learn`. A escolha foi intencional:

- Dataset leve, sem download externo (disponível via `sklearn.datasets`)
- Problema de classificação multi-classe realista
- Fácil de entender, mas com estrutura de código similar à produção
- Permite demonstrar todos os conceitos de Docker sem distração de complexidade de dados

---

## Jornada do Aluno

```
Aula 01-02 → Projeto rodando em container
Aula 03-05 → Boas práticas de Docker para ML
Aula 06-07 → Automação com CI/CD
Aula 08-09 → Serving e ambiente de produção
Aula 10    → Pipeline end-to-end completo
```

---

## Pré-requisitos

- Python 3.10+
- Docker instalado e rodando
- Git e conta no GitHub
- Conta no Docker Hub (necessário a partir da Aula 7)

---

## Como navegar

Cada módulo tem seu próprio `README.md` com:
- Contexto teórico da aula
- Passo a passo prático
- Comandos prontos para executar
- Checklist de entrega
