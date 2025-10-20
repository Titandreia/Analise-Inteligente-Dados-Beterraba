# 🧠 Análise Inteligente de Dados de Amostras de Beterraba

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/Licença-Académica-green)
![Status](https://img.shields.io/badge/Status-Concluído-success)

Projeto académico desenvolvido no âmbito da unidade curricular **Análise Inteligente de Dados**, da **Licenciatura em Engenharia Biomédica**.  
Este estudo aplica a metodologia **CRISP-DM** à análise de dados laboratoriais de amostras de beterraba, com o objetivo de compreender e otimizar a **extração de compostos bioativos** (fenólicos, antocianinas e antioxidantes) através de **modelação estatística e machine learning em Python**.

---

## 🎯 Objetivo do Projeto

O trabalho visa analisar dados laboratoriais de amostras de **Beterraba (Beta vulgaris)** para:
- Extrair informações significativas sobre o impacto de variáveis experimentais (solvente, tempo, razão volume/massa e ordem de extração);
- Desenvolver modelos preditivos de **conteúdo fenólico total (TPC)**, **atividade antioxidante (AOA)** e **teor de antocianinas (ANT)**;
- Fornecer **insights aplicáveis** às indústrias alimentar e farmacêutica.

---

## 🧩 Metodologia CRISP-DM

O projeto seguiu todas as fases do processo **CRISP-DM (Cross Industry Standard Process for Data Mining):**

1. **Compreensão do Problema:**  
   Estudo da importância dos compostos bioativos e definição das variáveis críticas.

2. **Compreensão dos Dados:**  
   Análise das variáveis categóricas e contínuas, descrição estatística e leitura do DataFrame a partir de Excel.

3. **Preparação dos Dados:**  
   Limpeza (remoção de *NaN*, duplicados e outliers), substituição de zeros pela média e normalização.

4. **Análise Exploratória:**  
   Visualização com **boxplots**, **histogramas**, **matrizes de correlação** e **gráficos de dispersão**.

5. **Modelação:**  
   Implementação de modelos de regressão, nomeadamente **Random Forest Regressor** e **Regressão Linear**, avaliando o desempenho segundo MSE, RMSE, R² e MAPE.

6. **Avaliação:**  
   Interpretação das métricas, comparação entre modelos e determinação das melhores condições experimentais para maximizar os compostos bioativos.

---

## 📊 Principais Conclusões

- O solvente **Acetona:Água (3)** apresentou melhor desempenho na extração de **TPC** e **ANT**;  
- O **Metanol (1)** foi mais eficaz para a **atividade antioxidante (AOA)**;  
- A razão **Vm-ratio = 20 (100 mL/5 g)** mostrou-se ideal em todos os casos;  
- A **primeira extração (Order = 1)** gera maiores concentrações de compostos bioativos;  
- O tempo de **15 minutos** favorece a **atividade antioxidante**, enquanto **60 minutos** maximiza **ANT**.

---

## ⚙️ Tecnologias e Bibliotecas

- **Python 3.10**
- **Pandas** – tratamento e análise de dados  
- **NumPy** – operações numéricas e normalização  
- **Matplotlib** e **Seaborn** – visualização e análise gráfica  
- **Scikit-learn** – modelação e avaliação preditiva (Random Forest, métricas de erro)  

---

## 🗂️ Estrutura do Projeto
```markdown
Analise-Inteligente-Dados-Beterraba/
├── 📂 docs/
│ └── Trabalho_AID.pdf
│
├── 📂 pythom/
│ └── beetroot_v2.py
│
├── 📄 requirements.txt
└── 📄 README.md
```
## 👩‍🔬 Autoras

| Nome | Função | Contacto |
|------|---------|-----------|
| **Andreia Domingues Fernandes** | Desenvolvimento completo do código, análise estatística, modelação e elaboração do relatório técnico | [andreia2000fernandes@gmail.com](mailto:andreia2000fernandes@gmail.com) |
| **Rita Quaresma** | Apoio na estruturação do relatório e revisão científica | - |

