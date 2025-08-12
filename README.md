# Desafio IA — Pipeline Completo (Visão Computacional + Áudio)

Este projeto implementa uma solução ponta a ponta para análise de uma cena simples capturada em vídeo, cumprindo os requisitos do **Desafio IA**:

- **Separação de canais**: extração de frames de vídeo e áudio.
- **Armazenamento e tratamento de imagens**: conversão para FITS com metadados, persistência em MongoDB (GridFS).
- **Processamento de imagens**: equalização de histograma.
- **Engenharia de features e modelagem**: extração de descritores, treino e avaliação de modelo.
- **Análise de áudio**: transcrição automática via Whisper.

---

## Estrutura do Projeto

```
desafio-ia/
├── app/                          
│   ├── data/                      # Armazena dados de entrada, saída e processados
│   │   ├── fits/                  
│   │   ├── frames/                 # Frames extraídos do vídeo
│   │   │   └── frames_tmp/         # Imagens PNG geradas a partir do vídeo
│   │   ├── hist_from_db/           # Histogramas de frames salvos a partir do banco de dados
│   │   └── raw/                    # Dados brutos, incluindo vídeo original e áudio extraído
│   ├── notebook/                  # Notebooks Jupyter para desenvolvimento e análise
│   │   └── desafio_ia.ipynb      
│   └── src/                       # Código-fonte organizado em módulos Python
│       ├── audio_utils.py          # Funções utilitárias para processamento de áudio
│       ├── db.py                   # Funções para interação com banco de dados
│       ├── features.py             # Extração e manipulação de features (imagens/áudio)
│       ├── image_utils.py          # Funções utilitárias para processamento de imagens
│       ├── model.py                # Definição e carregamento do modelo de machine learning
│       └── video_utils.py          # Funções utilitárias para processamento de vídeo
│
├── infra/                         # Configuração de infraestrutura e ambiente
│   ├── docker-compose.yml          # Orquestração dos serviços Docker
│   ├── Dockerfile                  # Configuração da imagem Docker da aplicação
│   └── Makefile                    # Comandos automatizados para build, execução e utilidades
│
├── poetry.lock                    # Arquivo de lock do Poetry, garante reprodutibilidade das deps
├── pyproject.toml                  # Configuração do Poetry e dependências do projeto
└── README.md                       # Documentação principal do projeto
```

---

## Pré-requisitos

- **Docker** e **Docker Compose** instalados.
- **Poetry** 
- **Git**
- **make (GNU Make)**
- Vídeo de input (`video.mp4`) em `app/data/raw/`.
- Faça o downlod do víde: [google drive](https://drive.google.com/drive/folders/1cy9tlC422eqs8baR8Cj3GUT4ovhP9AEu?usp=drive_link)

---

## Executando no Docker

1. **Subir ambiente**
```bash
cd infra
make up
```

2. **Abrir o notebook**
```bash
make notebook
```
O Jupyter Lab ficará disponível em: [http://localhost:8888](http://localhost:8888).

---

## Etapas do Pipeline

1. **Coleta e Pré-processamento de Dados**
   - Extração de frames a cada 0.1s (`OpenCV`).
   - Extração do áudio em WAV 16kHz mono (`ffmpeg-python`).

2. **Armazenamento e Tratamento de Imagens**
   - Conversão de frames para FITS (`astropy`) com metadados no header.
   - Persistência no MongoDB (GridFS) + coleção de metadados.
   - Leitura direta do banco para cálculo de histogramas.

3. **Análise e Processamento de Imagens**
   - Equalização de histograma.
   - Comparação entre imagens originais e equalizadas.

4. **Engenharia de Features e Modelagem**
   - Extração de histogramas BGR e GLCM.
   - Rotulagem automática por faixas de tempo.
   - Treinamento de classificador (`RandomForest`).
   - Avaliação com matriz de confusão e classification report.

5. **Visualização de Resultados**
   - Exibição de frames com rótulos previstos e confiança.

6. **Análise de Áudio**
   - Transcrição da narração (`whisper`), modelo `base`.

---

## Dependências Principais

- `opencv-python`
- `astropy`
- `pymongo`
- `scikit-learn`
- `matplotlib`
- `ffmpeg-python`
- `whisper`
- `jupyterlab`

---

## Desenvolvimento Local (opcional)

Caso queira rodar fora do Docker:
```bash
poetry install
poetry run jupyter lab
```

---

## Critérios de Avaliação Atendidos

✅ Extração e tratamento de imagens  
✅ Armazenamento estruturado com metadados  
✅ Pré-processamento e engenharia de features  
✅ Treinamento e avaliação de modelo  
✅ Transcrição de áudio  
✅ Organização modular e documentada

---
