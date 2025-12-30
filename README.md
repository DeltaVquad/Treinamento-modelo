# Visão Computacional - Treinamento do modelo

Este repositório contém o pipeline de processamento e treinamento para a detecção de objetos da equipe DeltaV Drones. O fluxo abrange desde a geração de dados sintéticos até a exportação do modelo otimizado.

## Scripts de Geração de Dataset

Os scripts utilizam a técnica de colagem sintética para aumentar a variabilidade do dataset e reduzir falsos positivos encontrados nos testes iniciais com o Webots.

### Preparação do ambiente
```bash 
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Estrutura de Pastas Necessária

Para executar os scripts localmente, organize os arquivos conforme abaixo:

* `input_objs/`: Imagens da plataforma em `.png` com fundo transparente.
* `input_bgs/`: Imagens de fundo (grama, asfalto, terrenos diversos).
* `output/`: Diretório onde as imagens geradas serão salvas.

### Como Usar

1. Instale as dependências: `pip install pillow numpy albumentations`.
2. **v1 (`gerar_dataset.py`)**: Gera imagens com aumentação básica de brilho, contraste e perspectiva.
3. **v2 (`gerar_dataset_v2.py`)**: Recomendado para produção. Inclui sombras dinâmicas, oclusões aleatórias e controle de colisão entre objetos para maior realismo.

---

## Treinamento (Google Colab)

O treinamento é realizado através do notebook `treino_ncnn.ipynb`, otimizado para YOLOv11.

### Passo a Passo

1. **Preparação**: Exporte as anotações do CVAT no formato **YOLO 1.1** e faça o download do arquivo `.zip`.
2. **Upload**: Suba o arquivo `.zip` para o seu Google Drive.
3. **Configuração**:
* Abra o notebook no Colab e conecte ao ambiente com GPU (T4).
* Na célula de configuração, altere a variável `NOME_ZIP` para o nome exato do arquivo subido no Drive.


4. **Execução**: Rode todas as células. O script irá:
* Extrair e reestruturar os dados para o padrão YOLO.
* Iniciar o treino por 100 épocas.
* Exportar o modelo final nos formatos `.pt` e `NCNN`.

---

## Teste e Inferência Local

Para testar o modelo em tempo real no Manjaro utilizando a webcam (Logitech Brio):

1. Instale a biblioteca da Ultralytics: `pip install ultralytics opencv-python`.
2. Certifique-se de que o arquivo do modelo (`.pt`) está no mesmo diretório que o `main.py`.
3. Execute o script: `python main.py`.
* O script está configurado com um limite de confiança de **0.85** para filtrar detecções imprecisas.

---

## Resumo de Tecnologias

| Componente | Ferramenta/Biblioteca |
| --- | --- |
| Linguagem | Python 3.1x |
| Processamento de Imagem | OpenCV, PIL, Albumentations |
| Arquitetura do Modelo | YOLOv11 Nano |
| Inferência | NCNN / PyTorch |
