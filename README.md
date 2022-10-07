# Computer Aided Skin Cancer Identification And Diagnosis
CASCID or Computer Aided Skin Cancer Identification and Diagnosis.


Put database already extracted into 'data' folder. Path to an image should be like 'data\images\PAT_8_15_820.png'

Database URL: https://data.mendeley.com/datasets/zr7vgbcyr2/1 

# Backlog

## Apresentação (Samuel)

- Apresentação
    - Corrigir a escrita
    - Figuras (resolução, contraste)

## Relatório (Gabi)

- Mudanças no relatório
    - Atualizar Tabelas
    - Atualizar Figuras
    - Corrigir escrita
    - Complementar análise de dados

## Melhorias dos modelos

(Dip)
- Terminar modelo Stacked
- Integrar modelo Stacked com API
- Resolver GPU Monstrão
(Todos)
- Criar coluna cor cabelo

    ### Metadados (Samuel)

    - Separação treino/teste considerando pacientes (Colocar como função que retorna x,y train/test no módulo cascid (PAD-UFES))
    - Analisar inputação de dados (Linhas faltantes mas de mesmo paciente?)
    - Reduzir ao máximo número de features dos metadados

    ### Imagens

    - Sift/Surf Feature Extraction para entrada em modelo de classificação
    - Modelo de previsão da cor de cabelo
    - Extrator de cabelo multifuncional
    - Incorporar outros pré-processamentos no pipeline de treino/teste dos modelos (Gabi)
    - Salvar remoção de cabelo em dataset separado (Dip, Gabi)
    - Restruturar CNN (BatchNormalization, Conv2D repetido) (Dip)
    - Classificador binário (Fernando)
    - Fazer testes com dados ISIC (Fernando)
    - Refatorar dataset com [tf.Data](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) (Fernando)

## Protótipo

- Adicionar questionário de metadados no App (Talvez perguntar cor do cabelo visível na imagem) (Fernando)

