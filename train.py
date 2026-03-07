# !uv pip install mlflow -qqq

"""###**Importação de Pacotes Necessários**

Nesta etapa, são importados os pacotes necessários para a execução do projeto.

- O TensorFlow e o Keras são fundamentais para a construção e treinamento de redes neurais, enquanto o Pandas e o NumPy são utilizados para manipulação e análise de dados.
- A biblioteca Scikit-learn oferece ferramentas para avaliação de modelos e pré-processamento dos dados, como a normalização e a divisão em conjuntos de treinamento e teste.
- O Plotly é utilizado para visualização de dados, permitindo a criação de gráficos interativos. Além disso, a configuração para ignorar avisos é realizada, garantindo que a execução do código seja mais limpa e focada nos resultados relevantes.

Essa preparação é essencial para o desenvolvimento eficiente do modelo de aprendizado profundo que será implementado a seguir.
"""

# Imports padrão da biblioteca Python
import os
import random
import warnings

# Imports de bibliotecas de terceiros (ordenados alfabeticamente)
import mlflow
from tensorflow import keras
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf

# Imports de módulos específicos de bibliotecas de terceiros (ordenados alfabeticamente)
from keras import layers, models, utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Ignora avisos, o que pode ser útil em notebooks para manter a saída limpa
warnings.filterwarnings('ignore')

"""# Definindo funções adicionais"""

def reset_seeds(seed: int = 42) -> None:
    """
    Reset and synchronize random seeds across Python, NumPy, and TensorFlow
    to improve experiment reproducibility.

    This function also enables deterministic TensorFlow operations when
    supported by the execution environment (CPU/GPU).

    Parameters
    ----------
    seed : int, optional
        Seed value used to initialize all random number generators.
        Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

"""###**Carregamento do Dataset**

Nesta etapa, o conjunto de dados é carregado diretamente de uma URL, utilizando a biblioteca `pandas`. O arquivo CSV contém informações relevantes para a análise de diabetes, que serão utilizadas para treinar e avaliar o modelo de rede neural.
"""

data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')

"""###**Separação de Variáveis**

Neste passo, as variáveis independentes e dependentes são extraídas do conjunto de dados. As variáveis independentes, que são utilizadas como entrada para o modelo, são selecionadas a partir de todas as colunas, exceto a última. A variável dependente, que representa a classe a ser prevista, é extraída como a última coluna do conjunto de dados. Essa separação é fundamental para preparar os dados para o treinamento da rede neural, permitindo que o modelo aprenda a partir das características fornecidas e faça previsões baseadas nelas.
"""

X =data.drop(["fetal_health"], axis=1)
y=data["fetal_health"]

"""###**Normalização dos Dados com StandardScaler**

A normalização dos dados é um passo crucial no pré-processamento, especialmente em modelos de aprendizado de máquina. Neste trecho, a transformação dos dados é realizada utilizando o `StandardScaler`, que ajusta os dados para que tenham média zero e desvio padrão um. Isso é importante para garantir que todas as características contribuam igualmente para o treinamento do modelo, evitando que variáveis com escalas maiores dominem o aprendizado. Após a aplicação do `fit_transform`, os dados de entrada `X` são atualizados com os valores normalizados, prontos para serem utilizados na construção e treinamento da rede neural.
"""

columns_names = list(X.columns)
scaler = StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

"""###**Separação dos Dados em Conjuntos de Treino e Teste**

A separação dos dados em conjuntos de treino e teste é uma etapa crucial no desenvolvimento de modelos de aprendizado de máquina. Neste processo, os dados são divididos de forma que 25% deles sejam reservados para teste, permitindo a avaliação do desempenho do modelo em dados que não foram utilizados durante o treinamento. A divisão é realizada de maneira estratificada, garantindo que a proporção das classes no conjunto de teste seja a mesma que no conjunto original, o que é especialmente importante em problemas de classificação desbalanceada.

Além disso, as variáveis de saída (y_train e y_test) são convertidas em uma representação categórica, facilitando o treinamento da rede neural, que requer que as classes sejam representadas de forma binária. Por fim, as dimensões dos conjuntos de dados resultantes são exibidas, proporcionando uma visão clara da quantidade de dados disponíveis para treinamento e teste.
"""

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

y_train = y_train -1
y_test = y_test - 1

"""# 4 - Criando o modelo e adicionando as camadas"""

data_shape = X_train.iloc[0].shape

# Resetando as sementes para reprodutibilidade
reset_seeds()

# Criando o modelo sequencial
model = models.Sequential()

# camada de entrada do modelo
model.add(keras.Input(shape=data_shape))

# camadas ocultas
model.add(layers.Dense(units=10, activation='relu'))
model.add(layers.Dense(units=10, activation='relu'))

# camada de saida
model.add(layers.Dense(units=3, activation='softmax'))

"""###**Compilação do Modelo de Rede Neural**

A compilação do modelo é um passo crucial na construção de redes neurais, onde são definidos os parâmetros que guiarão o treinamento. Neste caso, a função de perda escolhida é a `categorical_crossentropy`, que é adequada para problemas de classificação multiclasse, como o que estamos abordando. Além disso, a métrica de avaliação utilizada é a `accuracy`, que permitirá monitorar a proporção de previsões corretas durante o treinamento e a validação do modelo. Com essa configuração, o modelo está pronto para ser treinado com os dados preparados anteriormente.
"""

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""### **Usando MLflow**

Esta célula configura o MLflow para realizar o rastreamento de experimentos de aprendizado de máquina utilizando um servidor remoto hospedado no DagsHub. As credenciais de autenticação são fornecidas por meio de variáveis de ambiente, permitindo a comunicação segura com o serviço de tracking.

Em seguida, a URI do servidor de rastreamento é definida, estabelecendo o servidor remoto do MLflow como o destino padrão para todos os logs de experimentos gerados neste notebook.

Por fim, o autologging do Keras é ativado. Com essa configuração, o MLflow registra automaticamente parâmetros de treinamento, métricas por época, exemplos de entrada, assinaturas do modelo e o próprio modelo treinado, sem a necessidade de instrumentação manual do código. Essa abordagem promove reprodutibilidade, rastreabilidade e aderência às boas práticas de DataOps e MLOps.
"""

os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/mlops-ead.mlflow')

mlflow.tensorflow.autolog(log_models=False,
                     log_input_examples=True,
                     log_model_signatures=True)

"""###**Treinamento e Avaliação do Modelo**

Nesta etapa, o modelo de rede neural é treinado utilizando os dados de treinamento previamente preparados. O treinamento ocorre por meio de 50 épocas, onde o conjunto de dados é dividido em um subconjunto de validação de 20% para monitorar o desempenho do modelo durante o treinamento. A ponderação das classes é aplicada para lidar com possíveis desbalanceamentos nos dados, garantindo que o modelo aprenda de forma mais equilibrada. Após o treinamento, o desempenho do modelo é avaliado utilizando o conjunto de teste, onde são calculadas a perda e a acurácia. A acurácia é então exibida, fornecendo uma métrica clara sobre a eficácia do modelo em classificar corretamente os dados de teste.

#### **Gerenciamento de Contexto**

No MLflow, os gerenciadores de contexto definem escopos de execução que controlam como experimentos e traces são registrados.

O contexto `mlflow.start_run()` estabelece uma execução de experimento, que funciona como o contêiner de mais alto nível para o registro de parâmetros, métricas, artefatos e metadados relacionados a uma única execução de um fluxo de trabalho de aprendizado de máquina.

Dentro de uma run ativa, o contexto `mlflow.start_span()` cria um contexto de tracing que representa uma etapa ou operação específica do fluxo de trabalho. Um span mede o tempo de execução e delimita o início e o fim dessa etapa, permitindo observabilidade detalhada do pipeline.

Em conjunto, esses contextos permitem que o MLflow associe traces detalhados a uma execução específica de experimento, melhorando a reprodutibilidade, a depuração e a visibilidade operacional.
"""

with mlflow.start_run(run_name='experiment_mlops_ead') as run:
  with mlflow.start_span(name="model_training"):
        model.fit(
            x=X_train.to_numpy(),
            y=y_train.to_numpy(),
            epochs=50,
            validation_split=0.2
        )