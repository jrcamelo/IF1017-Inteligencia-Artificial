# Inteligência Artificial (2020.1)
## Relatório de Experimentos
## Star Type Classification
<hr>

## Introdução

Por meio deste relatório, busca-se pôr em prática os conhecimentos adquiridos na área de inteligência artificial. Alguns problemas de classificação foram identificados para que seja possível analisar a acurácia dos diferentes algoritmos apresentados ao longo do curso.  

A base de dados escolhida para este experimento foi um conjunto de estrelas e suas respectivas características que podem ser utilizadas na predição de suas propriedades, envolvendo temperatura, luminosidade, raio, circunferência, cor, classificação e tipo, com seus respectivos dados sendo disponibilizados pela NASA.  

É uma base de dados recente, adicionada no início do mês de Abril de 2021, possuindo dados normalizados de 240 estrelas, com 7 colunas para as características, e uma distribuição uniforme de 40 estrelas entre seus 6 tipos.
Esta base de dados pode ser obtida no Kaggle pelo seguinte link:
<a href="https://www.kaggle.com/brsdincer/star-type-classification">Star Type Classification / NASA</a>

<hr>


## Fundamentos

### k-Nearest Neighbors (kNN)
Conhecido por ser um dos algoritmos mais comuns e simplistas utilizados em Machine Learning. Pode ser classificado como um algoritmo lazy, por não demandar muitos requisitos computacionais na fase de treino, mas costuma ter uma classificação mais extensa e complexa. 

Como a tradução livre do termo, o algoritmo é responsável por determinar os vizinhos mais próximos de um determinado dado. Possui o “k” como parâmetro, que corresponde aos vizinhos mais próximos e classificados do dado em questão.  

O algoritmo deve ser capaz de analisar uma distância entre cada dimensão de um dado classificado e um dado não classificado. Com posse de tais distâncias, as mesmas são ordenadas para encontrar seus k (parâmetro utilizado que deve ser ímpar para garantir um funcionamento mais adequado) vizinhos mais próximos, determinando assim qual será sua classificação com base na proximidade.  

### Decision Tree

É gerado um procedimento onde cada feature pode ser analisada e levantar um questionamento binário. À medida que o dado corresponde ou não ao questionamento atual, ele pode gerar um novo ramo da árvore (profundidade) e seguir sua direção correspondente. Cada questionamento é capaz de gerar um novo ramo, no final esses ramos se tornam folhas, que por sua vez entregam a classificação correspondente do dado.  

O procedimento é realizado levando em conta o valor das características, até que todo conjunto de dados possam ser avaliados e classificados.  

### Random Forest  

Semelhante a Decision Tree, contudo os limiares de decisão são escolhidos de forma aleatória. Gera diversas Decision Trees, combinando seus resultados em um modelo poderoso. A esse método, se atribui o termo ensemble, combinando diferentes modelos e algoritmos robustos para chegar em um único resultado comum.  

Com a média da profundidade dos nós onde as features estão localizadas, é possível estimar a “importância” de cada feature, onde a mesma tende a estar na copa das árvores.    

<hr>

## Metodologia

### Base de Dados

A base de dados de Star Type Classification foi lida utilizando o Pandas, e tratada uniformemente entre os diferentes métodos de aprendizado de máquina.  

Suas características são Temperatura, Luminosidade, Raio, Magnitude, SMASS (Classificação espectral de asteroides) e Classificação (Classificação estelar).  

A característica prevista foi a Classificação, que se divide uniformemente entre 6 tipos: Anã vermelha, Anã marrom, Anã branca, Sequência principal, Super gigantes, Hiper gigantes. Cada uma tendo 40 amostras.  

<img src="results/Database - Boxes.png" width="300px" alt="Comparação entre Características e Classificação"/>

<img src="results/Database - PairPlot.png" width="300px" alt="Plotagem em pares de todas características"/>

### Separação

Foi feita uma validação cruzada na base de dados para todas as metodologias, separando 75% para treinamento e 25% para testes. Devido ao seu tamanho, é sugerido usar uma separação de 50% para cada tipo, porém foram identificados, em média, resultados negativos ao fazer isso em todas as metodologias utilizadas.  

Durante a separação, houve um esforço em manter uma quantidade balanceada dos targets, sendo sempre feita uma estratificação dos dados.  

### Normalização

Na base de dados não havia nenhum item nulo, porém as Cores estavam desorganizadas, tendo cores muito similares, problemas na capitalização e espaçamento, cores muito exatas com somente uma amostra, e etc.  

Portanto, houve um esforço em corrigir esses problemas durante a inicialização da base de dados para todas as metodologias.  


| Cores iniciais             | Qtd |
|--------------------|-----|
| Red                | 112 |
| Blue               | 56  |
| Blue-white         | 26  |
| Blue White         | 10  |
| yellow-white       | 8   |
| White              | 7   |
| Blue white         | 4   |
| white              | 3   |
| Yellowish White    | 3   |
| yellowish          | 2   |
| Whitish            | 2   |
| Orange             | 2   |
| White-Yellow       | 1   |
| Blue-White         | 1   |
| Pale yellow orange | 1   |
| Yellowish          | 1   |
| Orange-Red         | 1   |

| Cores normalizadas  | Qtd |
|--------------|-----|
| Red          | 112 |
| Blue         | 56  |
| Blue-White   | 41  |
| White        | 12  |
| Yellow-White | 11  |
| Yellow       | 5   |
| Orange       | 3   |


A base de dados então foi achatada, transformando as características textuais, Cor e SMASS, em características numéricas, agrupando-as em números.  

Por fim, foi feito um escalonamento das características, mudando os valores para se encaixarem em uma escala mais uniforme, de acordo com a variância de cada valor, porém só foi usado no kNN.  

### k-Nearest Neighbor

Foi usado a base de dados normalizada, achatada e escalonada para o kNN, sem complicações relevantes.  

Foram feitos testes com k variando entre 1 e 40. Para cada k foram feitas 500 iterações, somando os seus erros na base de testes e retornando uma média para cada k. Ao fim, essas médias foram plotadas em um gráfico e analisadas.  

Em todas iterações a base de dados foi embaralhada novamente, de forma que os testes fossem balanceados.  

Como testes adicionais, características foram removidas para tentar obter melhores resultados. Além disso, foi testada a mesma base de dados com as cores como target ao invés da classificação.  

### Decision Tree

Para o Decision Tree, a base de dados foi normalizada e achatada, sem necessidade de fazer escalonamento.  

Os testes foram analisados de duas maneiras: manualmente, analisando cada árvore gerada, as importâncias de cada característica e seus resultados; e automaticamente, gerando 500 árvores com uma base de dados misturada, sem plotar uma imagem e analisando sua acurácia.  

Como teste adicional, o mesmo método foi aplicado usando as cores como target ao invés da classificação.  

### Random Forest

Para o Random Forest, a base de dados foi normalizada e achatada, sem necessidade de fazer escalonamento.  

Os testes foram feitos com 100 execuções em florestas de 500 árvores, com variação de parâmetros discutida nos resultados. A análise foi feita por visualização de árvores individuais, pela acurácia da execução e pela média da acurácia das 100 execuções.   

Como teste adicional, foi aplicado o mesmo método usando as cores como target ao invés da classificação.  

<hr>

## Resultados

### k-Nearest Neighbor

O kNN se provou a metodologia mais simples, e, por sua vez, o mais preciso quando os parâmetros corretos foram usados.  

Em todos os diferentes testes, o k mais preciso foi sempre o k = 1, sendo clara a diminuição da acurácia em geral conforme o k aumentava.  

<img src="results/kNN - 500 runs.png" width="300px" alt="Erro nas 500 execuções para cada k, todas as características"/>

Com todas as características, obteve-se uma acurácia satisfatória de 99.16% na média das 500 execuções, estabilizando em cerca de 75% a partir do k = 29.  

Porém, baseando-se na importância de cada característica, foram feitos mais testes, que obtiveram resultados relevantes.  

<img src="results/kNN - 500 runs - Without Color and SMASS.png" width="300px" alt="Erro nas 500 execuções para cada k, sem Cor ou SMASS"/>

Removendo as características textuais achatadas, que não faziam sentido numérico, obteve-se um aumento significativo na precisão do k = 1, que ficou em 99.74%. A precisão do k > 29  também melhorou consideravelmente, ficando entre 95 e 85%.  

Foi decidido remover ainda mais características de baixa importância, começando pela Temperatura, e então a Luminosidade. Porém, os resultados foram negativos em ambos os casos.  


<img src="results/kNN - 500 runs - With just Luminosity, Radius and Magnitude.png" width="300px" alt="Erro nas 500 execuções para cada k, sem Cor, SMASS ou Temperatura"/>

<img src="results/kNN - 500 runs - With just Radius and Magnitude.png" width="300px" alt="Erro nas 500 execuções para cada k, com somente Raio e Magnitude"/>

Ao remover a Temperatura, a precisão no k = 1 voltou ao 99.02% do primeiro teste, e ao remover a Luminosidade, a precisão caiu um pouco mais, para 98.83%.  

|        | Todas Características | Sem textuais | Sem Temperatura | Só Raio e Magnitude  |
|--------|-----------------------|--------------|-----------------|----------------------|
| k = 1  | 99.16%                | 99.74%       | 99.02%          | 98.83%               |
| k = 29 | 83.36%                | 97.18%       | 78.91%          | 77.47%               |
| k = 39 | 74.02%                | 89.64%       | 73.89%          | 74.18%               |

Além desses resultados, também foi testado a previsão de cores, e os resultados foram positivos.  

<img src="results/kNN - 500 runs for Color.png" width="300px" alt="Erro nas 500 execuções para cada k, prevendo a Cor"/>

No k = 1, a precisão ficou de 98.11%, chegando a 74.65% em k = 39, o que foi uma surpresa, já que a base de dados foi feita para prever a classificação estelar.  

### Decision Tree

A Decision Tree foi uma metodologia bem visualizável, e consequentemente compreensível. Houve um problema durante o desenvolvimento onde a Classificação da estrela estava sendo usada para prever a sua Classificação, alcançando 100% de acurácia, mas foi devidamente corrigido.  

Possivelmente devido à simplicidade da base de dados, as árvores em geral tiveram o menor tamanho possível para a quantidade de Classificações, com 11 nós, sendo somente 5 deles condicionais.  

<img src="results\Decision Tree - Type.png" width="300px" alt="Árvore comum com 11 nós, 100% de acurácia"/>

Como mostrou a análise de importância de características abaixo, o Raio, seguido da Magnitude, foram as características mais relevantes para as árvores que conseguiram boa acurácia.  

<img src="results\RandomForest - Feature Importance.png" width="300px"/>

Além disso, também foi analisada a média da acurácia de 500 execuções de Decision Trees, todas execuções tendo uma nova mistura da base de dados, alcançando uma precisão de 98.51%.  

Similar aos resultados obtidos com o kNN, houve uma melhora na precisão ao remover o SMASS, alcançando 99.30%. E em seguida uma melhora um pouco menor ao remover também a Cor, com 99.35%. Porém, ao remover também a Temperatura, a precisão voltou a 99.31%  

|          | Todas Características | Sem SMASS | Sem Temperatura | Só Raio e Magnitude  |
|----------|-----------------------|-----------|-----------------|----------------------|
| Acurácia | 98.51%                | 99.30%    | 99.35%          | 99.31%               |

Além disso, foram feitos testes usando Cor como característica prevista, mudando drasticamente as características importantes.

Porém, os resultados não foram tão satisfatórios, criando árvores enormes, com em média 36 nós, e uma acurácia média de 89.90% em 500 execuções.  

<img src="results\Decision Tree - Color.png" width="300px"/>

Tentou-se então melhorar a acurácia removendo as características menos relevantes, como Classificação e Magnitude, porém houve uma diminuição na acurácia, com 89.51%, e deixando somente Temperatura e SMASS, chegou a 89.11% de acurácia.  


### Random Forest

O Random Forest por sua vez teve bons resultados sem remover nenhuma característica, alcançando 99.65% de acurácia em 100 execuções e 500 árvores, melhorando um pouco com 1000 árvores, para 99.70%.  

Removendo SMASS e Cor, houve uma pequena diminuição na acurácia, chegando a 99.61%.  

Foram analisadas também algumas árvores geradas, consequentemente similares às encontradas na metodologia de Decision Tree.  

<img src="results\RandomForest - Tree 1 on Random Forest.png" width="300px" alt="Primeira árvore de uma floresta"/>

<img src="results\RandomForest - Tree 500 on Random Forest.png" width="300px" alt="Última árvore da mesma floresta"/>

Além disso, também foi analisada a Cor como target, porém os resultados não foram satisfatórios, com 89.33% de acurácia e sem melhora ao remover características.  

<hr>

## Conclusões

|                           | k-Nearest Neighbor | Decision Tree | Random Forest  |
|---------------------------|--------------------|---------------|----------------|
| Melhor acurácia para tipo | 99.74%             | 99.35%        | 99.70%         |
| Melhor acurácia para cor  | 98.11%             | 89.51%        | 89.33%         |

O estudo conclui que as três metodologias podem ser utilizadas para predição da Classificação estelar usando as características desta base com taxas de acurácia satisfatórias e bem próximas entre eles.  

Ao tentar prever a Cor, uma característica não planejada como target da base de dados, Decision Tree e Random Forest tiveram uma precisão decente, mas o kNN se provou especialmente adequado para o trabalho.   

Os testes também mostraram que mesmo tendo um bom resultado em certas situações ao remover características com baixa importância, em outros momentos os resultados eram ínfimos e muitas vezes negativos.  

A base de dados escolhida foi satisfatória, mas apresentou alguns desafios, como o baixo número de amostras, sendo mitigado de certa parte por múltiplas execuções com uma mistura na base de dados e uma estratificação baseada no target escolhido; assim como a heterogeneidade da característica textual de Cor, trazendo problemas de capitalização, separação e granularidade, sendo mitigado por edição racionalizada do seu texto em favor de agrupamento.  
