
# Algorítmo de rede neural Backpropagation usando XOR, OR, AND e caractéres com ruído.
# Especificação do trabalho de IA – ENTREGA 1


O trabalho de Inteligência Artificial versa sobre a rede neural artificial Multilayer Perceptron e abrange a prática
com modelagem, implementação, testes e análise de resultados.
Objetivo: implementar uma rede neural artificial Multilayer Perceptron (MLP), com uma camada escondida e
treinada com o algoritmo Backpropagation em sua versão de Gradiente Descendente - algoritmo de
treinamento discutido em sala de aula.
Quatro conjuntos de dados devem ser usados para treinamento e teste da entrega 1 (são seus arquivos de
entrada):

- Conjunto de dados OR
- Conjunto de dados AND
- Conjunto de dados XOR
- Conjunto de dados CARACTERES (referente ao exercício explicado no livro da L. Fausset)
- Este conjunto possui a versão limpa e a versão com ruído. A versão com ruído é
adequada para usar nos testes.

Em todos os arquivos, a última coluna é o rótulo do dado, as demais são atributos descritivos.

Arquivos de saída úteis para o seu trabalho:

- Um arquivo contendo os parâmetros da arquitetura da rede neural e parâmetros de
inicialização.
- Um arquivo contendo os pesos iniciais da rede.
- Um arquivo contendo os pesos finais da rede.
- Um arquivo contendo o erro cometido pela rede neural em cada iteração do treinamento.
- Um arquivo contendo as saídas produzidas pela rede neural para cada um dos dados de teste.=
Algumas regras gerais
- Os alunos devem ser organizar em grupos de até cinco integrantes.
- Todas as entregas deverão ser feitas via Sistema e-Disciplinas, dentro dos deadlines
estabelecidos neste documento.
o Qualquer eventual problema com o sistema e-disciplina deve ser observado com
antecedência suficiente para que a entrega seja feita pessoalmente para a professora.
Isso significa que o grupo não deve deixar para fazer upload de arquivos no último
minuto possível. Preferencialmente, o upload deve ser feito com uma antecedência
mínima de um dia. O último dia de entrega deve ser deixado para upload de arquivos
que representam apenas ajustes finos no trabalho.
- As implementações deverão ser feitas em linguagens baseadas em Java, C ou Python.
- O código deve ser sempre muito bem documentado (em DETALHES) de forma que seja simples
identificar passagens do código que são importantes para a verificação do entendimento do
grupo sobre a lógica que implementa uma rede neural artificial.
- Não é permitido fazer uso de nenhuma biblioteca que implementa as funções de uma rede
neural artificial. Seja permitido apenas o uso de funções que implemente multiplicação de
matrizes. 
- Bibliotecas que implementem funções de I/O e funções de PLOT podem ser livremente usadas
para implementar entrada de dados e interface para exposição de resultados (a interface pode
ser construída em modo gráfico, modo texto ou apenas com gravação das saídas do algoritmo
em arquivos .txt ou .csv).
- Os vídeos poderão conter, no máximo, 15 minutos de gravação.
- Data limite para o upload dos arquivos referentes à Entrega 1: 31 de março.

Entregas:

1. Código desenvolvido pelo grupo: o grupo deverá fazer o upload de todos os arquivosreferentes
à implementação da sua rede neural MLP. O código que implementa a rede neural em si
(definição de estruturas de dados e implementação do algoritmo de treinamento e de teste da
rede neural) deverá estar massivamente comentado. O conhecimento do grupo será avaliado
mediante a análise do código comentado

2. Vídeo de apresentação: o grupo deverá gravar um vídeo no qual apresenta sua codificação, em
detalhes, e apresenta a execução de sua codificação sobre os conjuntos de dados de teste.
Detalhes esperados no vídeo (listagem não exaustiva):
  -  Apresentação dos integrantes do grupo (nomes completos).
  - Explicações sobre o código, ressaltando os detalhes que implementam as estruturas de
dados utilizadas e a lógica de implementação do algoritmo de treino e de teste.
  - Detalhamento das variáveis que determinam a arquitetura da rede neural artificial,
indicando os valores que elas assumem para um teste em um conjunto de dados dentre
os conjuntos OR, AND e XOR e um teste no conjunto de dados CARACTERES. Usem
ambientes de depuração, se o grupo achar conveniente para apresentar esse
detalhamento.
  - Apresentação sobre a organização dos arquivos de entrada e de saída utilizados em sua
implementação. Mostrem os arquivos no sistema de diretórios usados, se o grupo
achar conveniente para fazer essa apresentação.
  - Demonstração da execução do treinamento da rede neural artificial. Faça uso de
interface gráfica ou interface em modo texto para ecoar saídas intermediárias de sua
rede neural, como por exemplo, ecoar um contador de épocas, apresentar os pesos
iniciais e finais da rede neural, ecoar a resposta da rede neural para alguns dados de
teste.

Este documento pode ser complementado com textos que esclareçam dúvidas que os alunos podem levantar
durante a execução do trabalho. 
