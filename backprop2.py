import csv

import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:16.3f}'.format}, linewidth=130)

class MultiCamadaPerceptron(object):
    # Todo valor comentado abaixo refere-se com input de 2 neuronios, camada escondida de 2 neuronios e output 1.
    # Método construtor para a MultiCamadaPerceptron.
    # Recebe o num de inputs, camada_escondida e output em ordem.
    def __init__(self, inputs=2, camada_escondida=[2], outputs=1):
        self.inputs = inputs
        self.camada_escondida = camada_escondida
        self.outputs = outputs
        self.errosPorInteracao = []
        self.valorOutput = []

        # Cria uma representacao das layers.
        layers = [inputs] + camada_escondida + [outputs]
        print("Layers somados: {}".format(layers))

        # Cria os pesos com valores aleatorios.
        # (len(layers) - 1) resulta no valor 2.
        pesos = []
        for i in range(len(layers) - 1):
            # Cria matriz 2x2 e 2x1 com valores aleatorios entre 0 a 1
            w = np.random.rand(layers[i], layers[i + 1])
            pesos.append(w)
        # guarda na variavel self.peso
        self.pesos = pesos

        # Cria as derivadas por camada.
        # (len(layers) - 1) resulta no valor 2.
        derivadas = []
        for i in range(len(layers) - 1):
            # Cria matriz 2x2, 2x1 com valores 0.
            d = np.zeros((layers[i], layers[i + 1]))
            derivadas.append(d)
        self.derivadas = derivadas

        # Cria ativacoes por camada
        # (len(layers)) resulta no valor 3.
        ativacao = []
        for i in range(len(layers)):
            # Cria matriz 2x1, 2x1, 1x1 com valores 0
            a = np.zeros(layers[i])
            ativacao.append(a)
        self.ativacao = ativacao

        # Cria bias
        # Cria matriz 2x1 e 1x1
        # (len(layers)) resulta no valor 2.
        bias = []
        for i in range(len(layers) - 1):
            # Cria matriz 1x1, 1x1 com valores aleatorios
            a = np.random.rand(layers[i + 1])
            bias.append(a)
        self.bias = bias

    # Realiza o feedforward, recebe como argumento apenas os inputs e retorna o valor da funcao de ativacao
    def feedForward(self, inputs):
        # A ativacao da camada de entrada eh o proprio input.
        ativacao = inputs

        # Salva a ativacao para backpropagation
        self.ativacao[0] = ativacao
        # Itera atraves de toda camada neural
        for i, w in enumerate(self.pesos):
            # Existem 2 pesos: um peso eh um array 2x2 e o outro 2x1
            # Primeira Iteração: Multiplicao de matriz entre o input e o peso da matriz para a camada escondida
            # Segunda Iteração: Multiplicacao de matriz da camada escondida para o output

            net_inputs = np.dot(ativacao, w)

            # Soma com o bias
            net_inputs = np.add(net_inputs, self.bias[i])
            # Funcao de ativacao sigmoid dada em aula
            ativacao = self._sigmoid(net_inputs)
            # Guarda a ativacao para backpropogation
            self.ativacao[i + 1] = ativacao

        # retorna o array de output
        return ativacao

    # Realiza o backprogation, recebe como argumento o erro
    def backPropagation(self, error):
        # Itera atraves de toda camada
        for i in reversed(range(len(self.derivadas))):
            # Existem 2 derivadas: um de 2x2 e outro de 1x1
            # Pega funcao ativacao da ultima camada usada
            # Primeira Iteração: Ocorre entre o output e a camada escondida
            # Segunda Iteração: Ocorre entre a camada escondida e o input
            ativacao = self.ativacao[i + 1]

            # Aplica a derivada da funcao da sigmoide = gradiente
            # Delta = gradiente
            delta = error * self.derivadaSigmoide(ativacao)

            # Faz o delta se tornar 2D
            delta_re = delta.reshape(delta.shape[0], -1).T
            # Faz a tranposta (transforma linha em col) da ativacao em 2D
            current_ativacao = self.ativacao[i]
            current_ativacao = current_ativacao.reshape(current_ativacao.shape[0], -1)

            # Atualiza o bias
            self.bias[i] += delta

            # Multipliacao de matriz entre a ativacao e o delta em 2D
            # Representa o Quanto o peso tem que ser ajustado
            self.derivadas[i] = np.dot(current_ativacao, delta_re)

            # backpropogate o proximo erro
            error = np.dot(delta, self.pesos[i].T)

    # Recebe o numero de inputs, output e epocas e a taxa de aprendizado
    def treinar(self, inputs, targets, epocas, taxa_aprendizado):
        errosPorInteracao = []
        # Executando uma epoca
        for i in range(epocas):
            soma_erros = 0
            k = 0
            # Exitem 4 inputs de 2 nós: [-1,1], [1,1], [-1,-1], [1,-1]
            # Iterando para cada input
            for j, input in enumerate(inputs):
                # Esta pegando o output desejado relacionado ao tamanho da iteracao.
                # target[0], target[1], target[2]...
                if k == len(targets): k = 0
                target = targets[k]
                k += 1
                # ativacao da rede de feedforwarding
                output = self.feedForward(input)

                # Verificando o quanto errou
                error = target - output

                # Iniciando backpropagation
                self.backPropagation(error)

                # Faz alteracao dos valores atraves das derivadas
                self.gradient_desc(taxa_aprendizado)

                # Faz o erro quadratico medio
                soma_erros += self.erroQuadraticoMedio(target, output)

            # Adicionando os erros num array para gerar a saida
            errosPorInteracao.append(soma_erros / len(inputs))
            self.errosPorInteracao = errosPorInteracao

            # Epoca completa
            print("Erro: {} na epoca: {}".format(soma_erros / len(inputs), i + 1))

    # Recebe a taxa de aprendizado como argumento
    def gradient_desc(self, taxa_aprendizado=1):
        # Modifica os pesos  de acordo com o gradiente e
        # Existem 2 pesos: um peso eh um array 2x2 e o outro 2x1
        for i in range(len(self.pesos)):
            pesos = self.pesos[i]
            derivadas = self.derivadas[i]
            pesos += derivadas * taxa_aprendizado

    # Metodo da sigmoid dada em aula
    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    # Metodo da derivada da sigmoide, x precisa SER uma signoide
    def derivadaSigmoide(self, x):
        return x * (1.0 - x)

    # Metodo do Erro quadratico medio
    def erroQuadraticoMedio(self, target, output):
        return np.average((target - output) ** 2)


def ExecutarArquivoOR(input_no, hidden_no, output_no):
    input_treinamento = []
    targets = []
    with open('problemOR.csv', 'rt', encoding='utf-8-sig') as csvfile:
        # Transforma em float
        data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data:
            input_treinamento.append(row[:-1])
            targets.append([row[-1]])
    input_treinamento = np.array(input_treinamento)
    targets = np.array(targets)
    # print(len(input_treinamento[0]))
    targets = np.array(targets)
    # Instanciando Perceptron
    mlp = MultiCamadaPerceptron(input_no, hidden_no, output_no)
    # Criando a saida de pesos inicias
    with open('OR-pesos_inicias.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Inicial:\r\n {}\r\n".format(mlp.pesos[i], i))
    # Executando o método de treinar
    mlp.treinar(input_treinamento, targets, epocas, taxa_aprendizado)
    # Criando a saida parametros de inicializacao
    with open('OR-parametros_inicializacao.txt', 'w', newline='') as f:
        f.write("Numero de nos na camada inputs: {}\r\n".format(input_no))
        f.write("Numero de nos na camada escondia: {}\r\n".format(hidden_no))
        f.write("Numero de nos na camada output: {}\r\n".format(output_no))
        f.write("Numero de epocas: {}\r\n".format(epocas))
        f.write("Taxa de aprendizado: {}\r\n".format(taxa_aprendizado))
    # Criando a saida de pesos finais
    with open('OR-pesos_finais.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Final:\r\n {}\r\n".format(mlp.pesos[i], i))
    # Criando a saida de erros cometidos por interacao
    with open('OR-erros_cometidos.txt', 'w', newline='') as f:
        for i in range(len(mlp.errosPorInteracao)):
            f.write("Erro: {} na epoca {}\r\n".format(mlp.errosPorInteracao[i], i))
    # Fazendo acontecer
    output = mlp.feedForward(input_treinamento)
    # Criando a saida de output por interacao
    with open('OR-output.txt', 'w', newline='') as f:
        f.write("Resultado: {} OR {} is equal to {}\r\n".format(input_treinamento[0][0], input_treinamento[0][1],
                                                                 output[0]))
        f.write("Resultado: {} OR {} is equal to {}\r\n".format(input_treinamento[1][0], input_treinamento[1][1],
                                                                 output[1]))
        f.write("Resultado: {} OR {} is equal to {}\r\n".format(input_treinamento[2][0], input_treinamento[2][1],
                                                                 output[2]))
        f.write("Resultado: {} OR {} is equal to {}\r\n".format(input_treinamento[3][0], input_treinamento[3][1],
                                                                 output[3]))

def ExecutarArquivoXOR(input_no, hidden_no, output_no):
    input_treinamento = []
    targets = []
    with open('problemXOR.csv', 'rt', encoding='utf-8-sig') as csvfile:
        # Transforma em float
        data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data:
            input_treinamento.append(row[:-1])
            targets.append([row[-1]])
    input_treinamento = np.array(input_treinamento)
    targets = np.array(targets)
    # Instanciando Perceptron
    mlp = MultiCamadaPerceptron(input_no, hidden_no, output_no)
    # Criando a saida de pesos inicias
    with open('XOR-pesos_inicias.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Inicial:\r\n {}\r\n".format(mlp.pesos[i], i))
    # Executando o método de treinar
    mlp.treinar(input_treinamento, targets, epocas, taxa_aprendizado)
    # Criando a saida parametros de inicializacao
    with open('XOR-parametros_inicializacao.txt', 'w', newline='') as f:
        f.write("Numero de nos na camada inputs: {}\r\n".format(input_no))
        f.write("Numero de nos na camada escondia: {}\r\n".format(hidden_no))
        f.write("Numero de nos na camada output: {}\r\n".format(output_no))
        f.write("Numero de epocas: {}\r\n".format(epocas))
        f.write("Taxa de aprendizado: {}\r\n".format(taxa_aprendizado))
    # Criando a saida de pesos finais
    with open('XOR-pesos_finais.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Final:\r\n {}\r\n".format(mlp.pesos[i], i))
    # Criando a saida de erros cometidos por interacao
    with open('XOR-erros_cometidos.txt', 'w', newline='') as f:
        for i in range(len(mlp.errosPorInteracao)):
            f.write("Erro: {} na epoca {}\r\n".format(mlp.errosPorInteracao[i], i))
    # Fazendo acontecer
    output = mlp.feedForward(input_treinamento)
    # Criando a saida de output por interacao
    with open('XOR-output.txt', 'w', newline='') as f:
        f.write("Resultado: {} XOR {} is equal to {}\r\n".format(input_treinamento[0][0], input_treinamento[0][1],
                                                                 output[0]))
        f.write("Resultado: {} XOR {} is equal to {}\r\n".format(input_treinamento[1][0], input_treinamento[1][1],
                                                                 output[1]))
        f.write("Resultado: {} XOR {} is equal to {}\r\n".format(input_treinamento[2][0], input_treinamento[2][1],
                                                                 output[2]))
        f.write("Resultado: {} XOR {} is equal to {}\r\n".format(input_treinamento[3][0], input_treinamento[3][1],
                                                                 output[3]))

def ExecutarArquivoAND(input_no, hidden_no, output_no):
    global input_treinamento
    input_treinamento = []
    targets = []
    with open('problemAND.csv', 'rt', encoding='utf-8-sig') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data:
            input_treinamento.append(row[:-1])
            targets.append([row[-1]])

    input_treinamento = np.array(input_treinamento)
    targets = np.array(targets)
    # Instanciando Perceptron
    mlp = MultiCamadaPerceptron(input_no, hidden_no, output_no)

    # Criando a saida de pesos inicias
    with open('AND-pesos_inicias.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Inicial:\r\n {}\r\n".format(mlp.pesos[i], i))
    # Executando o método de treinar

    mlp.treinar(input_treinamento, targets, epocas, taxa_aprendizado)
    # Criando a saida parametros de inicializacao
    with open('AND-parametros_inicializacao.txt', 'w', newline='') as f:
        f.write("Numero de nos na camada inputs: {}\r\n".format(input_no))
        f.write("Numero de nos na camada escondia: {}\r\n".format(hidden_no))
        f.write("Numero de nos na camada output: {}\r\n".format(output_no))
        f.write("Numero de epocas: {}\r\n".format(epocas))
        f.write("Taxa de aprendizado: {}\r\n".format(taxa_aprendizado))
    # Criando a saida de pesos finais
    with open('AND-pesos_finais.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Final:\r\n {}\r\n".format(mlp.pesos[i], i))
    # Criando a saida de erros cometidos por interacao
    with open('AND-erros_cometidos.txt', 'w', newline='') as f:
        for i in range(len(mlp.errosPorInteracao)):
            f.write("Erro: {} na epoca {}\r\n".format(mlp.errosPorInteracao[i], i))
    # Fazendo acontecer
    output = mlp.feedForward(input_treinamento)
    # Criando a saida de output por interacao
    with open('AND-output.txt', 'w', newline='') as f:
        f.write("Resultado: {} AND {} is equal to {}\r\n".format(input_treinamento[0][0], input_treinamento[0][1],
                                                                 output[0]))
        f.write("Resultado: {} AND {} is equal to {}\r\n".format(input_treinamento[1][0], input_treinamento[1][1],
                                                                 output[1]))
        f.write("Resultado: {} AND {} is equal to {}\r\n".format(input_treinamento[2][0], input_treinamento[2][1],
                                                                 output[2]))
        f.write("Resultado: {} AND {} is equal to {}\r\n".format(input_treinamento[3][0], input_treinamento[3][1],
                                                                 output[3]))

def ExecutarArquivoRuivo(input_no, hidden_no, output_no):
    target = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]

    input_treinamento = []
    targets = []
    with open('caracteres-ruido.csv', 'rt', encoding='utf-8-sig') as csvfile:
        # Transforma em float
        data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data:
            input_treinamento.append(row[:-1])
        for i in range(output_no):
            targets.append(target[i])

    input_treinamento = np.array(input_treinamento)
    targets = np.array(targets)
    # Instanciando Perceptron, criando as matrizes de pesos
    mlp = MultiCamadaPerceptron(len(input_treinamento[0]), hidden_no, output_no)

    # Criando a saida de pesos inicias
    with open('ruido-pesos_inicias.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Inicial:\r\n {}\r\n".format(mlp.pesos[i], i))

    # Executando o método de treinar
    mlp.treinar(input_treinamento, targets, epocas, taxa_aprendizado)

    # Criando a saida parametros de inicializacao
    with open('ruido-parametros_inicializacao.txt', 'w', newline='') as f:
        f.write("Numero de nos na camada inputs: {}\r\n".format(input_no))
        f.write("Numero de nos na camada escondia: {}\r\n".format(hidden_no))
        f.write("Numero de nos na camada output: {}\r\n".format(output_no))
        f.write("Numero de epocas: {}\r\n".format(epocas))
        f.write("Taxa de aprendizado: {}\r\n".format(taxa_aprendizado))

    # Criando a saida de pesos finais
    with open('ruido-pesos_finais.txt', 'w', newline='') as f:
        for i in range(len(mlp.pesos)):
            f.write("Peso Final:\r\n {}\r\n".format(mlp.pesos[i], i))

    # Criando a saida de erros cometidos por interacao
    with open('ruido-erros_cometidos.txt', 'w', newline='') as f:
        for i in range(len(mlp.errosPorInteracao)):
            f.write("Erro: {} na epoca {}\r\n".format(mlp.errosPorInteracao[i], i))

    # Fazendo acontecer
    output = mlp.feedForward(input_treinamento)

    # Criando a saida de output por interacao
    with open('ruido-output.txt', 'w', newline='') as f:
        for i in range(len(output)):
            f.write("Resultado: {} é igual {}\r\n".format(i, output[i]))

if __name__ == "__main__":
    epocas = 50000
    taxa_aprendizado = 0.1

    ExecutarArquivoAND(2, [2], 1)
    ExecutarArquivoOR(2, [2], 1)
    ExecutarArquivoXOR(2, [2], 1)

    # Aumentando o numero de inputs pois o arquivo de ruido tem 63 inputs.
    ExecutarArquivoRuivo(63, [2], 7)

