########################################################################################################################
# BASED ON CODE FROM Oli Blum, available at: https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
########################################################################################################################
import math
from matplotlib import pyplot
from math import cos, sin, atan
import copy

FIGURE_NUM = 1

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.lines = []
        self.weights = []
        self.previous_weights = None
        self.weights == None

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def updateWeights(self,newWeights):
        if(self.weights == None or self.previous_weights == None):
            self.weights = copy.deepcopy(newWeights)
            self.previous_weights = copy.deepcopy(newWeights)
        else:
            self.previous_weights = copy.deepcopy(self.weights)
            self.weights = copy.deepcopy(newWeights)
        # updating delta
        delta = copy.deepcopy(newWeights)
        max = None
        min = None
        for neuron_index, neuron in enumerate(self.neurons):
            if self.previous_layer:
                for prev_neuron_index, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    delta[prev_neuron_index][neuron_index] = math.fabs(self.weights[prev_neuron_index][neuron_index] - self.previous_weights[prev_neuron_index][neuron_index])
                    if(max == None or max < delta[prev_neuron_index][neuron_index]):
                        max = delta[prev_neuron_index][neuron_index]
                    if(min == None or min > delta[prev_neuron_index][neuron_index]):
                        min = delta[prev_neuron_index][neuron_index]
        max -= min
        self.colorMatrix = copy.deepcopy(newWeights)
        for neuron_index, neuron in enumerate(self.neurons):
            if self.previous_layer:
                for prev_neuron_index, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    delta[prev_neuron_index][neuron_index] -= min
                    delta[prev_neuron_index][neuron_index] /= 1 if max == 0 else max
                    if(delta[prev_neuron_index][neuron_index] > 1 ):
                        delta[prev_neuron_index][neuron_index] = 1
                    elif(delta[prev_neuron_index][neuron_index] < 0):
                        delta[prev_neuron_index][neuron_index] = 0
                    self.colorMatrix[prev_neuron_index][neuron_index] = (delta[prev_neuron_index][neuron_index],0,1-delta[prev_neuron_index][neuron_index])

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2,prev_neuron_index,neuron_index, weight = 1,):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment),linewidth=weight,color=self.colorMatrix[prev_neuron_index][neuron_index])
        pyplot.gca().add_line(line)
        self.lines.append(line)

    def draw(self, layerType=0):
        self.updateFigure()
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

    def updateFigure(self):
        for line in self.lines:
            line.remove()
        self.lines = []
        if self.weights:
            if len(self.weights) == 0:
                if self.previous_layer:
                    self.weights = [[0.7 for j in self.previous_layer.neurons] for i in range(0,len(self.neurons))]
            for neuron_index,neuron in enumerate(self.neurons):
                neuron.draw( self.neuron_radius )
                if self.previous_layer:
                    for prev_neuron_index,previous_layer_neuron in enumerate(self.previous_layer.neurons):
                        self.__line_between_two_neurons(neuron, previous_layer_neuron,prev_neuron_index,neuron_index,weight=self.weights[prev_neuron_index][neuron_index])
        else:
            for neuron_index,neuron in enumerate(self.neurons):
                neuron.draw( self.neuron_radius )


class NeuralNetwork():
    def __init__(self, hiddenUnits,inputs,outputs):
        self.figure = pyplot.figure(FIGURE_NUM)
        self.axes = self.figure.add_subplot(111)
        #determining widest layer
        widest_layer = 0
        for layer in hiddenUnits:
            if(widest_layer<layer):
                widest_layer = layer
        self.number_of_neurons_in_widest_layer = widest_layer
        self.layers = []
        self.layertype = 0
        # building network
        self.add_layer(inputs)
        for layer in hiddenUnits:
            self.add_layer(layer)
        self.add_layer(outputs)
        self.isFirstDraw = True

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def updateLayerWeights(self,layerNum,weights):
        layer = self.layers[layerNum]
        layer.updateWeights(weights)

    def draw(self):
        pyplot.figure(FIGURE_NUM)
        for i in range( len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        self.axes.axis('scaled')
        self.axes.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        if self.isFirstDraw:
            self.isFirstDraw = False
            pyplot.show(block=False)
        else:
            pyplot.pause(0.1)
