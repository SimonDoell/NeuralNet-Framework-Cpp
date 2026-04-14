#pragma once
#include <iostream>
#include <random>
#include <cmath>
#include <vector> 

// --- Activation Functions ---
struct Activation {
    float (*forward)(float);
    float (*derivative)(float);
};

Activation ReLU {
    [](float x){ return x < 0 ? 0.f : x; },
    [](float x){ return x < 0 ? 0.f : 1.f; }
};

Activation Sigmoid {
    [](float x){ return 1.0f / (1.0f + std::exp(-x)); },
    [](float x){ float s = 1.0f / (1.0f + std::exp(-x)); return s * (1.0f - s);}
};

constexpr float leakyReLUAlpha = 0.15f; // You can change this
Activation leakyReLU {
    [](float x){ return std::max(x*leakyReLUAlpha, x); },
    [](float x){ return x <= 0 ? leakyReLUAlpha : 1.0f; }
};

Activation Swish {
    [](float x){ return x / (1.0f + std::exp(-x)); },
    [](float x){ float s = 1.0f / (1.0f + std::exp(-x)); return s + x * s * (1.0f - s); }
};

constexpr float leakySigmoidAlpha = 0.15f; // You can change this
Activation leakySigmoid {
    [](float x){ return 1.0f/(1.0f + std::exp(-x)) + leakySigmoidAlpha * x; },
    [](float x){ float s = 1.0f / (1.0f + std::exp(-x)); return s * (1.0f - s) + leakySigmoidAlpha; }
};

Activation Tanh {
    [](float x){ float e1 = exp(x); float e2 = exp(-x); return (e1 - e2) / (e1 + e2); },
    [](float x){ float e1 = exp(x); float e2 = exp(-x); float t = (e1 - e2) / (e1 + e2); return 1.0f - (t * t); }
};

Activation Linear {
    [](float x){ return x; },
    [](float x){ return 1.0f; }
};


// --- Helper Functions ---
float randFloat(float min, float max) {return ((float)rand() / (float)RAND_MAX * (max - min)) + min;}


// --- Neural Net ---
struct Neuron {
    public:
        float activation;
        float bias;
        std::vector<float> weights;

        float preActivation;
        float delta;

        Neuron(int _nextLayerNeuronAmount, int _currentLayerNeuronAmount)
        : bias(0.0f) {
            float weightRandRange = sqrt(1.0f / _currentLayerNeuronAmount);
            for (int i = 0; i < _nextLayerNeuronAmount; ++i) {
                weights.emplace_back(randFloat(-weightRandRange, weightRandRange));
            }
        }
};


struct Layer {
    public:
        int amountNeurons;
        std::vector<Neuron> neurons;
        std::vector<Neuron*> nextLayerNeurons;
        Activation activationFunc;

        Layer() {}
        Layer(int _amountNeuron, std::vector<Neuron*> _nextLayerNeurons, Activation _activationFunc = ReLU)
        : nextLayerNeurons(_nextLayerNeurons), amountNeurons(_amountNeuron), activationFunc(_activationFunc) {
            for (int i = 0; i < _amountNeuron; ++i) neurons.emplace_back(Neuron(_nextLayerNeurons.size(), _amountNeuron));
        }

        std::vector<Neuron*> getNeuronsAsPtr() {
            std::vector<Neuron*> pointers;
            for (int i = 0; i < neurons.size(); ++i) pointers.emplace_back(&neurons[i]);
            return pointers;
        }

        void calculateNextLayer(Activation nextLayerActivationFunc) {
            for (int x = 0; x < nextLayerNeurons.size(); ++x) {
                nextLayerNeurons[x]->activation = 0;

                // Weights
                for (int i = 0; i < neurons.size(); ++i) {
                    nextLayerNeurons[x]->activation += neurons[i].activation * neurons[i].weights[x];
                }
            }

            for (int i = 0; i < nextLayerNeurons.size(); ++i) {
                // Bias
                nextLayerNeurons[i]->activation   += nextLayerNeurons[i]->bias;
                // Activation Function
                nextLayerNeurons[i]->preActivation = nextLayerNeurons[i]->activation;
                nextLayerNeurons[i]->activation    = nextLayerActivationFunc.forward(nextLayerNeurons[i]->activation);
            }
        }
};



struct NeuralNet {
    public:
        float learningRate = 0.01f;

        NeuralNet(std::vector<int> _layersNeuronAmount, Activation _activationFunctionHiddenLayers = ReLU, Activation _activationFunctionOutputLayer = Sigmoid)
        : layersNeuronAmount(_layersNeuronAmount)
        {
            layers.resize(_layersNeuronAmount.size());
            layers.back() = (Layer(_layersNeuronAmount.back(), {}, _activationFunctionOutputLayer));

            for (int i = _layersNeuronAmount.size()-2; i >= 0; --i) 
                layers[i] = Layer(_layersNeuronAmount[i], layers[i+1].getNeuronsAsPtr(), _activationFunctionHiddenLayers);
        }

        std::vector<float> getOutputValues() {
            std::vector<float> values;

            for (Neuron& n : layers.back().neurons) 
                values.emplace_back(n.activation);
            
            return values;
        }

        void forward(const std::vector<float>& inputValues) {
            if (inputValues.size() != layers.front().neurons.size()) {std::cout << "Input values are the wrong size!\n"; return;}
            for (int i = 0; i < inputValues.size(); ++i) layers.front().neurons[i].activation = inputValues[i];
            for (int i = 0; i < layers.size()-1; ++i) {layers[i].calculateNextLayer(layers[i+1].activationFunc);}
        }

        void backpropagation(const std::vector<float>& desiredValuesForOutputLayer) {
            calculateDelta(desiredValuesForOutputLayer);
            calculateBackpropChanges();
        }

    private:
        std::vector<Layer> layers;
        std::vector<int> layersNeuronAmount;

        // Calculate delta
        void calculateDelta(const std::vector<float>& desiredValuesForOutputLayer) {
            if (layers.back().neurons.size() != desiredValuesForOutputLayer.size()) {std::cout << "List size mismatch in backpropagation!\n"; return;}

            for (int i = layers.size()-1; i >= 0; --i) {
                for (int x = 0; x < layers[i].neurons.size(); ++x) {
                    Neuron& currNeuron = layers[i].neurons[x];

                    if (i == layers.size()-1) {
                        currNeuron.delta = (currNeuron.activation - desiredValuesForOutputLayer[x]) * layers[i].activationFunc.derivative(currNeuron.preActivation);
                        // Delta = error * funcDerivative
                    } else {
                        float sum = 0;
                        for (int w = 0; w < layers[i].nextLayerNeurons.size(); ++w) 
                            sum += currNeuron.weights[w] * layers[i].nextLayerNeurons[w]->delta;
                        
                        currNeuron.delta = sum * layers[i].activationFunc.derivative(currNeuron.preActivation);
                        // Current neuron delta scaled by the amount its contributing to the value of the next neuron (so derivative(preActivation))
                    }
                }
            }
        }

        // Backpropagation
        void calculateBackpropChanges() {
            for (int l = 0; l < layers.size(); ++l) {
                for (int n = 0; n < layers[l].neurons.size(); ++n) {
                    for (int w = 0; w < layers[l].neurons[n].weights.size(); ++w) {
                        layers[l].neurons[n].weights[w] -= learningRate * layers[l].neurons[n].activation * layers[l].nextLayerNeurons[w]->delta;
                    }
                }
            }

            for (int l = 0; l < layers.size(); ++l) {
                for (int n = 0; n < layers[l].neurons.size(); ++n) {
                    layers[l].neurons[n].bias -= learningRate * layers[l].neurons[n].delta;
                }
            }
        }
};