# Neural Net Framework C++

---

A simple framework for a feedforward, fully connected Neural net, with various different activation functions.


## How to use:

### Construct:

First of, construct your neural net with `NeuralNet net({n, 32, 32, m}, leakyReLU, Tanh);` This creates a neural net with `n` input neurons, 2 hidden layer with each 32 neurons and the `m` output neurons. Also the activation of the hidden layers are in this case the `leakyReLU` function, and the output layer has the `Tanh` function.

### How to train (example):

A neural net maps any amount of input values to any amount of output values. For this architecture, you always need to know the values to train the neural net with. A simple example would be teaching the neural net the `f(x) = sin(x)` function, for which we will need 1 input and 1 output neuron. So we construct:

`NeuralNet net({1, 20, 1}, leakySigmoid, Tanh);`

To choose input and output, we will first sample values from the function at random.

`float intputValue = randFloat(0, 2.0f * PI)`

After that, feed the input values into the network, which will do a full forwards propagation:

`net.forward({inputValue});` (The inputValue is wrapped in "{}", as the input is a `std::vector`, bacuse input and output neurons can be more than 1)


After that, we calculate the value, the neural net should map to the input value.

`float functionValue = sin(intputValue);`

Not the last step is to use the backpropagation function, to change the weights and biases.

`net.backpropagation({functionValue});`

This is a full step to train the AI to map the input values to the output values of a sin function. This procedure needs to be repeated multiple times (from 100 up to 100000 times and higher).

### How to read/use the network:

After you finsihed training the network, you can set the input values and get the prediction of the network for the function you trained it on. Before you do this, you first need to set the input value.

`net.forward({inputValue});`

After that just get the output value.

`float result = net.getOutputValues().front();`

As the output is a `std::vector` and you only need the first value (as there is only a single output) of the output.


### Example runs:

A test run showed, that after running the train loop 10000 times, the average error to the sin function on 1000 random values was just 0.011, so about 1% deviation from the sin function! I am also sure that changing some parameters and longer learning can yield even better results.

I also made it recognize hand-drawn digits from the MNIST dataset. With some tweaking, I got the network to a 92% accuracy. The neural network had a size of 784 input neurons, 2 hidden layers with each 32 neurons and an output layer of 10 neurons. So it correcly chose the right number out of 10 numbers, 92% of the time!


### Additionell note:

The weights are randomly initialized. The random function by default will always return the same values when executed multiple times. To get fresh random values each execution, you can put `srand(time(0))` at the top of your code.



## How to optimise training:

There are a few parameters you can tweak to change the way the network learns. Some changes can make the network worse and better, so experiment with different parameters and create a way to benchmark the training.

The most obvios: increase the `learningRate` (default is 0.01). This makes the network learn faster, at the cost of making it more instable, which can cause explosions in the weights and make it unable to learn. Something you could try is gradually decreaing the learning rate while learning, which can but doesnt guarentee better results. I do not recommend going above 0.01 or 0.02.

Another thing you can do is switch the activation function for other activation functions, in the hidden and output layer.

Making the network larger (adding more hidden layers with more neurons) has many interesting effects. First of all, it increases the complexity of the task the neural net can do. It does not guarentee your network to be generally more accurate (google overfitting/underfitting). It also increases the learning time, by being computationaly more expensive.


---

## How to use it in my code:

Using it as external code is really simple, first clone the repo into any folder you like. Then, include it with `#include "/path/to/folder/NeuralNet.h"` and you are done!