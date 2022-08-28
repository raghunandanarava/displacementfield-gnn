### Resist Modelling Layer

The Resist Message Passing Layer that is defined extends the base Message Passing Layer from Spektral. It can be utilised in multiple ways. For the time being, with the data available, this layer has been utilised for modelling to predict the displacement values in x and y. As a matter of fact, this can be further extended to any dimensions depending on the problem. There are two Multi-Layer Perceptrons in this layer, which are the Message Computing MLP and the Updating MLP. The former is the one which is trained in such a way that all the messages that a node in a graph received are computed by it (This is done for the entire graph) and the latter updates the node embeddings based on the aggregated messages that a node receives.

The same layer can be used in conjunction with predicting displacement fields, to predict the forces acting on each of the node in the graph. In simple words, a separate instance of this layer predicts the forces in x and y (these values can be saved from FEM simulations to use during the training phase) and another instance takes the inputs of spatial orientation of the node along with the corresponding force values from the first instance for more accurate prediction of displacement fields.

## Aggregation

There are different types of aggregation mechanisms in a message passing neural network. Some of them are sum, product, maximum, minimum, and mean (sum, prod, max, min, and mean are the keywords to be utilised in the model). For the time being, these mechanisms are applied individually. In further developments, a more sophisticated aggregation mechanism which is heterogeneous in nature can also be utilised. Doing this, it will further help the models to learn swiftly with faster convergence.

## Modelling Elasticity in a MPNN

The current model is also constructed in such a way that it is elasticity regularised. Linear Elasticity has been used in this model. This is done by calculating the strain values from the predicted displacement values and in turn calculate the stress values from it. These predicted values are then compared with the actual values to compute the loss which is utilised for backpropagation during the training phase. The regularisation is not included during the test/evaluation time.

## How to use
1. Please make sure to have Tensorflow and Spektral packages installed before using
2. The Layer defined in ResistModelling.py is the base for utilising the message passing neural network.
3. With this, the train step and test step can be overridden (based on the requirements of the problem) as it can be seen in the file train.py.