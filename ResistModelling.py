import inspect
import tensorflow as tf

from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential, Model

from spektral.layers.convolutional.message_passing import MessagePassing


"""
A message passing layer modified to suit the modelling of a photoresist cross-section

Inputs
message_channels: Number of output channels in the message computing MLP
update_channels: Number of output channels in the message updating MLP
aggregate: The aggregation meechanism to use, default is set to "sum"
mlp_messages_hidden: A list consisting of the number of units in the hidden layers of the message computing MLP
mlp_update_hidden: A list consisting of the number of units in the hidden layers of the message updating MLP
mlp_activation: The intermediate activation function used in the MLPs
mlp_final_activation: The activation function in the final layers of the MLPs
"""
class ResistMessagePassing(MessagePassing):
    def __init__(
        self,
        message_channels,
        update_channels,
        aggregate="sum",
        mlp_messages_hidden=None,
        mlp_update_hidden=None,
        mlp_activation="relu",
        mlp_final_activation="linear",
        mlp_batchnorm=False,
        kernel_initializer="glorot_uniform",
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs):
        super().__init__(
            aggregate=aggregate,
            mlp_messages_hidden=mlp_messages_hidden,
            mlp_update_hidden=mlp_update_hidden,
            mlp_batchnorm=mlp_batchnorm,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, 
            **kwargs)

        self.message_channels = message_channels
        self.update_channels = update_channels
        self.mlp_activation = activations.get(mlp_activation)
        self.mlp_final_activation = activations.get(mlp_final_activation)

        self.mlp_messages =  MLP(self.message_channels, mlp_messages_hidden, self.mlp_activation, self.mlp_final_activation)
        self.mlp_update = MLP(self.update_channels, mlp_update_hidden, self.mlp_activation, self.mlp_final_activation)

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters

    """
    Inputs
    x: Input features
    a: Adjacency Matrix defining the graph
    e: Edge features
    i: Iterator
    """
    def call(self, inputs, **kwargs):
        x, a, e, i = inputs
        return self.propagate(x, a, e)
    
    def propagate(self, x, a, e, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

         # Message
        msg_kwargs = self.get_kwargs(x, a, e, signature=self.msg_signature, kwargs=kwargs)
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a, e, signature=self.agg_signature, kwargs=kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a, e, signature=self.upd_signature, kwargs=kwargs)
        output = self.update(embeddings, x, **upd_kwargs)

        return output
    
    def message(self, x, **kwargs):
        sender_features = self.get_j(x)
        receiver_features = self.get_i(x)
        distances = tf.norm(sender_features[:, 0:2] - receiver_features[:, 0:2], ord="euclidean", axis=-1, keepdims=True)
        messages = self.mlp_messages(tf.concat([sender_features, distances, receiver_features], axis=1))
        return messages
    
    def update(self, embeddings, x, **kwargs):
        embeddings = self.mlp_update(tf.concat([embeddings, x], axis=1))
        return embeddings


class MLP(Model):
    def __init__(self, outputs, mlp_hidden, mlp_activation, mlp_final_activation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outputs = outputs
        self.mlp_update = Sequential()

        for c in mlp_hidden:
            self.mlp_update.add(Dense(c, activation=mlp_activation))
            # self.mlp_update.add(Dropout(0.3))
            self.mlp_update.add(BatchNormalization())
        self.mlp_update.add(Dense(self.outputs, activation=mlp_final_activation))
    
    def call(self, input):
        return self.mlp_update(input)