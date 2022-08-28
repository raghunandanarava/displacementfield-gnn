import tensorflow as tf
import numpy as np
import os

from spektral.data.loaders import DisjointLoader
from ResistModelling import ResistMessagePassing
from DelaunayGraphElasticityUtil import DelaunayGraph

"""GPU Settings"""
GPU_INDEX = 0
tf.keras.backend.set_floatx('float32')
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
physical_devices = tf.config.list_physical_devices('GPU')
print("Number of GPUs:", len(physical_devices))

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

"""Hyperparameters"""
learning_rate = 1e-4
training_epochs = 300
batch_size=32

data_tr = DelaunayGraph("/user_data/oe330/arava/DelaunayElasticMinMax/train_set", transforms=None)
data_va = DelaunayGraph("/user_data/oe330/arava/DelaunayElasticMinMax/validation_set", transforms=None)
data_te = DelaunayGraph("/user_data/oe330/arava/DelaunayElasticMinMax/test_set", transforms=None)

loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=batch_size)
loader_va = DisjointLoader(data_va, node_level=True, batch_size=batch_size)
loader_te = DisjointLoader(data_te, node_level=True, batch_size=batch_size)

"""Model, Optimiser, and the loss function are defined here"""
class ResistNet(tf.keras.models.Model):
    def __init__(self, message_channels, update_channels, aggregate, mlp_messages_hidden, mlp_update_hidden, mlp_activation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x_max = tf.cast(86.005, dtype=tf.float32)
        self.x_min = tf.cast(51.995, dtype=tf.float32)
        self.y_max = tf.cast(55.205, dtype=tf.float32)
        self.y_min = tf.cast(19.995, dtype=tf.float32)
        self.ux_max = tf.cast(31.34, dtype=tf.float32)
        self.ux_min = tf.cast(-0.9073, dtype=tf.float32)
        self.uy_max = tf.cast(17.47, dtype=tf.float32)
        self.uy_min = tf.cast(-14.72, dtype=tf.float32)
        self.strainx_max = tf.cast(0.59576923, dtype=tf.float32)
        self.strainx_min =  tf.cast(-0.015607127, dtype=tf.float32)
        self.strainy_max = tf.cast(0.31648551, dtype=tf.float32)
        self.strainy_min = tf.cast(-0.317241379, dtype=tf.float32)
        self.stressx_max = tf.cast(0.51731128, dtype=tf.float32)
        self.stressx_min = tf.cast(-0.0700459388, dtype=tf.float32)
        self.stressy_max = tf.cast(0.45680483, dtype=tf.float32)
        self.stressy_min = tf.cast(-0.144518289, dtype=tf.float32)
        self.del_ux = self.ux_max - self.ux_min
        self.del_uy = self.uy_max - self.uy_min
        self.E = tf.cast(0.3, dtype=tf.float32)
        self.nu = tf.cast(0.4, dtype=tf.float32)
        self.temp = tf.convert_to_tensor(self.E / ((1 + self.nu) * (1 - (2 * self.nu))))


        self.mp_disp = ResistMessagePassing(message_channels=message_channels, update_channels=update_channels, aggregate=aggregate, mlp_messages_hidden=mlp_messages_hidden, mlp_update_hidden=mlp_update_hidden, mlp_activation=mlp_activation, mlp_final_activation="linear")
    
    def call(self, inputs):
        return self.mp_disp(inputs)
    
    def train_step(self, data):
        inputs, targets = data
        targets = tf.cast(targets, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            
            loss = self.regularise(y_pred, targets)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(targets[:, 0:2], y_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        inputs, targets = data
        targets = tf.cast(targets, dtype=tf.float32)

        y_pred = self(inputs, training=False)

        loss = self.regularise(y_pred, targets)

        self.compiled_metrics.update_state(targets[:, 0:2], y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def regularise(self, y_pred, targets):
        ux_pred = tf.expand_dims(tf.add(tf.multiply(y_pred[:, 0], self.del_ux), self.ux_min), axis=1)
        uy_pred = tf.expand_dims(tf.add(tf.multiply(y_pred[:, 1], self.del_uy), self.uy_min), axis=1)
        u_original = tf.concat([ux_pred, uy_pred], axis=1)

        strain = tf.divide(u_original, targets[:, 6:])

        stress_x = tf.expand_dims(self.temp * tf.add(tf.multiply(1 - self.nu, strain[:, 0]), tf.multiply(self.nu, strain[:, 1])), axis=1)
        stress_y = tf.expand_dims(self.temp * tf.add(tf.multiply(1 - self.nu, strain[:, 1]), tf.multiply(self.nu, strain[:, 0])), axis=1)
        stress = tf.concat([stress_x, stress_y], axis=1)

        strain_x_no, strain_y_no = tf.expand_dims(tf.divide(
            tf.subtract(strain[:, 0], self.strainx_min), tf.subtract(self.strainx_max, self.strainx_min)
        ), axis=1), tf.expand_dims(tf.divide(
            tf.subtract(strain[:, 1], self.strainy_min), tf.subtract(self.strainy_max, self.strainy_min)
        ), axis=1)
        
        strain_no = tf.concat([strain_x_no, strain_y_no], axis=1)

        stress_x_no, stress_y_no = tf.expand_dims(tf.divide(
            tf.subtract(stress[:, 0], self.stressx_min), tf.subtract(self.stressx_max, self.stressx_min)
        ), axis=1), tf.expand_dims(tf.divide(
            tf.subtract(stress[:, 1], self.stressy_min), tf.subtract(self.stressy_max, self.stressy_min)
        ), axis=1)

        stress_no = tf.concat([stress_x_no, stress_y_no], axis=1)
        
        #mse_u = self.compiled_loss(targets[:, 0:2], y_pred)
        mse_u = self.compiled_loss(targets[:, 0:2], y_pred)
        mse_strain = self.compiled_loss(targets[:, 2:4], strain_no)
        mse_stress = self.compiled_loss(targets[:, 4:6], stress_no)
        loss = mse_u + mse_strain + mse_stress
        return loss


"""
Custom Model Checkpoint for saving intermediate models
The condition, as seen below, can be customised
"""
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['mean_squared_error']}, valid_acc: {logs['val_mean_squared_error']}")
        if logs['val_mean_squared_error'] < logs['mean_squared_error']: # your custom condition
            self.model.save('Tests/0/checkpoint.tf', overwrite=True, include_optimizer=True)

"""
Example model
"""
graph_model = ResistNet(
    message_channels=256,
    update_channels=2,
    aggregate="sum",
    mlp_messages_hidden=[4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256],
    mlp_update_hidden=[256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4],
    mlp_activation="relu"
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.Huber()

cbk = CustomModelCheckpoint()

graph_model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

history = graph_model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    epochs=training_epochs,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    callbacks=[cbk]
)

loss, loss_metric = graph_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test Loss: {}, Test Acc: {}".format(loss, loss_metric))

tf.keras.models.save_model(graph_model, "Tests/0/final_model.tf", include_optimizer=True)
np.save("Tests/0/model.npy", history.history)
