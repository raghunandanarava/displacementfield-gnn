import tensorflow as tf
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from spektral.data.loaders import DisjointLoader
from ResistMessagePassing import ResistMessagePassing
from DelaunayGraphUtil import DelaunayGraph

"""GPU Settings"""
GPU_INDEX = 0
tf.keras.backend.set_floatx('float32')
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
physical_devices = tf.config.list_physical_devices('GPU')
print("Number of GPUs:", len(physical_devices))

"""Hyperparameters"""
learning_rate = 1e-4
training_epochs = 40
batch_size=16

"""Train & Validation Data Split"""
dataset = DelaunayGraph("/user_data/oe330/arava/Delaunay/",transforms=None)

np.random.shuffle(dataset)
va_split, test_split = int(0.6 * len(dataset)), int(0.8 * len(dataset))
data_tr = dataset[:va_split]
data_va = dataset[va_split:test_split]
data_te = dataset[test_split:]

loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=batch_size)
loader_va = DisjointLoader(data_va, node_level=True, batch_size=batch_size)
loader_te = DisjointLoader(data_te, node_level=True, batch_size=batch_size)

"""Model, Optimiser, and the loss function are defined here"""
class ResistNet(tf.keras.models.Model):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mp1 = ResistMessagePassing(channels=channels, aggregate="max", mlp_hidden=[128, 64, 32, 16, 8, 4], mlp_final_activation="linear")

    
    def call(self, inputs):
        output = self.mp1(inputs)
        return output

graph_model = ResistNet(channels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

graph_model.compile(
    optimizer=optimizer, 
    loss=loss_fn, 
    metrics=[tf.keras.metrics.MeanSquaredError()])
# graph_model.fit(
#     loader_tr.load(),
#     steps_per_epoch=loader_tr.steps_per_epoch,
#     epochs=training_epochs,
#     validation_data=loader_va.load(),
#     validation_steps=loader_va.steps_per_epoch,
#     callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

# loss,  loss_metric = graph_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
# print("Done. Test loss: {}. Test acc: {}".format(loss, loss_metric))

"""Training and Validation steps are defined here"""
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = graph_model(inputs, training=True)
        loss = loss_fn(targets, predictions) + sum(graph_model.losses)
    gradients = tape.gradient(loss, graph_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, graph_model.trainable_variables))

    #train_loss_metric.update_state(targets, predictions)
    return loss


@tf.function(input_signature=loader_va.tf_signature(), experimental_relax_shapes=True)
def validation_step(val_inputs, val_tar):
    val_pred = graph_model(val_inputs, training=False)
    vl = loss_fn(val_tar, val_pred)

    #validation_loss_metric.update_state(val_tar, val_pred)
    return vl

"""Training and Validation"""
def train(train_loader, val_loader):
    results = []
    step = t_loss = epoch = 0
    # wait = 0
    # best = 0
    # patience = 5
    #callbacks.on_epoch_begin(1, logs=logs)
    for input, target in train_loader:
        step += 1
        #graph_model.reset_states()
        #callbacks.on_batch_begin(step, logs=logs)
       # callbacks.on_train_batch_begin(step, logs=logs)
        #logs = graph_model.train_on_batch(x=input, y=target, return_dict=True)
        t_loss += train_step(input, target)
        if step == train_loader.steps_per_epoch:
            epoch += 1
            
            val_loss = validation(val_loader)
            template = 'Epoch {}, Train Loss: {}, Validation Loss: {}'
            print(template.format(epoch, t_loss / train_loader.steps_per_epoch, val_loss / val_loader.steps_per_epoch))
            results.append([epoch, t_loss / train_loader.steps_per_epoch, val_loss / val_loader.steps_per_epoch])
            step = 0
            t_loss = 0
    
    return results

def validation(validation_loader):
    v_loss = 0
    v_step = 0
    for v_input, v_target in validation_loader:
        v_step += 1
        v_loss += validation_step(v_input, v_target)
        if v_step == validation_loader.steps_per_epoch:
            break
    # with validation_summary_writer.as_default():
        # tf.summary.scalar('loss', val_l_m.result(), step=epoch)
    return v_loss

results = np.asarray(train(loader_tr, loader_va))
np.savetxt("Results_1_Fresh.csv", results)

"""Test"""
test_loss = 0
for test_input, test_target in loader_te:
    test_loss += validation_step(test_input, test_target)

print("Test Loss after {} epochs: {}".format(training_epochs, test_loss / loader_te.steps_per_epoch))

tf.keras.models.save_model(graph_model, "Models/Run_First_Checkpoint.tf", include_optimizer=True)
