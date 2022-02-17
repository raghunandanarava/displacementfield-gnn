import tensorflow as tf
import os
import numpy as np
import datetime

from spektral.data.loaders import DisjointLoader

from DataConversionUtil import GraphData
from ResistMessagePassing import ResistMessagePassing

"""GPU Settings"""
GPU_INDEX = 1
tf.keras.backend.set_floatx('float16')
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
physical_devices = tf.config.list_physical_devices('GPU')
print("Number of GPUs:", len(physical_devices))

"""Hyperparameters"""
learning_rate = 1e-4
training_epochs = 50

"""Train & Validation Data Split"""
dataset = GraphData(transforms=None)

np.random.shuffle(dataset)
va_split, test_split = int(0.6 * len(dataset)), int(0.8 * len(dataset))
data_tr = dataset[:va_split]
data_va = dataset[va_split:test_split]
data_te = dataset[test_split:]

loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=16, epochs=training_epochs, shuffle=True)
loader_va = DisjointLoader(data_va, node_level=True, batch_size=16, epochs=training_epochs, shuffle=True)
loader_te = DisjointLoader(data_te, node_level=True, batch_size=16, epochs=1, shuffle=False)

"""Model, Optimiser, and the loss function are defined here"""
class ResistNet(tf.keras.models.Model):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mp1 = ResistMessagePassing(channels=channels, aggregate="prod", mlp_hidden=[128, 64, 32, 16, 8, 4])
    
    def call(self, inputs):
        output = self.mp1(inputs)
        return output

graph_model = ResistNet(channels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# """Metrics"""
# train_loss_metric = tf.keras.metrics.MeanSquaredError('train_loss', dtype=tf.float16)
# validation_loss_metric = tf.keras.metrics.MeanSquaredError('validation_loss', dtype=tf.float16)

"""Training and Validation steps are defined here"""
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = graph_model(inputs, training=True)
        loss = loss_fn(targets, predictions) + sum(graph_model.losses)
    gradients = tape.gradient(loss, graph_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, graph_model.trainable_variables))

    # tlm(targets, predictions)
    return loss


@tf.function(input_signature=loader_va.tf_signature(), experimental_relax_shapes=True)
def validation_step(val_inputs, val_tar):
    val_pred = graph_model(val_inputs, training=False)
    vl = loss_fn(val_tar, val_pred)

    # val_lm(val_tar, val_pred)
    return vl

# '"""Summary Writers"""
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# validation_log_dir = 'logs/gradient_tape/' + current_time + '/validation'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)'

"""Training and Validation"""
def train(train_loader, val_loader):
    results = []
    step = t_loss = epoch = 0
    for input, target in train_loader:
        step += 1
        t_loss += train_step(input, target)
        if step == train_loader.steps_per_epoch:
            epoch += 1
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', tlm.result(), step=epoch)
            
            val_loss = validation(val_loader, epoch)
            template = 'Epoch {}, Train Loss: {}, Validation Loss: {}'
            print(template.format(epoch, t_loss / train_loader.steps_per_epoch, val_loss / val_loader.steps_per_epoch))
            results.append([epoch, t_loss / train_loader.steps_per_epoch, val_loss / val_loader.steps_per_epoch])
            step = 0
            t_loss = 0
    
    return results, epoch


def validation(validation_loader, epoch):
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

results, epoch = np.asarray(train(loader_tr, loader_va))
np.savetxt("Results_1_Normalisation_Based_On_Inputs.csv", results)

"""Test"""
test_loss = 0
for test_input, test_target in loader_te:
    test_loss += validation_step(test_input, test_target)

print("Test Loss after {} epochs: {}".format(epoch, test_loss / loader_te.steps_per_epoch))

tf.keras.models.save_model(graph_model, "Models/Run_1_Normalisation_Based_On_Input.tf")
