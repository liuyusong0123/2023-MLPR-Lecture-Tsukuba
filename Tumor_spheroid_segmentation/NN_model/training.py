from utils.early_stop_tool import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from model import model_
import numpy as np
early_stop = EarlyStopping()
model = model_()

OCT_log = True


def lr_decay(ini, epoch):
    return (ini / (epoch + 1)) + 0.0001



train_data= np.load(r"E:\Segmentation\Lecture\Dataset\train_data_500.npy")

val_data = np.load(r"E:\Segmentation\Lecture\Dataset\validation_data_200.npy")
OCT_train, mask_train = train_data[:, 0], train_data[:, 1]
OCT_val, mask_val = val_data[:, 0],  val_data[:, 1]



mask_train = np.expand_dims(mask_train, axis=-1)
mask_val = np.expand_dims(mask_val, axis=-1)
OCT_train = np.expand_dims(OCT_train, axis=-1)
OCT_val = np.expand_dims(OCT_val, axis=-1)

if OCT_log == True:
    OCT_train = 10*np.log10(OCT_train)
    OCT_val = 10*np.log10(OCT_val)




t_dataset = tf.data.Dataset.from_tensor_slices((OCT_train, mask_train))
v_dataset = tf.data.Dataset.from_tensor_slices((OCT_val, mask_val))
test_dataset = v_dataset.batch(1)
train_dataset = t_dataset.shuffle(mask_train.shape[0]).batch(1)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


def train_step(model, optimizer, x_train, m_train, epoch_):
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = tf.keras.losses.BinaryCrossentropy()(predictions, m_train)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_loss.update_state(loss)
  if epoch_ == 0:
      print(train_loss.result())

def test_step(model, x_test, m_val):
  predictions = model(x_test, training=False)
  loss = tf.keras.losses.BinaryCrossentropy()(predictions, m_val)
  test_loss.update_state(loss)

train_losses=[]
validation_losses =[]
model.summary()

for epoch in range(500):
    optimizer = tf.keras.optimizers.Adam(lr_decay(0.001, epoch))

    for (x_train, m_train) in train_dataset:
        train_step(model, optimizer, x_train, m_train, epoch)
    for (x_test, m_test) in test_dataset:
        test_step(model, x_test,  m_test)
    train_losses.append(train_loss.result())
    validation_losses.append(test_loss.result())
    template = 'Epoch {}, train_Loss: {},  Test Loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          test_loss.result()))
    early_stop(test_loss.result(), model)
    train_loss.reset_states()
    test_loss.reset_states()
    if early_stop.early_stop:
        print('early_stop')
        break

    # Reset metrics every epoch


with open(r"E:\Segmentation\Lecture\Dataset\NNmodel\loss\trainloss.txt",'w') as tl:
    for value in train_losses:
        tl.write(str(np.array(value)) + '\n')
tl.close()
with open(r"E:\Segmentation\Lecture\Dataset\NNmodel\loss\valloss.txt",'w') as vl:
    for value in validation_losses:
        vl.write(str(np.array(value)) + '\n')
vl.close()
plot_path=r"E:\Segmentation\Lecture\Dataset\NNmodel\loss\learning_curve.jpg"
plt.figure(figsize=(8, 8))
plt.plot(train_losses)
plt.plot(validation_losses)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.title("Learning curve", fontsize=20)
plt.legend(['training loss', 'validation loss'], fontsize=15, loc='upper right')
plt.savefig(plot_path)