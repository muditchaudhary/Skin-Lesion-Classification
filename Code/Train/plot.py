import pickle
import matplotlib.pyplot as plt

file_pi = open("trainHistoryDict", "rb")
history =  pickle.load(file_pi)

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

file_metric = open("metricsDict", "rb")
metrics =  pickle.load(file_metric)

i = [j+1 for j in range(len(metrics))]


plt.plot(i,[data['val_recall'] for data in metrics], 'r-')
plt.plot(i,[data['val_precision'] for data in metrics], 'k-')
plt.xlabel('epoch')
plt.legend(['recall', 'precision'], loc = 'upper left')
plt.show()


plt.plot(i,[data['val_f1_score'] for data in metrics], 'r-')
plt.xlabel('epoch')
plt.ylabel('f1_score')
plt.show()
