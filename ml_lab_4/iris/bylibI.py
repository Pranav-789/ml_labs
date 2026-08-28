import lib
import pandas as pd

df = pd.read_csv('Iris.csv', comment='#')
data = df.values.tolist()

data = lib.encode_class(data)

for i in range(len(data)):
    for j in range(len(data[i]) - 1):
        data[i][j] = float(data[i][j])

ratio = 0.7
train_data, test_data = lib.splitting(data, ratio)

print('Total number of examples:', len(data))
print('Training examples:', len(train_data))
print('Test examples:', len(test_data))

info = lib.MeanAndStdDevForClass(train_data)

predictions = lib.getPredictions(info, test_data)
accuracy = lib.accuracy_rate(test_data, predictions)
print('Accuracy of the model:', accuracy)