import numpy as np
import pandas as pd
import deepchem as dc
import torch
from deepchem.models import AttentiveFPModel

df = pd.read_csv('Training_Smiles.csv')

df_train = pd.read_csv('Training_Smiles.csv')
df_test = pd.read_csv('Testing_Smiles.csv')


smiles_train = df_train['Smiles']
label_train = df_train['Property']

smiles_test = df_test['Smiles']
label_test = df_test['Property']

model = AttentiveFPModel(mode='classification', n_tasks=1, batch_size=128, learning_rate=0.001)

featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
X_train = featurizer.featurize(smiles_train)
X_test = featurizer.featurize(smiles_test)

training_dataset = dc.data.NumpyDataset(X=X_train, y=label_train)
testing_dataset = dc.data.NumpyDataset(X=X_test, y=label_test)
print("Training Begin")
loss = model.fit(dataset=training_dataset, nb_epoch=50, checkpoint_interval=1)
print("Training End")
print("LOSS:", loss)

def test(model: AttentiveFPModel, testing_dataset):
    predictions = model.predict(testing_dataset)
    predictions = np.argmax(predictions, axis=1)

    acc = np.sum(predictions == testing_dataset.y.tolist())
    return acc/len(testing_dataset)


print("Testing...")
acc_test = test(model, testing_dataset)
acc_train = test(model, training_dataset)
print("Accuracy on training dataset:", acc_train)
print("Accuracy on test dataset:", acc_test)

