import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

dados = pd.read_csv("iris.csv")

previsores = dados.iloc[:,0:4].values
classe = dados.iloc[:,4].values


X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe,
                                                                  test_size = 0.3)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

previsoes = naive_bayes.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

print(taxa_acerto)
print("\n")

taxa_erro = 1 - taxa_acerto


from yellowbrick.classifier import ConfusionMatrix
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()