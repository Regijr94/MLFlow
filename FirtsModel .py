#---------------- IMportando os pacotes necessarios
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt


import mlflow
import mlflow.sklearn


#----------------- Lendo o arquivo 

credito = pd.read_csv("download/Credit.csv")

print(credito.shape)

print(credito.head())

#------------------

for col in credito.columns:
    if credito[col].dtype == 'object':
        credito[col] = credito[col].astype('category').cat.codes
        
print(credito.head())

#------------------
predicao = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

#------------------

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(predicao, classe, 
  
                                                                 test_size=0.3, random_state=123)
mlflow.set_experiment("nbexperimento")

with mlflow.start_run():
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_treinamento, y_treinamento)

    predicao = naive_bayes.predict(x_teste)

    #métrica
    acuracia = accuracy_score(y_teste, predicao)
    recall = recall_score(y_teste, predicao)
    precision = precision_score(y_teste, predicao)
    f1 = f1_score(y_teste, predicao)
    auc= roc_auc_score(y_teste, predicao)
    log= log_loss(y_teste, predicao)

    #registrar métrica
    mlflow.log_metric("acuracia", acuracia)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("log", log)
    
    #gráficos 
    confusion = ConfusionMatrixDisplay.from_estimator(naive_bayes, x_teste, y_teste)
    plt.savefig("confusion.png")    
    
    roc = RocCurveDisplay.from_estimator(naive_bayes, x_teste, y_teste)
    plt.savefig('roc.png')
    
    # logar gráficos
    mlflow.log_artifact("confusion.png")
    mlflow.log_artifact("roc.png")
    
    #modelo
    mlflow.sklearn.log_model(naive_bayes, "ModeloNB")
    
    #informacoes da execução
    print("Modelo ", mlflow.active_run().info.run_uuid)

mlflow.end_run()

