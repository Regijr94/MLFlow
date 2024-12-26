#---------------- IMportando os pacotes necessarios
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

def treinar_rf(n_estimators):
    mlflow.set_experiment("rfexperimento")

    with mlflow.start_run():
        # n_estimators=30
        modelorf = RandomForestClassifier(n_estimators=n_estimators)
        modelorf.fit(x_treinamento, y_treinamento)

        predicao = modelorf.predict(x_teste)

        # Métrica
        acuracia = accuracy_score(y_teste, predicao)
        recall = recall_score(y_teste, predicao)
        precision = precision_score(y_teste, predicao)
        f1 = f1_score(y_teste, predicao)
        auc= roc_auc_score(y_teste, predicao)
        log= log_loss(y_teste, predicao)
        # Log de hiperparamentos 
        mlflow.log_param("n_estimators", n_estimators)
        # Registrar métrica
        mlflow.log_metric("acuracia", acuracia)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("log", log)
        
        #gráficos 
        confusion = ConfusionMatrixDisplay.from_estimator(modelorf, x_teste, y_teste)
        plt.savefig("confusionrf.png")    
        
        roc = RocCurveDisplay.from_estimator(modelorf, x_teste, y_teste)
        plt.savefig('rocrf.png')
        
        # logar gráficos
        mlflow.log_artifact("confusionrf.png")
        mlflow.log_artifact("rocrf.png")
        
        #modelo
        mlflow.sklearn.log_model(modelorf, "ModeloNB")
        
        #informacoes da execução
        print("Modelo ", mlflow.active_run().info.run_uuid)

    mlflow.end_run()

arvores = [50,100,500,750,1000]

for n in arvores:
    treinar_rf(n)