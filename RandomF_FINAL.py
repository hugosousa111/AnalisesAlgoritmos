#Deteccao de Intrusao usando Sklearn
#Hugo Sousa

# variaveis => metodo_tipoDePreProcessamento_nomeDaVariavel

# Metodos
#K-Nearest Neighbour -> knn
#Support Vector Machine -> svm
#Naive Bayes -> nb
#Arvore de Decisao -> ad
#Random Forest -> rf
#Regressao Logıstica -> rl

# Tipo de PreProcessamento
#Padronizacao -> p
#One-hot -> oh
#Padronizacao + One-hot -> poh

# Nome Das Variaveis de tempo
# Tempo de PreProcessamento -> timePP
# Tempo de Treino -> timeTrain
# Tempo do Teste -> timeTest

# Nome Das Variaveis de media de tempo
# Media de Tempo de PreProcessamento -> mdTimePP
# Media de Tempo de Treino -> mdTimeTrain
# Media de Tempo do Teste -> mdTimeTest
# Soma das Medias de Tempo -> sm_md_Time

# Nome Das Variaveis de desvio padrao de tempo
# Desvio padrao de Tempo de PreProcessamento -> stdTimePP
# Desvio padrao de Tempo de Treino -> stdTimeTrain
# Desvio padrao de Tempo do Teste -> stdTimeTest

# Nome da Acuracia
# Acuracias -> acuracia
# Media acuracias -> mdAcuracia
# Desvio Padrao acuracias -> sdtAcuracia

#bibliotecas
import time #para calculo do tempo de processamento
import numpy as np #para calcular a media e o desvio padrao
import pandas as pd #para leitura da base csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#numero de iteracoes para treino e teste
count_test = 5 

#carregando a base de dados NSL KDD
nsl_kdd_data = pd.read_csv('./KDDTrain+.csv')

#Ramdom Forest

#Padronizao + OneHot
rf_poh_timePP = []
rf_poh_timeTrain = []
rf_poh_timeTest = []

rf_poh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    rf_poh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    rf_poh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_rf_poh_atributos = LabelEncoder()
    rf_poh_atributos[:, 1] = labelencoder_rf_poh_atributos.fit_transform(rf_poh_atributos[:, 1])
    rf_poh_atributos[:, 2] = labelencoder_rf_poh_atributos.fit_transform(rf_poh_atributos[:, 2])
    rf_poh_atributos[:, 3] = labelencoder_rf_poh_atributos.fit_transform(rf_poh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_rf_poh_classes= LabelEncoder()
    rf_poh_classes = labelencoder_rf_poh_classes.fit_transform(rf_poh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    rf_poh_atributos = onehotencoder.fit_transform(rf_poh_atributos).toarray()
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    rf_poh_atributos = scaler.fit_transform(rf_poh_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    rf_poh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    rf_poh_atributos_train, rf_poh_atributos_test, rf_poh_classes_train, rf_poh_classes_test = train_test_split(rf_poh_atributos, rf_poh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = RandomForestClassifier(n_estimators = 5 ,criterion='entropy')
    classificador.fit(rf_poh_atributos_train, rf_poh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    rf_poh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    rf_poh_previsoes = classificador.predict(rf_poh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    rf_poh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    rf_poh_acuracia.append(accuracy_score(rf_poh_classes_test, rf_poh_previsoes))
    
#medias
rf_poh_mdTimePP = np.mean(rf_poh_timePP)
rf_poh_mdTimeTrain = np.mean(rf_poh_timeTrain)
rf_poh_mdTimeTest = np.mean(rf_poh_timeTest)

rf_poh_sm_md_Time = rf_poh_mdTimePP + rf_poh_mdTimeTrain + rf_poh_mdTimeTest

rf_poh_mdAcuracia = np.mean(rf_poh_acuracia)

#desvio padrao
rf_poh_stdTimePP = np.std(rf_poh_timePP)
rf_poh_stdTimeTrain = np.std(rf_poh_timeTrain)
rf_poh_stdTimeTest = np.std(rf_poh_timeTest)

rf_poh_stdAcuracia = np.std(rf_poh_acuracia)






#Padronizao
rf_p_timePP = []
rf_p_timeTrain = []
rf_p_timeTest = []

rf_p_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    rf_p_atributos = nsl_kdd_data.iloc[:, 0:41].values
    rf_p_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_rf_p_atributos = LabelEncoder()
    rf_p_atributos[:, 1] = labelencoder_rf_p_atributos.fit_transform(rf_p_atributos[:, 1])
    rf_p_atributos[:, 2] = labelencoder_rf_p_atributos.fit_transform(rf_p_atributos[:, 2])
    rf_p_atributos[:, 3] = labelencoder_rf_p_atributos.fit_transform(rf_p_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_rf_p_classes= LabelEncoder()
    rf_p_classes = labelencoder_rf_p_classes.fit_transform(rf_p_classes)
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    rf_p_atributos = scaler.fit_transform(rf_p_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    rf_p_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    rf_p_atributos_train, rf_p_atributos_test, rf_p_classes_train, rf_p_classes_test = train_test_split(rf_p_atributos, rf_p_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = RandomForestClassifier(n_estimators = 5 ,criterion='entropy')
    classificador.fit(rf_p_atributos_train, rf_p_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    rf_p_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    rf_p_previsoes = classificador.predict(rf_p_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    rf_p_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    rf_p_acuracia.append(accuracy_score(rf_p_classes_test, rf_p_previsoes))
    
#medias
rf_p_mdTimePP = np.mean(rf_p_timePP)
rf_p_mdTimeTrain = np.mean(rf_p_timeTrain)
rf_p_mdTimeTest = np.mean(rf_p_timeTest)

rf_p_mdAcuracia = np.mean(rf_p_acuracia)

rf_p_sm_md_Time = rf_p_mdTimePP + rf_p_mdTimeTrain + rf_p_mdTimeTest

#desvio padrao
rf_p_stdTimePP = np.std(rf_p_timePP)
rf_p_stdTimeTrain = np.std(rf_p_timeTrain)
rf_p_stdTimeTest = np.std(rf_p_timeTest)

rf_p_stdAcuracia = np.std(rf_p_acuracia)






#OneHot
rf_oh_timePP = []
rf_oh_timeTrain = []
rf_oh_timeTest = []

rf_oh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    rf_oh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    rf_oh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_rf_oh_atributos = LabelEncoder()
    rf_oh_atributos[:, 1] = labelencoder_rf_oh_atributos.fit_transform(rf_oh_atributos[:, 1])
    rf_oh_atributos[:, 2] = labelencoder_rf_oh_atributos.fit_transform(rf_oh_atributos[:, 2])
    rf_oh_atributos[:, 3] = labelencoder_rf_oh_atributos.fit_transform(rf_oh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_rf_oh_classes= LabelEncoder()
    rf_oh_classes = labelencoder_rf_oh_classes.fit_transform(rf_oh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    rf_oh_atributos = onehotencoder.fit_transform(rf_oh_atributos).toarray()
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    rf_oh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    rf_oh_atributos_train, rf_oh_atributos_test, rf_oh_classes_train, rf_oh_classes_test = train_test_split(rf_oh_atributos, rf_oh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = RandomForestClassifier(n_estimators = 5 ,criterion='entropy')
    classificador.fit(rf_oh_atributos_train, rf_oh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    rf_oh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    rf_oh_previsoes = classificador.predict(rf_oh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    rf_oh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    rf_oh_acuracia.append(accuracy_score(rf_oh_classes_test, rf_oh_previsoes))
    
#medias
rf_oh_mdTimePP = np.mean(rf_oh_timePP)
rf_oh_mdTimeTrain = np.mean(rf_oh_timeTrain)
rf_oh_mdTimeTest = np.mean(rf_oh_timeTest)

rf_oh_sm_md_Time = rf_oh_mdTimePP + rf_oh_mdTimeTrain + rf_oh_mdTimeTest

rf_oh_mdAcuracia = np.mean(rf_oh_acuracia)

#desvio padrao
rf_oh_stdTimePP = np.std(rf_oh_timePP)
rf_oh_stdTimeTrain = np.std(rf_oh_timeTrain)
rf_oh_stdTimeTest = np.std(rf_oh_timeTest)

rf_oh_stdAcuracia = np.std(rf_oh_acuracia)






#Sem pradronizar e sem onehot
rf_spp_timePP = []
rf_spp_timeTrain = []
rf_spp_timeTest = []

rf_spp_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    rf_spp_atributos = nsl_kdd_data.iloc[:, 0:41].values
    rf_spp_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_rf_spp_atributos = LabelEncoder()
    rf_spp_atributos[:, 1] = labelencoder_rf_spp_atributos.fit_transform(rf_spp_atributos[:, 1])
    rf_spp_atributos[:, 2] = labelencoder_rf_spp_atributos.fit_transform(rf_spp_atributos[:, 2])
    rf_spp_atributos[:, 3] = labelencoder_rf_spp_atributos.fit_transform(rf_spp_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_rf_spp_classes= LabelEncoder()
    rf_spp_classes = labelencoder_rf_spp_classes.fit_transform(rf_spp_classes)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    rf_spp_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    rf_spp_atributos_train, rf_spp_atributos_test, rf_spp_classes_train, rf_spp_classes_test = train_test_split(rf_spp_atributos, rf_spp_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = RandomForestClassifier(n_estimators = 5 ,criterion='entropy')
    classificador.fit(rf_spp_atributos_train, rf_spp_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    rf_spp_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    rf_spp_previsoes = classificador.predict(rf_spp_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    rf_spp_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    rf_spp_acuracia.append(accuracy_score(rf_spp_classes_test, rf_spp_previsoes))
    
#medias
rf_spp_mdTimePP = np.mean(rf_spp_timePP)
rf_spp_mdTimeTrain = np.mean(rf_spp_timeTrain)
rf_spp_mdTimeTest = np.mean(rf_spp_timeTest)

rf_spp_sm_md_Time = rf_spp_mdTimePP + rf_spp_mdTimeTrain + rf_spp_mdTimeTest

rf_spp_mdAcuracia = np.mean(rf_spp_acuracia)

#desvio padrao
rf_spp_stdTimePP = np.std(rf_spp_timePP)
rf_spp_stdTimeTrain = np.std(rf_spp_timeTrain)
rf_spp_stdTimeTest = np.std(rf_spp_timeTest)
