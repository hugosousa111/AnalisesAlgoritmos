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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#numero de iteracoes para treino e teste
count_test = 5

#carregando a base de dados NSL KDD
nsl_kdd_data = pd.read_csv('./KDDTrain+.csv')

#Naive Bayes

#Padronizao + OneHot
knn_poh_timePP = []
knn_poh_timeTrain = []
knn_poh_timeTest = []

knn_poh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    knn_poh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    knn_poh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_knn_poh_atributos = LabelEncoder()
    knn_poh_atributos[:, 1] = labelencoder_knn_poh_atributos.fit_transform(knn_poh_atributos[:, 1])
    knn_poh_atributos[:, 2] = labelencoder_knn_poh_atributos.fit_transform(knn_poh_atributos[:, 2])
    knn_poh_atributos[:, 3] = labelencoder_knn_poh_atributos.fit_transform(knn_poh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_knn_poh_classes= LabelEncoder()
    knn_poh_classes = labelencoder_knn_poh_classes.fit_transform(knn_poh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    knn_poh_atributos = onehotencoder.fit_transform(knn_poh_atributos).toarray()
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    knn_poh_atributos = scaler.fit_transform(knn_poh_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    knn_poh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    knn_poh_atributos_train, knn_poh_atributos_test, knn_poh_classes_train, knn_poh_classes_test = train_test_split(knn_poh_atributos, knn_poh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    classificador.fit(knn_poh_atributos_train, knn_poh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    knn_poh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    knn_poh_previsoes = classificador.predict(knn_poh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    knn_poh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    knn_poh_acuracia.append(accuracy_score(knn_poh_classes_test, knn_poh_previsoes))
    
#medias
knn_poh_mdTimePP = np.mean(knn_poh_timePP)
knn_poh_mdTimeTrain = np.mean(knn_poh_timeTrain)
knn_poh_mdTimeTest = np.mean(knn_poh_timeTest)

knn_poh_sm_md_Time = knn_poh_mdTimePP + knn_poh_mdTimeTrain + knn_poh_mdTimeTest

knn_poh_mdAcuracia = np.mean(knn_poh_acuracia)

#desvio padrao
knn_poh_stdTimePP = np.std(knn_poh_timePP)
knn_poh_stdTimeTrain = np.std(knn_poh_timeTrain)
knn_poh_stdTimeTest = np.std(knn_poh_timeTest)

knn_poh_stdAcuracia = np.std(knn_poh_acuracia)






#Padronizao
knn_p_timePP = []
knn_p_timeTrain = []
knn_p_timeTest = []

knn_p_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    knn_p_atributos = nsl_kdd_data.iloc[:, 0:41].values
    knn_p_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_knn_p_atributos = LabelEncoder()
    knn_p_atributos[:, 1] = labelencoder_knn_p_atributos.fit_transform(knn_p_atributos[:, 1])
    knn_p_atributos[:, 2] = labelencoder_knn_p_atributos.fit_transform(knn_p_atributos[:, 2])
    knn_p_atributos[:, 3] = labelencoder_knn_p_atributos.fit_transform(knn_p_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_knn_p_classes= LabelEncoder()
    knn_p_classes = labelencoder_knn_p_classes.fit_transform(knn_p_classes)
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    knn_p_atributos = scaler.fit_transform(knn_p_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    knn_p_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    knn_p_atributos_train, knn_p_atributos_test, knn_p_classes_train, knn_p_classes_test = train_test_split(knn_p_atributos, knn_p_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    classificador.fit(knn_p_atributos_train, knn_p_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    knn_p_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    knn_p_previsoes = classificador.predict(knn_p_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    knn_p_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    knn_p_acuracia.append(accuracy_score(knn_p_classes_test, knn_p_previsoes))
    
#medias
knn_p_mdTimePP = np.mean(knn_p_timePP)
knn_p_mdTimeTrain = np.mean(knn_p_timeTrain)
knn_p_mdTimeTest = np.mean(knn_p_timeTest)

knn_p_mdAcuracia = np.mean(knn_p_acuracia)

knn_p_sm_md_Time = knn_p_mdTimePP + knn_p_mdTimeTrain + knn_p_mdTimeTest

#desvio padrao
knn_p_stdTimePP = np.std(knn_p_timePP)
knn_p_stdTimeTrain = np.std(knn_p_timeTrain)
knn_p_stdTimeTest = np.std(knn_p_timeTest)

knn_p_stdAcuracia = np.std(knn_p_acuracia)






#OneHot
knn_oh_timePP = []
knn_oh_timeTrain = []
knn_oh_timeTest = []

knn_oh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    knn_oh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    knn_oh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_knn_oh_atributos = LabelEncoder()
    knn_oh_atributos[:, 1] = labelencoder_knn_oh_atributos.fit_transform(knn_oh_atributos[:, 1])
    knn_oh_atributos[:, 2] = labelencoder_knn_oh_atributos.fit_transform(knn_oh_atributos[:, 2])
    knn_oh_atributos[:, 3] = labelencoder_knn_oh_atributos.fit_transform(knn_oh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_knn_oh_classes= LabelEncoder()
    knn_oh_classes = labelencoder_knn_oh_classes.fit_transform(knn_oh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    knn_oh_atributos = onehotencoder.fit_transform(knn_oh_atributos).toarray()
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    knn_oh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    knn_oh_atributos_train, knn_oh_atributos_test, knn_oh_classes_train, knn_oh_classes_test = train_test_split(knn_oh_atributos, knn_oh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    classificador.fit(knn_oh_atributos_train, knn_oh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    knn_oh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    knn_oh_previsoes = classificador.predict(knn_oh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    knn_oh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    knn_oh_acuracia.append(accuracy_score(knn_oh_classes_test, knn_oh_previsoes))
    
#medias
knn_oh_mdTimePP = np.mean(knn_oh_timePP)
knn_oh_mdTimeTrain = np.mean(knn_oh_timeTrain)
knn_oh_mdTimeTest = np.mean(knn_oh_timeTest)

knn_oh_sm_md_Time = knn_oh_mdTimePP + knn_oh_mdTimeTrain + knn_oh_mdTimeTest

knn_oh_mdAcuracia = np.mean(knn_oh_acuracia)

#desvio padrao
knn_oh_stdTimePP = np.std(knn_oh_timePP)
knn_oh_stdTimeTrain = np.std(knn_oh_timeTrain)
knn_oh_stdTimeTest = np.std(knn_oh_timeTest)

knn_oh_stdAcuracia = np.std(knn_oh_acuracia)






#Sem pradronizar e sem onehot
knn_spp_timePP = []
knn_spp_timeTrain = []
knn_spp_timeTest = []

knn_spp_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    knn_spp_atributos = nsl_kdd_data.iloc[:, 0:41].values
    knn_spp_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_knn_spp_atributos = LabelEncoder()
    knn_spp_atributos[:, 1] = labelencoder_knn_spp_atributos.fit_transform(knn_spp_atributos[:, 1])
    knn_spp_atributos[:, 2] = labelencoder_knn_spp_atributos.fit_transform(knn_spp_atributos[:, 2])
    knn_spp_atributos[:, 3] = labelencoder_knn_spp_atributos.fit_transform(knn_spp_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_knn_spp_classes= LabelEncoder()
    knn_spp_classes = labelencoder_knn_spp_classes.fit_transform(knn_spp_classes)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    knn_spp_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    knn_spp_atributos_train, knn_spp_atributos_test, knn_spp_classes_train, knn_spp_classes_test = train_test_split(knn_spp_atributos, knn_spp_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    classificador.fit(knn_spp_atributos_train, knn_spp_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    knn_spp_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    knn_spp_previsoes = classificador.predict(knn_spp_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    knn_spp_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    knn_spp_acuracia.append(accuracy_score(knn_spp_classes_test, knn_spp_previsoes))
    
#medias
knn_spp_mdTimePP = np.mean(knn_spp_timePP)
knn_spp_mdTimeTrain = np.mean(knn_spp_timeTrain)
knn_spp_mdTimeTest = np.mean(knn_spp_timeTest)

knn_spp_sm_md_Time = knn_spp_mdTimePP + knn_spp_mdTimeTrain + knn_spp_mdTimeTest

knn_spp_mdAcuracia = np.mean(knn_spp_acuracia)

#desvio padrao
knn_spp_stdTimePP = np.std(knn_spp_timePP)
knn_spp_stdTimeTrain = np.std(knn_spp_timeTrain)
knn_spp_stdTimeTest = np.std(knn_spp_timeTest)

knn_spp_stdAcuracia = np.std(knn_spp_acuracia)