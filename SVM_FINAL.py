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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#numero de iteracoes para treino e teste
count_test = 5 

#carregando a base de dados NSL KDD
nsl_kdd_data = pd.read_csv('./KDDTrain+.csv')

#Naive Bayes

#Padronizao + OneHot
svm_poh_timePP = []
svm_poh_timeTrain = []
svm_poh_timeTest = []

svm_poh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    svm_poh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    svm_poh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_svm_poh_atributos = LabelEncoder()
    svm_poh_atributos[:, 1] = labelencoder_svm_poh_atributos.fit_transform(svm_poh_atributos[:, 1])
    svm_poh_atributos[:, 2] = labelencoder_svm_poh_atributos.fit_transform(svm_poh_atributos[:, 2])
    svm_poh_atributos[:, 3] = labelencoder_svm_poh_atributos.fit_transform(svm_poh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_svm_poh_classes= LabelEncoder()
    svm_poh_classes = labelencoder_svm_poh_classes.fit_transform(svm_poh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    svm_poh_atributos = onehotencoder.fit_transform(svm_poh_atributos).toarray()
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    svm_poh_atributos = scaler.fit_transform(svm_poh_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    svm_poh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    svm_poh_atributos_train, svm_poh_atributos_test, svm_poh_classes_train, svm_poh_classes_test = train_test_split(svm_poh_atributos, svm_poh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = SVC(kernel = 'rbf')
    classificador.fit(svm_poh_atributos_train, svm_poh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    svm_poh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    svm_poh_previsoes = classificador.predict(svm_poh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    svm_poh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    svm_poh_acuracia.append(accuracy_score(svm_poh_classes_test, svm_poh_previsoes))
    
#medias
svm_poh_mdTimePP = np.mean(svm_poh_timePP)
svm_poh_mdTimeTrain = np.mean(svm_poh_timeTrain)
svm_poh_mdTimeTest = np.mean(svm_poh_timeTest)

svm_poh_sm_md_Time = svm_poh_mdTimePP + svm_poh_mdTimeTrain + svm_poh_mdTimeTest

svm_poh_mdAcuracia = np.mean(svm_poh_acuracia)

#desvio padrao
svm_poh_stdTimePP = np.std(svm_poh_timePP)
svm_poh_stdTimeTrain = np.std(svm_poh_timeTrain)
svm_poh_stdTimeTest = np.std(svm_poh_timeTest)

svm_poh_stdAcuracia = np.std(svm_poh_acuracia)






#Padronizao
svm_p_timePP = []
svm_p_timeTrain = []
svm_p_timeTest = []

svm_p_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    svm_p_atributos = nsl_kdd_data.iloc[:, 0:41].values
    svm_p_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_svm_p_atributos = LabelEncoder()
    svm_p_atributos[:, 1] = labelencoder_svm_p_atributos.fit_transform(svm_p_atributos[:, 1])
    svm_p_atributos[:, 2] = labelencoder_svm_p_atributos.fit_transform(svm_p_atributos[:, 2])
    svm_p_atributos[:, 3] = labelencoder_svm_p_atributos.fit_transform(svm_p_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_svm_p_classes= LabelEncoder()
    svm_p_classes = labelencoder_svm_p_classes.fit_transform(svm_p_classes)
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    svm_p_atributos = scaler.fit_transform(svm_p_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    svm_p_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    svm_p_atributos_train, svm_p_atributos_test, svm_p_classes_train, svm_p_classes_test = train_test_split(svm_p_atributos, svm_p_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = SVC(kernel = 'rbf')
    classificador.fit(svm_p_atributos_train, svm_p_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    svm_p_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    svm_p_previsoes = classificador.predict(svm_p_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    svm_p_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    svm_p_acuracia.append(accuracy_score(svm_p_classes_test, svm_p_previsoes))
    
#medias
svm_p_mdTimePP = np.mean(svm_p_timePP)
svm_p_mdTimeTrain = np.mean(svm_p_timeTrain)
svm_p_mdTimeTest = np.mean(svm_p_timeTest)

svm_p_mdAcuracia = np.mean(svm_p_acuracia)

svm_p_sm_md_Time = svm_p_mdTimePP + svm_p_mdTimeTrain + svm_p_mdTimeTest

#desvio padrao
svm_p_stdTimePP = np.std(svm_p_timePP)
svm_p_stdTimeTrain = np.std(svm_p_timeTrain)
svm_p_stdTimeTest = np.std(svm_p_timeTest)

svm_p_stdAcuracia = np.std(svm_p_acuracia)






#OneHot
svm_oh_timePP = []
svm_oh_timeTrain = []
svm_oh_timeTest = []

svm_oh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    svm_oh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    svm_oh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_svm_oh_atributos = LabelEncoder()
    svm_oh_atributos[:, 1] = labelencoder_svm_oh_atributos.fit_transform(svm_oh_atributos[:, 1])
    svm_oh_atributos[:, 2] = labelencoder_svm_oh_atributos.fit_transform(svm_oh_atributos[:, 2])
    svm_oh_atributos[:, 3] = labelencoder_svm_oh_atributos.fit_transform(svm_oh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_svm_oh_classes= LabelEncoder()
    svm_oh_classes = labelencoder_svm_oh_classes.fit_transform(svm_oh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    svm_oh_atributos = onehotencoder.fit_transform(svm_oh_atributos).toarray()
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    svm_oh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    svm_oh_atributos_train, svm_oh_atributos_test, svm_oh_classes_train, svm_oh_classes_test = train_test_split(svm_oh_atributos, svm_oh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = SVC(kernel = 'rbf')
    classificador.fit(svm_oh_atributos_train, svm_oh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    svm_oh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    svm_oh_previsoes = classificador.predict(svm_oh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    svm_oh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    svm_oh_acuracia.append(accuracy_score(svm_oh_classes_test, svm_oh_previsoes))
    
#medias
svm_oh_mdTimePP = np.mean(svm_oh_timePP)
svm_oh_mdTimeTrain = np.mean(svm_oh_timeTrain)
svm_oh_mdTimeTest = np.mean(svm_oh_timeTest)

svm_oh_sm_md_Time = svm_oh_mdTimePP + svm_oh_mdTimeTrain + svm_oh_mdTimeTest

svm_oh_mdAcuracia = np.mean(svm_oh_acuracia)

#desvio padrao
svm_oh_stdTimePP = np.std(svm_oh_timePP)
svm_oh_stdTimeTrain = np.std(svm_oh_timeTrain)
svm_oh_stdTimeTest = np.std(svm_oh_timeTest)

svm_oh_stdAcuracia = np.std(svm_oh_acuracia)






#Sem pradronizar e sem onehot
svm_spp_timePP = []
svm_spp_timeTrain = []
svm_spp_timeTest = []

svm_spp_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    svm_spp_atributos = nsl_kdd_data.iloc[:, 0:41].values
    svm_spp_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_svm_spp_atributos = LabelEncoder()
    svm_spp_atributos[:, 1] = labelencoder_svm_spp_atributos.fit_transform(svm_spp_atributos[:, 1])
    svm_spp_atributos[:, 2] = labelencoder_svm_spp_atributos.fit_transform(svm_spp_atributos[:, 2])
    svm_spp_atributos[:, 3] = labelencoder_svm_spp_atributos.fit_transform(svm_spp_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_svm_spp_classes= LabelEncoder()
    svm_spp_classes = labelencoder_svm_spp_classes.fit_transform(svm_spp_classes)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    svm_spp_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    svm_spp_atributos_train, svm_spp_atributos_test, svm_spp_classes_train, svm_spp_classes_test = train_test_split(svm_spp_atributos, svm_spp_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = SVC(kernel = 'rbf')
    classificador.fit(svm_spp_atributos_train, svm_spp_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    svm_spp_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    svm_spp_previsoes = classificador.predict(svm_spp_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    svm_spp_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    svm_spp_acuracia.append(accuracy_score(svm_spp_classes_test, svm_spp_previsoes))
    
#medias
svm_spp_mdTimePP = np.mean(svm_spp_timePP)
svm_spp_mdTimeTrain = np.mean(svm_spp_timeTrain)
svm_spp_mdTimeTest = np.mean(svm_spp_timeTest)

svm_spp_sm_md_Time = svm_spp_mdTimePP + svm_spp_mdTimeTrain + svm_spp_mdTimeTest

svm_spp_mdAcuracia = np.mean(svm_spp_acuracia)

#desvio padrao
svm_spp_stdTimePP = np.std(svm_spp_timePP)
svm_spp_stdTimeTrain = np.std(svm_spp_timeTrain)
svm_spp_stdTimeTest = np.std(svm_spp_timeTest)

svm_spp_stdAcuracia = np.std(svm_spp_acuracia)