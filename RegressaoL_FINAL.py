#Deteccao de Intrusao usando Sklearn
#Hugo Sousa

# variaveis => metodo_tipoDePreProcessamento_nomeDaVariavel

# Metodos
#K-Nearest Neighbour -> knn
#Support Vector Machine -> svm
#Naive Bayes -> nb
#Arvore de Decisao -> ad
#Random Forest -> rf
#Regressao Logıstica -> lr

# Tipo de PreProcessamento
#Padronizacao -> p
#One-hot -> oh
#Padronizacao + One-hot -> poh
#Sem preprocessamento -> spp

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#numero de iteracoes para treino e teste
count_test = 5 

#carregando a base de dados NSL KDD
nsl_kdd_data = pd.read_csv('./KDDTrain+.csv')

#Regressão Logistisca

#Padronizao + OneHot
lr_poh_timePP = []
lr_poh_timeTrain = []
lr_poh_timeTest = []

lr_poh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    lr_poh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    lr_poh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_lr_poh_atributos = LabelEncoder()
    lr_poh_atributos[:, 1] = labelencoder_lr_poh_atributos.fit_transform(lr_poh_atributos[:, 1])
    lr_poh_atributos[:, 2] = labelencoder_lr_poh_atributos.fit_transform(lr_poh_atributos[:, 2])
    lr_poh_atributos[:, 3] = labelencoder_lr_poh_atributos.fit_transform(lr_poh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_lr_poh_classes= LabelEncoder()
    lr_poh_classes = labelencoder_lr_poh_classes.fit_transform(lr_poh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    lr_poh_atributos = onehotencoder.fit_transform(lr_poh_atributos).toarray()
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    lr_poh_atributos = scaler.fit_transform(lr_poh_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    lr_poh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    lr_poh_atributos_train, lr_poh_atributos_test, lr_poh_classes_train, lr_poh_classes_test = train_test_split(lr_poh_atributos, lr_poh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = LogisticRegression()
    classificador.fit(lr_poh_atributos_train, lr_poh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    lr_poh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    lr_poh_previsoes = classificador.predict(lr_poh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    lr_poh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    lr_poh_acuracia.append(accuracy_score(lr_poh_classes_test, lr_poh_previsoes))
    
#medias
lr_poh_mdTimePP = np.mean(lr_poh_timePP)
lr_poh_mdTimeTrain = np.mean(lr_poh_timeTrain)
lr_poh_mdTimeTest = np.mean(lr_poh_timeTest)

lr_poh_sm_md_Time = lr_poh_mdTimePP + lr_poh_mdTimeTrain + lr_poh_mdTimeTest

lr_poh_mdAcuracia = np.mean(lr_poh_acuracia)

#desvio padrao
lr_poh_stdTimePP = np.std(lr_poh_timePP)
lr_poh_stdTimeTrain = np.std(lr_poh_timeTrain)
lr_poh_stdTimeTest = np.std(lr_poh_timeTest)

lr_poh_stdAcuracia = np.std(lr_poh_acuracia)






#Padronizao
lr_p_timePP = []
lr_p_timeTrain = []
lr_p_timeTest = []

lr_p_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    lr_p_atributos = nsl_kdd_data.iloc[:, 0:41].values
    lr_p_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_lr_p_atributos = LabelEncoder()
    lr_p_atributos[:, 1] = labelencoder_lr_p_atributos.fit_transform(lr_p_atributos[:, 1])
    lr_p_atributos[:, 2] = labelencoder_lr_p_atributos.fit_transform(lr_p_atributos[:, 2])
    lr_p_atributos[:, 3] = labelencoder_lr_p_atributos.fit_transform(lr_p_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_lr_p_classes= LabelEncoder()
    lr_p_classes = labelencoder_lr_p_classes.fit_transform(lr_p_classes)
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    lr_p_atributos = scaler.fit_transform(lr_p_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    lr_p_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    lr_p_atributos_train, lr_p_atributos_test, lr_p_classes_train, lr_p_classes_test = train_test_split(lr_p_atributos, lr_p_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = LogisticRegression()
    classificador.fit(lr_p_atributos_train, lr_p_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    lr_p_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    lr_p_previsoes = classificador.predict(lr_p_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    lr_p_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    lr_p_acuracia.append(accuracy_score(lr_p_classes_test, lr_p_previsoes))
    
#medias
lr_p_mdTimePP = np.mean(lr_p_timePP)
lr_p_mdTimeTrain = np.mean(lr_p_timeTrain)
lr_p_mdTimeTest = np.mean(lr_p_timeTest)

lr_p_mdAcuracia = np.mean(lr_p_acuracia)

lr_p_sm_md_Time = lr_p_mdTimePP + lr_p_mdTimeTrain + lr_p_mdTimeTest

#desvio padrao
lr_p_stdTimePP = np.std(lr_p_timePP)
lr_p_stdTimeTrain = np.std(lr_p_timeTrain)
lr_p_stdTimeTest = np.std(lr_p_timeTest)

lr_p_stdAcuracia = np.std(lr_p_acuracia)






#OneHot
lr_oh_timePP = []
lr_oh_timeTrain = []
lr_oh_timeTest = []

lr_oh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    lr_oh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    lr_oh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_lr_oh_atributos = LabelEncoder()
    lr_oh_atributos[:, 1] = labelencoder_lr_oh_atributos.fit_transform(lr_oh_atributos[:, 1])
    lr_oh_atributos[:, 2] = labelencoder_lr_oh_atributos.fit_transform(lr_oh_atributos[:, 2])
    lr_oh_atributos[:, 3] = labelencoder_lr_oh_atributos.fit_transform(lr_oh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_lr_oh_classes= LabelEncoder()
    lr_oh_classes = labelencoder_lr_oh_classes.fit_transform(lr_oh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    lr_oh_atributos = onehotencoder.fit_transform(lr_oh_atributos).toarray()
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    lr_oh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    lr_oh_atributos_train, lr_oh_atributos_test, lr_oh_classes_train, lr_oh_classes_test = train_test_split(lr_oh_atributos, lr_oh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = LogisticRegression()
    classificador.fit(lr_oh_atributos_train, lr_oh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    lr_oh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    lr_oh_previsoes = classificador.predict(lr_oh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    lr_oh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    lr_oh_acuracia.append(accuracy_score(lr_oh_classes_test, lr_oh_previsoes))
    
#medias
lr_oh_mdTimePP = np.mean(lr_oh_timePP)
lr_oh_mdTimeTrain = np.mean(lr_oh_timeTrain)
lr_oh_mdTimeTest = np.mean(lr_oh_timeTest)

lr_oh_sm_md_Time = lr_oh_mdTimePP + lr_oh_mdTimeTrain + lr_oh_mdTimeTest

lr_oh_mdAcuracia = np.mean(lr_oh_acuracia)

#desvio padrao
lr_oh_stdTimePP = np.std(lr_oh_timePP)
lr_oh_stdTimeTrain = np.std(lr_oh_timeTrain)
lr_oh_stdTimeTest = np.std(lr_oh_timeTest)

lr_oh_stdAcuracia = np.std(lr_oh_acuracia)






#Sem pradronizar e sem onehot
lr_spp_timePP = []
lr_spp_timeTrain = []
lr_spp_timeTest = []

lr_spp_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    lr_spp_atributos = nsl_kdd_data.iloc[:, 0:41].values
    lr_spp_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_lr_spp_atributos = LabelEncoder()
    lr_spp_atributos[:, 1] = labelencoder_lr_spp_atributos.fit_transform(lr_spp_atributos[:, 1])
    lr_spp_atributos[:, 2] = labelencoder_lr_spp_atributos.fit_transform(lr_spp_atributos[:, 2])
    lr_spp_atributos[:, 3] = labelencoder_lr_spp_atributos.fit_transform(lr_spp_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_lr_spp_classes= LabelEncoder()
    lr_spp_classes = labelencoder_lr_spp_classes.fit_transform(lr_spp_classes)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    lr_spp_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    lr_spp_atributos_train, lr_spp_atributos_test, lr_spp_classes_train, lr_spp_classes_test = train_test_split(lr_spp_atributos, lr_spp_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = LogisticRegression()
    classificador.fit(lr_spp_atributos_train, lr_spp_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    lr_spp_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    lr_spp_previsoes = classificador.predict(lr_spp_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    lr_spp_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    lr_spp_acuracia.append(accuracy_score(lr_spp_classes_test, lr_spp_previsoes))
    
#medias
lr_spp_mdTimePP = np.mean(lr_spp_timePP)
lr_spp_mdTimeTrain = np.mean(lr_spp_timeTrain)
lr_spp_mdTimeTest = np.mean(lr_spp_timeTest)

lr_spp_sm_md_Time = lr_spp_mdTimePP + lr_spp_mdTimeTrain + lr_spp_mdTimeTest

lr_spp_mdAcuracia = np.mean(lr_spp_acuracia)

#desvio padrao
lr_spp_stdTimePP = np.std(lr_spp_timePP)
lr_spp_stdTimeTrain = np.std(lr_spp_timeTrain)
lr_spp_stdTimeTest = np.std(lr_spp_timeTest)

lr_spp_stdAcuracia = np.std(lr_spp_acuracia)