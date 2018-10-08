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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#numero de iteracoes para treino e teste
count_test = 5 

#carregando a base de dados NSL KDD
nsl_kdd_data = pd.read_csv('./KDDTrain+.csv')

#Naive Bayes

#Padronizao + OneHot
nb_poh_timePP = []
nb_poh_timeTrain = []
nb_poh_timeTest = []

nb_poh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    nb_poh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    nb_poh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_nb_poh_atributos = LabelEncoder()
    nb_poh_atributos[:, 1] = labelencoder_nb_poh_atributos.fit_transform(nb_poh_atributos[:, 1])
    nb_poh_atributos[:, 2] = labelencoder_nb_poh_atributos.fit_transform(nb_poh_atributos[:, 2])
    nb_poh_atributos[:, 3] = labelencoder_nb_poh_atributos.fit_transform(nb_poh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_nb_poh_classes= LabelEncoder()
    nb_poh_classes = labelencoder_nb_poh_classes.fit_transform(nb_poh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    nb_poh_atributos = onehotencoder.fit_transform(nb_poh_atributos).toarray()
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    nb_poh_atributos = scaler.fit_transform(nb_poh_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    nb_poh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    nb_poh_atributos_train, nb_poh_atributos_test, nb_poh_classes_train, nb_poh_classes_test = train_test_split(nb_poh_atributos, nb_poh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = GaussianNB()
    classificador.fit(nb_poh_atributos_train, nb_poh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    nb_poh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    nb_poh_previsoes = classificador.predict(nb_poh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    nb_poh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    nb_poh_acuracia.append(accuracy_score(nb_poh_classes_test, nb_poh_previsoes))
    
#medias
nb_poh_mdTimePP = np.mean(nb_poh_timePP)
nb_poh_mdTimeTrain = np.mean(nb_poh_timeTrain)
nb_poh_mdTimeTest = np.mean(nb_poh_timeTest)

nb_poh_sm_md_Time = nb_poh_mdTimePP + nb_poh_mdTimeTrain + nb_poh_mdTimeTest

nb_poh_mdAcuracia = np.mean(nb_poh_acuracia)

#desvio padrao
nb_poh_stdTimePP = np.std(nb_poh_timePP)
nb_poh_stdTimeTrain = np.std(nb_poh_timeTrain)
nb_poh_stdTimeTest = np.std(nb_poh_timeTest)

nb_poh_stdAcuracia = np.std(nb_poh_acuracia)






#Padronizao
nb_p_timePP = []
nb_p_timeTrain = []
nb_p_timeTest = []

nb_p_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    nb_p_atributos = nsl_kdd_data.iloc[:, 0:41].values
    nb_p_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_nb_p_atributos = LabelEncoder()
    nb_p_atributos[:, 1] = labelencoder_nb_p_atributos.fit_transform(nb_p_atributos[:, 1])
    nb_p_atributos[:, 2] = labelencoder_nb_p_atributos.fit_transform(nb_p_atributos[:, 2])
    nb_p_atributos[:, 3] = labelencoder_nb_p_atributos.fit_transform(nb_p_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_nb_p_classes= LabelEncoder()
    nb_p_classes = labelencoder_nb_p_classes.fit_transform(nb_p_classes)
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    nb_p_atributos = scaler.fit_transform(nb_p_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    nb_p_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    nb_p_atributos_train, nb_p_atributos_test, nb_p_classes_train, nb_p_classes_test = train_test_split(nb_p_atributos, nb_p_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = GaussianNB()
    classificador.fit(nb_p_atributos_train, nb_p_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    nb_p_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    nb_p_previsoes = classificador.predict(nb_p_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    nb_p_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    nb_p_acuracia.append(accuracy_score(nb_p_classes_test, nb_p_previsoes))
    
#medias
nb_p_mdTimePP = np.mean(nb_p_timePP)
nb_p_mdTimeTrain = np.mean(nb_p_timeTrain)
nb_p_mdTimeTest = np.mean(nb_p_timeTest)

nb_p_mdAcuracia = np.mean(nb_p_acuracia)

nb_p_sm_md_Time = nb_p_mdTimePP + nb_p_mdTimeTrain + nb_p_mdTimeTest

#desvio padrao
nb_p_stdTimePP = np.std(nb_p_timePP)
nb_p_stdTimeTrain = np.std(nb_p_timeTrain)
nb_p_stdTimeTest = np.std(nb_p_timeTest)

nb_p_stdAcuracia = np.std(nb_p_acuracia)






#OneHot
nb_oh_timePP = []
nb_oh_timeTrain = []
nb_oh_timeTest = []

nb_oh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    nb_oh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    nb_oh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_nb_oh_atributos = LabelEncoder()
    nb_oh_atributos[:, 1] = labelencoder_nb_oh_atributos.fit_transform(nb_oh_atributos[:, 1])
    nb_oh_atributos[:, 2] = labelencoder_nb_oh_atributos.fit_transform(nb_oh_atributos[:, 2])
    nb_oh_atributos[:, 3] = labelencoder_nb_oh_atributos.fit_transform(nb_oh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_nb_oh_classes= LabelEncoder()
    nb_oh_classes = labelencoder_nb_oh_classes.fit_transform(nb_oh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    nb_oh_atributos = onehotencoder.fit_transform(nb_oh_atributos).toarray()
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    nb_oh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    nb_oh_atributos_train, nb_oh_atributos_test, nb_oh_classes_train, nb_oh_classes_test = train_test_split(nb_oh_atributos, nb_oh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = GaussianNB()
    classificador.fit(nb_oh_atributos_train, nb_oh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    nb_oh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    nb_oh_previsoes = classificador.predict(nb_oh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    nb_oh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    nb_oh_acuracia.append(accuracy_score(nb_oh_classes_test, nb_oh_previsoes))
    
#medias
nb_oh_mdTimePP = np.mean(nb_oh_timePP)
nb_oh_mdTimeTrain = np.mean(nb_oh_timeTrain)
nb_oh_mdTimeTest = np.mean(nb_oh_timeTest)

nb_oh_sm_md_Time = nb_oh_mdTimePP + nb_oh_mdTimeTrain + nb_oh_mdTimeTest

nb_oh_mdAcuracia = np.mean(nb_oh_acuracia)

#desvio padrao
nb_oh_stdTimePP = np.std(nb_oh_timePP)
nb_oh_stdTimeTrain = np.std(nb_oh_timeTrain)
nb_oh_stdTimeTest = np.std(nb_oh_timeTest)

nb_oh_stdAcuracia = np.std(nb_oh_acuracia)






#Sem pradronizar e sem onehot
nb_spp_timePP = []
nb_spp_timeTrain = []
nb_spp_timeTest = []

nb_spp_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    nb_spp_atributos = nsl_kdd_data.iloc[:, 0:41].values
    nb_spp_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_nb_spp_atributos = LabelEncoder()
    nb_spp_atributos[:, 1] = labelencoder_nb_spp_atributos.fit_transform(nb_spp_atributos[:, 1])
    nb_spp_atributos[:, 2] = labelencoder_nb_spp_atributos.fit_transform(nb_spp_atributos[:, 2])
    nb_spp_atributos[:, 3] = labelencoder_nb_spp_atributos.fit_transform(nb_spp_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_nb_spp_classes= LabelEncoder()
    nb_spp_classes = labelencoder_nb_spp_classes.fit_transform(nb_spp_classes)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    nb_spp_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    nb_spp_atributos_train, nb_spp_atributos_test, nb_spp_classes_train, nb_spp_classes_test = train_test_split(nb_spp_atributos, nb_spp_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = GaussianNB()
    classificador.fit(nb_spp_atributos_train, nb_spp_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    nb_spp_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    nb_spp_previsoes = classificador.predict(nb_spp_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    nb_spp_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    nb_spp_acuracia.append(accuracy_score(nb_spp_classes_test, nb_spp_previsoes))
    
#medias
nb_spp_mdTimePP = np.mean(nb_spp_timePP)
nb_spp_mdTimeTrain = np.mean(nb_spp_timeTrain)
nb_spp_mdTimeTest = np.mean(nb_spp_timeTest)

nb_spp_sm_md_Time = nb_spp_mdTimePP + nb_spp_mdTimeTrain + nb_spp_mdTimeTest

nb_spp_mdAcuracia = np.mean(nb_spp_acuracia)

#desvio padrao
nb_spp_stdTimePP = np.std(nb_spp_timePP)
nb_spp_stdTimeTrain = np.std(nb_spp_timeTrain)
nb_spp_stdTimeTest = np.std(nb_spp_timeTest)

nb_spp_stdAcuracia = np.std(nb_spp_acuracia)