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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#numero de iteracoes para treino e teste
count_test = 5 

#carregando a base de dados NSL KDD
nsl_kdd_data = pd.read_csv('./KDDTrain+.csv')

#Arvore de Decisao 

#Padronizao + OneHot
ad_poh_timePP = []
ad_poh_timeTrain = []
ad_poh_timeTest = []

ad_poh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    ad_poh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    ad_poh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_ad_poh_atributos = LabelEncoder()
    ad_poh_atributos[:, 1] = labelencoder_ad_poh_atributos.fit_transform(ad_poh_atributos[:, 1])
    ad_poh_atributos[:, 2] = labelencoder_ad_poh_atributos.fit_transform(ad_poh_atributos[:, 2])
    ad_poh_atributos[:, 3] = labelencoder_ad_poh_atributos.fit_transform(ad_poh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_ad_poh_classes= LabelEncoder()
    ad_poh_classes = labelencoder_ad_poh_classes.fit_transform(ad_poh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    ad_poh_atributos = onehotencoder.fit_transform(ad_poh_atributos).toarray()
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    ad_poh_atributos = scaler.fit_transform(ad_poh_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    ad_poh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    ad_poh_atributos_train, ad_poh_atributos_test, ad_poh_classes_train, ad_poh_classes_test = train_test_split(ad_poh_atributos, ad_poh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = DecisionTreeClassifier(criterion='entropy')
    classificador.fit(ad_poh_atributos_train, ad_poh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    ad_poh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    ad_poh_previsoes = classificador.predict(ad_poh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    ad_poh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    ad_poh_acuracia.append(accuracy_score(ad_poh_classes_test, ad_poh_previsoes))
    
#medias
ad_poh_mdTimePP = np.mean(ad_poh_timePP)
ad_poh_mdTimeTrain = np.mean(ad_poh_timeTrain)
ad_poh_mdTimeTest = np.mean(ad_poh_timeTest)

ad_poh_sm_md_Time = ad_poh_mdTimePP + ad_poh_mdTimeTrain + ad_poh_mdTimeTest

ad_poh_mdAcuracia = np.mean(ad_poh_acuracia)

#desvio padrao
ad_poh_stdTimePP = np.std(ad_poh_timePP)
ad_poh_stdTimeTrain = np.std(ad_poh_timeTrain)
ad_poh_stdTimeTest = np.std(ad_poh_timeTest)

ad_poh_stdAcuracia = np.std(ad_poh_acuracia)






#Padronizao
ad_p_timePP = []
ad_p_timeTrain = []
ad_p_timeTest = []

ad_p_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    ad_p_atributos = nsl_kdd_data.iloc[:, 0:41].values
    ad_p_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_ad_p_atributos = LabelEncoder()
    ad_p_atributos[:, 1] = labelencoder_ad_p_atributos.fit_transform(ad_p_atributos[:, 1])
    ad_p_atributos[:, 2] = labelencoder_ad_p_atributos.fit_transform(ad_p_atributos[:, 2])
    ad_p_atributos[:, 3] = labelencoder_ad_p_atributos.fit_transform(ad_p_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_ad_p_classes= LabelEncoder()
    ad_p_classes = labelencoder_ad_p_classes.fit_transform(ad_p_classes)
    
    #fazendo a padronizacao dos dados
    scaler = StandardScaler()
    ad_p_atributos = scaler.fit_transform(ad_p_atributos)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    ad_p_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    ad_p_atributos_train, ad_p_atributos_test, ad_p_classes_train, ad_p_classes_test = train_test_split(ad_p_atributos, ad_p_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = DecisionTreeClassifier(criterion='entropy')
    classificador.fit(ad_p_atributos_train, ad_p_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    ad_p_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    ad_p_previsoes = classificador.predict(ad_p_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    ad_p_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    ad_p_acuracia.append(accuracy_score(ad_p_classes_test, ad_p_previsoes))
    
#medias
ad_p_mdTimePP = np.mean(ad_p_timePP)
ad_p_mdTimeTrain = np.mean(ad_p_timeTrain)
ad_p_mdTimeTest = np.mean(ad_p_timeTest)

ad_p_mdAcuracia = np.mean(ad_p_acuracia)

ad_p_sm_md_Time = ad_p_mdTimePP + ad_p_mdTimeTrain + ad_p_mdTimeTest

#desvio padrao
ad_p_stdTimePP = np.std(ad_p_timePP)
ad_p_stdTimeTrain = np.std(ad_p_timeTrain)
ad_p_stdTimeTest = np.std(ad_p_timeTest)

ad_p_stdAcuracia = np.std(ad_p_acuracia)






#OneHot
ad_oh_timePP = []
ad_oh_timeTrain = []
ad_oh_timeTest = []

ad_oh_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    ad_oh_atributos = nsl_kdd_data.iloc[:, 0:41].values
    ad_oh_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_ad_oh_atributos = LabelEncoder()
    ad_oh_atributos[:, 1] = labelencoder_ad_oh_atributos.fit_transform(ad_oh_atributos[:, 1])
    ad_oh_atributos[:, 2] = labelencoder_ad_oh_atributos.fit_transform(ad_oh_atributos[:, 2])
    ad_oh_atributos[:, 3] = labelencoder_ad_oh_atributos.fit_transform(ad_oh_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_ad_oh_classes= LabelEncoder()
    ad_oh_classes = labelencoder_ad_oh_classes.fit_transform(ad_oh_classes)
    
    #fazendo o One-hot
    onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
    ad_oh_atributos = onehotencoder.fit_transform(ad_oh_atributos).toarray()
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    ad_oh_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    ad_oh_atributos_train, ad_oh_atributos_test, ad_oh_classes_train, ad_oh_classes_test = train_test_split(ad_oh_atributos, ad_oh_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = DecisionTreeClassifier(criterion='entropy')
    classificador.fit(ad_oh_atributos_train, ad_oh_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    ad_oh_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    ad_oh_previsoes = classificador.predict(ad_oh_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    ad_oh_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    ad_oh_acuracia.append(accuracy_score(ad_oh_classes_test, ad_oh_previsoes))
    
#medias
ad_oh_mdTimePP = np.mean(ad_oh_timePP)
ad_oh_mdTimeTrain = np.mean(ad_oh_timeTrain)
ad_oh_mdTimeTest = np.mean(ad_oh_timeTest)

ad_oh_sm_md_Time = ad_oh_mdTimePP + ad_oh_mdTimeTrain + ad_oh_mdTimeTest

ad_oh_mdAcuracia = np.mean(ad_oh_acuracia)

#desvio padrao
ad_oh_stdTimePP = np.std(ad_oh_timePP)
ad_oh_stdTimeTrain = np.std(ad_oh_timeTrain)
ad_oh_stdTimeTest = np.std(ad_oh_timeTest)

ad_oh_stdAcuracia = np.std(ad_oh_acuracia)






#Sem pradronizar e sem onehot
ad_spp_timePP = []
ad_spp_timeTrain = []
ad_spp_timeTest = []

ad_spp_acuracia = []

for n in range(count_test):
    
    #dividindo entre atributos e classes
    ad_spp_atributos = nsl_kdd_data.iloc[:, 0:41].values
    ad_spp_classes = nsl_kdd_data.iloc[:, 41].values
    
    #inicio do calculo do tempo do pre processamento
    begin = time.time()
    
    #transformando os atributos com cadeias de caracteres em numeros
    labelencoder_ad_spp_atributos = LabelEncoder()
    ad_spp_atributos[:, 1] = labelencoder_ad_spp_atributos.fit_transform(ad_spp_atributos[:, 1])
    ad_spp_atributos[:, 2] = labelencoder_ad_spp_atributos.fit_transform(ad_spp_atributos[:, 2])
    ad_spp_atributos[:, 3] = labelencoder_ad_spp_atributos.fit_transform(ad_spp_atributos[:, 3])
    
    #transformando os classes que são cadeias de caracteres em numeros
    labelencoder_ad_spp_classes= LabelEncoder()
    ad_spp_classes = labelencoder_ad_spp_classes.fit_transform(ad_spp_classes)
    
    #fim do calculo do tempo preprocessamento
    end = time.time()
    ad_spp_timePP.append(end - begin)
    
    #inicio do calculo do tempo treino
    begin = time.time()
    
    #divide em treino e teste
    ad_spp_atributos_train, ad_spp_atributos_test, ad_spp_classes_train, ad_spp_classes_test = train_test_split(ad_spp_atributos, ad_spp_classes, test_size=0.15)
    
    #treino do algoritmo com a base de treino
    classificador = DecisionTreeClassifier(criterion='entropy')
    classificador.fit(ad_spp_atributos_train, ad_spp_classes_train)
    
    #fim do calculo do tempo
    end = time.time()
    ad_spp_timeTrain.append(end - begin)
    
    #inicio do calculo do tempo teste
    begin = time.time()
    
    #classificando as amostras de teste
    ad_spp_previsoes = classificador.predict(ad_spp_atributos_test)
    
    #fim do calculo do tempo
    end = time.time()
    ad_spp_timeTest.append(end - begin)

    #calculo da acuracia do algorimo
    ad_spp_acuracia.append(accuracy_score(ad_spp_classes_test, ad_spp_previsoes))
    
#medias
ad_spp_mdTimePP = np.mean(ad_spp_timePP)
ad_spp_mdTimeTrain = np.mean(ad_spp_timeTrain)
ad_spp_mdTimeTest = np.mean(ad_spp_timeTest)

ad_spp_sm_md_Time = ad_spp_mdTimePP + ad_spp_mdTimeTrain + ad_spp_mdTimeTest

ad_spp_mdAcuracia = np.mean(ad_spp_acuracia)

#desvio padrao
ad_spp_stdTimePP = np.std(ad_spp_timePP)
ad_spp_stdTimeTrain = np.std(ad_spp_timeTrain)
ad_spp_stdTimeTest = np.std(ad_spp_timeTest)

ad_spp_stdAcuracia = np.std(ad_spp_acuracia)





