import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Aux_ import *

df=pd.read_csv("vélib_données.csv")
dfref=pd.read_csv("csvAux/vélib_données_ref.csv")

#couleurs 
bleufonce =  (15 /255, 208/255, 230/255)
bleuclair =  (130/255, 235/255, 247/255)
vertfonce =  (13 /255, 221/255, 29 /255)
vertclair =  (145/255, 249/255, 152/255)
rougefonce = (255/255, 0  /255, 0  /255)
rougeclair = (255/255, 128/255, 128/255)

mecarea = vertclair
mecapred = vertfonce
elecrea = bleuclair
elecpred = bleufonce


#==================================================================================#
#================================Places disponibles================================#
#==================================================================================#

#----------------------Partie prévision naïve par moyenne----------------------#  

#................AVEC JOUR................

def moyenne_ciblée_Timer_df(Timer,jour,station,dof,dt=10):
    ''' format Timer : "2022-06-02 00:56:16"
        format jour : int entre 0 et 6
        retourne la moyenne des places libres pour un station et un jour donnés, avec un delta t de 10min'''
    ''' utilise : RIEN
        est utilisé par : prediction_moyenne_ciblée_unique_df '''
    return moyenne(dof.loc[(abs(dof["heure en seconde"] - heurejournéeTimer(Timer)) < dt*60) & (dof["Station"]==station) & (dof["jour"]==jour)]["Nb bornes disponibles"])

def prediction_moyenne_ciblée_unique_df(station,jour):
    ''' format station : "Saint-Antoine Sévigné" 
        retourne un couple : la station et une liste : valeurs moyennes de places libres '''
    ''' utilise : moyenne_ciblée_Timer_df
        est utilisé par : affichage_prediction_moyenne_ciblée_unique_sansjour_df '''
    tdebut=time.time()
    dfunique = df.loc[ df["Station"] == station ] #df_unique(station,fichier)
    X,Y= [],[]
    for i in range(0,24*3600+1,5*60):
        X.append(i)
        Y.append(moyenne_ciblée_Timer_df(Seconde_format_Timer(i),jour,station,dfunique,10))
        #printProgressBar(i//300,(24*3600+1)//300, prefix = 'Progression :', suffix = 'Complete', length =50)
    dfunique=pd.DataFrame({})
    print(time.time()-tdebut) 
    return station,Y

def affichage_prediction_moyenne_ciblée_unique_df(station,jour,fichierref):
    '''station : nom de la station, 
    fichierref : fichier csv contenant les données de référence
    affiche la prédiction pour la station donnée'''
    '''utilise : prediction_moyenne_ciblée_unique_sansjour_df
        est utilisé par : RIEN '''
    dfref=pd.read_csv(fichierref)

    X,Y,Xref,Yref = prediction_moyenne_ciblée_unique_df(station,jour),dfref['heure en seconde'],dfref['Nb bornes disponibles']
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Yref,label="réalité")
    plt.plot(X_24h,Y,label="prédiction pour dt<10 min")
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.grid()
    plt.legend()
    plt.ylabel("Places dipsonibles")
    plt.xlabel("Heure de la journée")
    plt.title("Moyenne pour : "+station)
    plt.show()

#................SANS JOUR................

def moyenne_ciblée_Timer_sansjour_df(Timer,station,dof,dt):
    ''' format Timer : "2022-06-02 00:56:16"
        retourne la moyenne des places libres pour un station et un jour donnés, avec un delta t de 10min'''
    '''utilise : RIEN
        est utilisé par : prediction_moyenne_ciblée_unique_sansjour_df '''
    return moyenne(dof.loc[(abs(dof["heure en seconde"] - heurejournéeTimer(Timer)) < dt*60) & (dof["Station"]==station)]["Nb bornes disponibles"])

def prediction_moyenne_ciblée_unique_sansjour_df(station):
    ''' format station : "Saint-Antoine Sévigné" 
        retourne un couple : la station et une liste : valeurs moyennes de places libres '''
    '''utilise : moyenne_ciblée_Timer_sansjour_df
        est utilisé par : affichage_prediction_moyenne_ciblée_unique_sansjour_df '''
    tdebut=time.time()
    dfunique = df.loc[ df["Station"] == station ] #df_unique(station,fichier)
    X,Y= [],[]
    for i in range(0,24*3600+1,5*60):
        X.append(i)
        Y.append(moyenne_ciblée_Timer_sansjour_df(Seconde_format_Timer(i),station,dfunique,10))
        #printProgressBar(i//300,(24*3600+1)//300, prefix = 'Progression :', suffix = 'Complete', length =50)
    dfunique=pd.DataFrame({})
    print(time.time()-tdebut) 
    return station,Y

def affichage_prediction_moyenne_ciblée_unique_sansjour_df(station,fichierref):
    '''station : nom de la station, 
    fichierref : fichier csv contenant les données de référence
    affiche la prédiction pour la station donnée'''
    '''utilise : prediction_moyenne_ciblée_unique_sansjour_df
        est utilisé par : RIEN '''
    dfref=pd.read_csv(fichierref)

    X,Y,Xref,Yref = [i for i in range(0,86400,5*60)],prediction_moyenne_ciblée_unique_sansjour_df(station)[1],dfref['heure en seconde'],dfref['Nb bornes disponibles']
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Yref,label="réalité")
    plt.plot(X_24h,Y,label="prédiction pour dt<10 min")
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.grid()
    plt.legend()
    plt.ylabel("Places dipsonibles")
    plt.xlabel("Heure de la journée")
    plt.title("Moyenne pour : "+station)
    plt.show()

def affichage_moyenne_ciblée_stock(STATION,fichierref):
    res = np.load("resultats/résultat_moyenne_ciblée.npy",allow_pickle=True)
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref["Station"] == STATION ]
    for station in res :
        if station[0] == STATION :
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],station[1],dfrefstation['heure en seconde'][:-1],dfrefstation['Nb bornes disponibles'][:-1]
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour dt<10 min")
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne pour : "+STATION)
            plt.show()
            break


#----------------------Fin de la partie prévision naïve par moyenne----------------------#  

#----------------------Partie prévision par groupes/clusters----------------------#

def GridGeneration(n, fichier="csvAux/vélib_données_geo.csv", eps=0.0002):
    '''fichier est le csv utilisé, il doit contenir une colonne 'Coord' surement de la forme "[48.865983, 2.275725]"
    a est la taille de la grille 
    n le nombre de découpage en x et en y
    Retourne la liste de liste correspondant à la grille
    ATTENTION GRILLE EST UNE LISTE 3-D, UNE MATRICE DE LISTE'''
    '''
  Algorithm 1 Network Generation Method Based on a Grid 
    1: Input: Locations of bike stations {Si} for i=1 to n
    2: Output: G(E,V)
    3: Create a grid with the size of a, covering all bike stations,
       and a grid network G0(E0,V0) based a grid. Cells Ci in the grid are in V0, 
       and an edge e_ij between Ci and Cj is in E0 if Ci and Cj are adjacent to each other.
    '''
    df=pd.read_csv(fichier, converters={'Coord': pd.eval})
    X,Y= [] , []                            #X=[x1,x2,...] et Y=[y1,y2,...]
    Coord = df["Coord"].values              #df["Coord"].values est [[x,y],[x,y]]
    for pos in Coord:              
        X.append(pos[1])
        Y.append(pos[0])
    maxx,maxy,minx,miny=max(X)+eps,max(Y)+eps,min(X)-eps,min(Y)-eps
    largeur=maxy-miny
    longueur=maxx-minx
    ax,ay=longueur/n,largeur/n 
    M = np.array(df.values)
    grid=[]
    for k in range(n):
        grid.append([ [] for j in range(n)])

    for i in range(len(Coord)):
        grid[n - int((Coord[i][0] - miny)//ay +1)][int((Coord[i][1] - minx)//ax)].append(M[i,2])

    return grid

def stat_cluster(groupe,fichier='vélib_données.csv',dt=5):
    '''groupe : liste de stations définissant un cluster  
    fichier : fichier csv contenant les données
    retourne la liste des 288 taux (correspondant à 288 pas de 5 minutes)'''
    '''pour l'analyse statistique des cluster, on fera :
        pour chaque groupe calculer les sommes, en déduire 'un taux d'occupation' du groupe
        puis une moyenne des taux d'occupation pour chaque t
        et déduire une prédiction pour le groupe. '''
    ''' utilise : une liste groupe (np.load)
        est utilisé par : stat_cluster_entier'''
    tdebut=time.time()
    #df=pd.read_csv(fichier)
    dfu=pd.DataFrame({})
    Y= []
    
    Capa_max=0
    for station in groupe :
        dfu = pd.concat( [ dfu, df.loc[ (df["Code Station"] == station) ] ] )
        Capa_max += dfu.loc[ (dfu["Code Station"]==station) ]['Nombres de bornes en station'].values[0]
    for t in range(0,24*3600+1,dt*60):
        #X.append(t)
        #print('nombre de temps correspondant',len(L_temps_correspondants))
        somme_groupe = 0
        for station in groupe :
            try:
                ajout = dfu.loc[ (abs(dfu["heure en seconde"] - t) < dt*60) & (dfu["Code Station"] == station)]["Nb bornes disponibles"].values.mean()
                somme_groupe += ajout
                ancien=ajout
            except:
                somme_groupe += ancien
        Y.append(somme_groupe/Capa_max)
        #printProgressBar(t//300,(24*3600+1)//300, prefix = 'Progression :', suffix = 'Complete', length =50)
    print(time.time()-tdebut) 
    return groupe,Y

def stat_cluster_entier(grille,fichier='vélib_données.csv'):
    '''grille : liste de liste de liste de stations
    fichier : fichier csv contenant les données
    retourne la une liste des prévisions pour chaque cluster/groupe'''
    ''' utilise : stat_cluster
        est utilisé par : groupe_to_station'''
    Liste_y=[]
    for ligne in grille:
        for groupe in ligne:
            if len(groupe)!=0:
                Liste_y.append(stat_cluster(groupe,fichier))
    return Liste_y

def groupe_to_station(Y_groupe,station):
    '''Y_groupe : liste de taux d'occupation pour un groupe de stations
    station : station du groupe format : 'Code Station'
    Retourne une liste de taux d'occupation pour une station du groupe'''
    ''' utilise : stat_cluster
        est utilisé par : utilisationgrille'''
    Y=[]
    try:
        Capa=df.loc[ (df["Code Station"]==int(station)) ]['Nombres de bornes en station'].values[0]
    except:
        Capa=df.loc[ (df["Code Station"]==station) ]['Nombres de bornes en station'].values[0]
    for i in Y_groupe:
        Y.append(i*Capa)
    return Y

def utilisationgrille(grillenpy,n):
    '''grillenpy : fichier .npy contenant la grille
        de la forme : grille[i][j][k] : i : format de la grille, j : le groupe, k : le couple (groupe,liste prédiction)
                                    i:[0:3] pour les valeurs 24,44,64,99
                                    j:[0:nombre de groupe] le jieme groupe
                                    k:[0:1] 0 : liste des codes de station, 1 : liste de prédiction (taux d'occupation)
        n : le format de la grille. (24,44,64,99)
        retourne : une liste utilisable pour l'analyse, de la forme : resultats = [ [station,liste de la prévision], ]
        '''  
    ''' utilise : groupe_to_station
        est utilisé par : affichage_prediction_cluster'''   
    grilles=np.load("grilles/"+grillenpy,allow_pickle=True)
    res=[]
    dico={24:0,44:1,64:2,99:3}
    grille=grilles[dico[n]]
    for i in range(len(grille)):
        for station in grille[i][0] :
            res.append( (Code_Station(station) , groupe_to_station(grille[i][1],station) ))
    return res

def affichage_prediction_cluster(STATION,ngrille,fichierref='csvAux/vélib_données_ref.csv'):
    '''STATION : station au format 'Station'
    affiche la prédiction pour une station'''
    ''' utilise : utilisationgrille
        est utilisé par : affichage_analyse_places'''
    dfref=pd.read_csv(fichierref)
    dfrefstation=dfref.loc[ (dfref["Station"]==STATION) ]

    resultats=np.load('resultats/résultat_cluster_'+str(ngrille)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'][:-1],dfrefstation['Nb bornes disponibles'][:-1]
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(ngrille))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(ngrille)+" pour : "+STATION)
            plt.show()
            break

def analyse_places(resultats,fichierref='csvAux/vélib_données_ref.csv'):
    '''resultats : liste de couples : [ (station,liste de prévisions des places disponibles) ]
    retourne : la meilleure prévision, la pire prévision, la moyenne des prévisions, la variance des prévisions, la prévision médiane(graphique)
    la mesure est faite avec les carrés des différences'''
    ''' utilise : carrés_des_écarts, utilisationgrille
        est utilisé par : affichage_analyse_places'''
    dfref=pd.read_csv(fichierref)
    ret=[]
    for i in range(len(resultats)):
        listeref=dfref.loc[(dfref["Station"]==resultats[i][0])]["Nb bornes disponibles"].values
        if len(resultats[i][1])!=len(listeref):
            print("Erreur : ",resultats[i][0])
        else:
            ret.append( carrés_des_écarts(resultats[i][1][:-1],listeref)[1] )
    return ret

def affichage_analyse_places(resultats):
    '''resultats : liste de couples : (station,liste de prévisions des places disponibles)
    Affiche les prévisions pour les stations'''
    ''' utilise : analyse_places
        est utilisé par : RIEN IL EST FINAL'''
    resul=analyse_places(resultats)
    mini,maxi,moy,var,med=min(resul),max(resul),moyenne([i[0] for i in resul]),np.var([i[0] for i in resul]),np.median([i[0] for i in resul])
    print("Meilleure prévision : ",mini[0] ," pour la station ",mini[1])
    X=[t for t in range(0,24*3600+1,5*60)]
    Yref=dfref.loc[(dfref["Station"]==mini[1])]["Nb bornes disponibles"].values[:-1]
    plt.plot(X,mini[2],label='Meilleure prévision : '+mini[1] )
    plt.plot(X,Yref,label="Référence")
    plt.title("Prévisions avec la moyenne")
    plt.xlabel("Temps (s)")
    plt.ylabel("Nombre de vélos disponibles")
    plt.legend()
    plt.show()
    print("Pire prévision : ",maxi[0]," pour la station ",maxi[1])
    Yref=dfref.loc[(dfref["Station"]==maxi[1])]["Nb bornes disponibles"].values[:-1]
    plt.plot(X,Yref,label="Référence")
    plt.plot(X,maxi[2],label="Pire prévision : "+str(maxi[1]) )
    plt.title("Prévisions avec la moyenne")
    plt.xlabel("Temps (s)")
    plt.ylabel("Nombre de places disponibles")
    plt.legend()
    plt.show()
    print("Moyenne des prévisions : ",moy)
    print("Variance des prévisions : ",var)
    print("Prévision médiane : ",med)


#----------------------Fin de la partie prévision par groupes/clusters----------------------#

#----------------------Partie prévision par Random Forest----------------------#

def prediction_rf(STATION,fichierCible):
    '''STATION : nom de la station
    retourne : la liste des prédictions pour la station STATION
    '''
    ''' utilise : l'algo RF de la bibliothèque sklearn
        est utilisé par : affichage_prediction_rf'''
    tdebut = time.time()

    #df = pd.read_csv('vélib_données.csv')
    #dfref = pd.read_csv('csvAux/vélib_données_ref.csv')
    #dataref = dfref.loc[ dfref['Station']==STATION ]
    data = df.loc[ df['Station']==STATION ]

    features = ['heure en seconde','jour','mois','precip','feelslike','windspeed','conditions_int']
    target = 'Nb bornes disponibles'

    X_train = data[features]
    y_train = data[target]

    rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf_model.fit(X_train, y_train)

    dfcible = pd.read_csv(fichierCible)
    dfcibleStation = dfcible.loc[ dfcible['Station']==STATION ]

    y_pred = rf_model.predict(dfcibleStation[features])
    print("Temps d'exécution : ",time.time()-tdebut)
    return y_pred

def prediction_rf_sansMTO(STATION,fichierCible):
    '''STATION : nom de la station
    retourne : la liste des prédictions pour la station STATION
    '''
    ''' utilise : l'algo RF de la bibliothèque sklearn
        est utilisé par : affichage_prediction_rf'''
    tdebut = time.time()

    #df = pd.read_csv('vélib_données.csv')
    #dfref = pd.read_csv('csvAux/vélib_données_ref.csv')
    #dataref = dfref.loc[ dfref['Station']==STATION ]
    data = df.loc[ df['Station']==STATION ]

    features = ['heure en seconde','jour','mois']
    target = 'Nb bornes disponibles'

    X_train = data[features]
    y_train = data[target]

    rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf_model.fit(X_train, y_train)

    dfcible = pd.read_csv(fichierCible)
    dfcibleStation = dfcible.loc[ dfcible['Station']==STATION ]

    y_pred = rf_model.predict(dfcibleStation[features])
    print("Temps d'exécution : ",time.time()-tdebut)
    return y_pred

def affichage_prediction_rf(STATION,fichierCible,fichierref='csvAux/vélib_données_ref.csv'):
    '''STATION : nom de la station
    retourne : affiche le graphique de la prédiction
    '''
    ''' utilise : prediction_rf
        est utilisé par : RIEN IL EST FINAL'''
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref['Station']==STATION ]

    y_pred2 = prediction_rf(STATION,fichierCible)
    dfcible = pd.read_csv(fichierCible)
    dfciblestation = dfcible.loc[ dfcible['Station']==STATION ]

    X,Y,Xref,Yref = dfciblestation['heure en seconde'] , y_pred2, dfrefstation['heure en seconde'][:-1], dfrefstation['Nb bornes disponibles'][:-1]
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Yref,label="réalité")
    plt.plot(X_24h,Y,label="prédiction pour la station "+STATION)
    
    plt.title("Prévision avec Random Forest pour la station "+STATION)
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de places disponibles")
    plt.grid()
    plt.legend()
    plt.show()

def affichage_prediction_rf_compMTO(STATION,fichierCible,fichierref='csvAux/vélib_données_ref.csv'):
    '''STATION : nom de la station
    retourne : affiche le graphique de la prédiction
    '''
    ''' utilise : prediction_rf
        est utilisé par : RIEN IL EST FINAL'''
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref['Station']==STATION ]

    y_pred1 = prediction_rf(STATION,fichierCible)
    y_pred2 = prediction_rf_sansMTO(STATION,fichierCible)
    dfcible = pd.read_csv(fichierCible)

    X,Y1,Y2,Xref,Yref = dfcible['heure en seconde'] , y_pred1, y_pred2, dfrefstation['heure en seconde'][:-1], dfrefstation['Nb bornes disponibles'][:-1]
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Yref,label="réalité")
    plt.plot(X_24h,Y1,label="prédiction pour la station "+STATION+" avec météo")
    plt.plot(X_24h,Y2,label="prédiction pour la station "+STATION+" sans météo")
    
    plt.title("Prévision avec Random Forest pour la station "+STATION)
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de places disponibles")
    plt.grid()
    plt.legend()
    plt.show()

#----------------------Fin de la partie prévision par Random Forest----------------------#


#=================================================================================#
#================================Vélos disponibles================================#
#=================================================================================#

#----------------------Partie prévision naïve par moyenne----------------------#  

#................AVEC JOUR................

def moyenne_ciblée_Timer_df_velo(Timer,jour,station,dof,dt=10):
    ''' format Timer : "2022-06-02 00:56:16"
        format jour : int entre 0 et 6
        retourne la moyenne des vélos libres pour une station et un jour donnés, avec un delta t de 10min'''
    ''' utilise : RIEN
        est utilisé par : prediction_moyenne_ciblée_unique_df_velo '''
    return (moyenne(dof.loc[(abs(dof["heure en seconde"] - heurejournéeTimer(Timer)) < dt*60) & (dof["Station"]==station) & (dof["jour"]==jour)]["Nb vélo mécanique"]), 
            moyenne(dof.loc[(abs(dof["heure en seconde"] - heurejournéeTimer(Timer)) < dt*60) & (dof["Station"]==station) & (dof["jour"]==jour)]["Nb vélo électrique"]) )

def prediction_moyenne_ciblée_unique_df_velo(station,jour):
    ''' format station : "Saint-Antoine Sévigné" 
        retourne un couple : la station et une liste : valeurs moyennes de places libres '''
    ''' utilise : moyenne_ciblée_Timer_df_velo
        est utilisé par : affichage_prediction_moyenne_ciblée_unique_sansjour_df_velo '''
    tdebut=time.time()
    dfunique = df.loc[ df["Station"] == station ] #df_unique(station,fichier)
    X,Ym,Ye= [],[],[]
    for i in range(0,24*3600+1,5*60):
        X.append(i)
        Ym.append(moyenne_ciblée_Timer_df_velo(Seconde_format_Timer(i),jour,station,dfunique,10)[0])
        Ye.append(moyenne_ciblée_Timer_df_velo(Seconde_format_Timer(i),jour,station,dfunique,10)[1])
        #printProgressBar(i//300,(24*3600+1)//300, prefix = 'Progression :', suffix = 'Complete', length =50)
    dfunique=pd.DataFrame({})
    print(time.time()-tdebut) 
    return station,Ym,Ye

def affichage_prediction_moyenne_ciblée_unique_df_velo(station,jour,fichierref='csvAux/vélib_données_ref.csv'):
    '''station : nom de la station, 
    fichierref : fichier csv contenant les données de référence
    affiche la prédiction pour la station donnée'''
    '''utilise : prediction_moyenne_ciblée_unique_sansjour_df_velo
        est utilisé par : RIEN '''
    semaine = {0:"Lundi",1:"Mardi",2:"Mercredi",3:"Jeudi",4:"Vendredi",5:"Samedi",6:"Dimanche"}

    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref['Station']==station ]

    pred = prediction_moyenne_ciblée_unique_df_velo(station,jour)

    X, Ym,Ye, Xref, Ymref,Yeref = [i for i in range(0,86401,5*60)], pred[1],pred[2], dfrefstation['heure en seconde'][:-1], dfrefstation['Nb vélo mécanique'][:-1], dfrefstation['Nb vélo électrique'][:-1]
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Ymref,label="réalité : Vélos mécaniques",color=vertclair)
    plt.plot(Xref_24h,Yeref,label="réalite : Vélos électriques",color=bleuclair)
    plt.plot(X_24h,Ym,label="prédiction : Vélos mécaniques",color=vertfonce)
    plt.plot(X_24h,Ye,label="prédiction : Vélos électriques",color=bleufonce)
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.grid()
    plt.legend()
    plt.ylabel("vélos dipsonibles")
    plt.xlabel("Heure de la journée")
    plt.title("Moyenne pour : "+station+" un "+str(semaine[jour]) )
    plt.show()

#................SANS JOUR................

def moyenne_ciblée_Timer_sansjour_df_velo(Timer,station,dof,dt=10):
    ''' format Timer : "2022-06-02 00:56:16"
        format jour : int entre 0 et 6
        retourne la moyenne des vélos libres pour une station et un jour donnés, avec un delta t de 10min'''
    ''' utilise : RIEN
        est utilisé par : prediction_moyenne_ciblée_sansjour_unique_df_velo '''
    return (moyenne(dof.loc[(abs(dof["heure en seconde"] - heurejournéeTimer(Timer)) < dt*60) & (dof["Station"]==station) ]["Nb vélo mécanique"]), 
            moyenne(dof.loc[(abs(dof["heure en seconde"] - heurejournéeTimer(Timer)) < dt*60) & (dof["Station"]==station) ]["Nb vélo électrique"]) )

def prediction_moyenne_ciblée_sansjour_unique_df_velo(station):
    ''' format station : "Saint-Antoine Sévigné" 
        retourne un couple : la station et une liste : valeurs moyennes de places libres '''
    ''' utilise : moyenne_ciblée_Timer_sanjour_df_velo
        est utilisé par : affichage_prediction_moyenne_ciblée_sansjour_unique_sansjour_df_velo '''
    tdebut=time.time()
    dfunique = df.loc[ df["Station"] == station ] #df_unique(station,fichier)
    X,Ym,Ye= [],[],[]
    for i in range(0,24*3600+1,5*60):
        X.append(i)
        Ym.append(moyenne_ciblée_Timer_sansjour_df_velo(Seconde_format_Timer(i),station,dfunique,10)[0])
        Ye.append(moyenne_ciblée_Timer_sansjour_df_velo(Seconde_format_Timer(i),station,dfunique,10)[1])
        #printProgressBar(i//300,(24*3600+1)//300, prefix = 'Progression :', suffix = 'Complete', length =50)
    dfunique=pd.DataFrame({})
    print(time.time()-tdebut) 
    return station,Ym,Ye

def affichage_prediction_moyenne_ciblée_sansjour_unique_df_velo(station,fichierref='csvAux/vélib_données_ref.csv'):
    '''station : nom de la station, 
    fichierref : fichier csv contenant les données de référence
    affiche la prédiction pour la station donnée'''
    '''utilise : prediction_moyenne_ciblée_unique_sansjour_df_velo
        est utilisé par : RIEN '''
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref['Station']==station ]

    pred = prediction_moyenne_ciblée_sansjour_unique_df_velo(station)

    X, Ym,Ye, Xref, Ymref,Yeref = [i for i in range(0,86401,5*60)], pred[1],pred[2], dfrefstation['heure en seconde'][:-1], dfrefstation['Nb vélo mécanique'][:-1], dfrefstation['Nb vélo électrique'][:-1]
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Ymref,label="réalité : Vélos mécaniques",color=vertclair)
    plt.plot(Xref_24h,Yeref,label="réalite : Vélos électriques",color=bleuclair)
    plt.plot(X_24h,Ym,label="prédiction : Vélos mécaniques",color=vertfonce)
    plt.plot(X_24h,Ye,label="prédiction : Vélos électriques",color=bleufonce)
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.grid()
    plt.legend()
    plt.ylabel("vélos dipsonibles")
    plt.xlabel("Heure de la journée")
    plt.title("Moyenne pour : "+station )
    plt.show()

def affichage_moyenne_ciblée_stock_velo(STATION,fichierref='csvAux/vélib_données_ref.csv'):
    res = np.load("resultats/résultat_moyenne_ciblée_velo.npy",allow_pickle=True)
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref["Station"] == STATION ]
    for station in res :
        if station[0] == STATION :
            X, Ym,Ye, Xref, Ymref,Yeref = [i for i in range(0,86401,5*60)], station[1],station[2], dfrefstation['heure en seconde'][:-1], dfrefstation['Nb vélo mécanique'][:-1], dfrefstation['Nb vélo électrique'][:-1]
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Ymref,label="réalité : Vélos mécaniques",color=vertclair)
            plt.plot(Xref_24h,Yeref,label="réalite : Vélos électriques",color=bleuclair)
            plt.plot(X_24h,Ym,label="prédiction : Vélos mécaniques",color=vertfonce)
            plt.plot(X_24h,Ye,label="prédiction : Vélos électriques",color=bleufonce)
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("vélos dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne pour : "+STATION)
            plt.show()
            break

#----------------------Fin de la partie prévision naïve par moyenne----------------------#  

#----------------------Partie prévision par groupes/clusters----------------------#

def GridGeneration(n, fichier="csvAux/vélib_données_geo.csv", eps=0.0002):
    '''fichier est le csv utilisé, il doit contenir une colonne 'Coord' surement de la forme "[48.865983, 2.275725]"
    a est la taille de la grille 
    n le nombre de découpage en x et en y
    Retourne la liste de liste correspondant à la grille
    ATTENTION GRILLE EST UNE LISTE 3-D, UNE MATRICE DE LISTE'''
    '''
  Algorithm 1 Network Generation Method Based on a Grid 
    1: Input: Locations of bike stations {Si} for i=1 to n
    2: Output: G(E,V)
    3: Create a grid with the size of a, covering all bike stations,
       and a grid network G0(E0,V0) based a grid. Cells Ci in the grid are in V0, 
       and an edge e_ij between Ci and Cj is in E0 if Ci and Cj are adjacent to each other.
    '''
    df=pd.read_csv(fichier, converters={'Coord': pd.eval})
    X,Y= [] , []                            #X=[x1,x2,...] et Y=[y1,y2,...]
    Coord = df["Coord"].values              #df["Coord"].values est [[x,y],[x,y]]
    for pos in Coord:              
        X.append(pos[1])
        Y.append(pos[0])
    maxx,maxy,minx,miny=max(X)+eps,max(Y)+eps,min(X)-eps,min(Y)-eps
    largeur=maxy-miny
    longueur=maxx-minx
    ax,ay=longueur/n,largeur/n 
    M = np.array(df.values)
    grid=[]
    for k in range(n):
        grid.append([ [] for j in range(n)])

    for i in range(len(Coord)):
        grid[n - int((Coord[i][0] - miny)//ay +1)][int((Coord[i][1] - minx)//ax)].append(M[i,2])

    return grid

def stat_cluster_velo(groupe,fichier='vélib_données.csv',dt=5):
    '''groupe : liste de stations définissant un cluster  
    fichier : fichier csv contenant les données
    retourne la liste des 288 taux (correspondant à 288 pas de 5 minutes)'''
    '''pour l'analyse statistique des cluster, on fera :
        pour chaque groupe calculer les sommes, en déduire 'un taux d'occupation' du groupe
        puis une moyenne des taux d'occupation pour chaque t
        et déduire une prédiction pour le groupe. '''
    ''' utilise : une liste groupe (np.load)
        est utilisé par : stat_cluster_entier_velo'''
    tdebut=time.time()
    #df=pd.read_csv(fichier)
    dfu=pd.DataFrame({})
    Ymeca,Yelec = [],[]
    
    Capa_max=0
    for station in groupe :
        dfu = pd.concat( [ dfu, df.loc[ (df["Code Station"] == station) ] ] )
        Capa_max += dfu.loc[ (dfu["Code Station"]==station) ]['Nombres de bornes en station'].values[0]
        #'2022-10-23 18:30:02'
    for t in range(0,24*3600+1,dt*60):
        #X.append(t)
        #print('nombre de temps correspondant',len(L_temps_correspondants))
        somme_groupe_meca = 0
        somme_groupe_elec = 0
        for station in groupe :
            try:
                ajout_meca = dfu.loc[ (abs(dfu["heure en seconde"] - t) < dt*60) & (dfu["Code Station"] == station)]["Nb vélo mécanique"].values.mean()
                ajout_elec = dfu.loc[ (abs(dfu["heure en seconde"] - t) < dt*60) & (dfu["Code Station"] == station)]["Nb vélo électrique"].values.mean()
                somme_groupe_meca += ajout_meca
                somme_groupe_elec += ajout_elec
                ancien_meca=ajout_meca
                ancien_elec=ajout_elec
            except:
                somme_groupe_meca += ancien_meca
                somme_groupe_elec += ancien_elec
        Ymeca.append(somme_groupe_meca/Capa_max)
        Yelec.append(somme_groupe_elec/Capa_max)
        #printProgressBar(t//300,(24*3600+1)//300, prefix = 'Progression :', suffix = 'Complete', length =50)
    print(time.time()-tdebut) 
    return groupe,Ymeca,Yelec

def stat_cluster_entier_velo(grille,fichier='vélib_données.csv'):
    '''grille : liste de liste de liste de stations
    fichier : fichier csv contenant les données
    retourne la une liste des prévisions pour chaque cluster/groupe'''
    ''' utilise : stat_cluster_velo
        est utilisé par : groupe_to_station'''
    Liste_y=[]
    for ligne in grille:
        for groupe in ligne:
            if len(groupe)!=0:
                Liste_y.append(stat_cluster(groupe,fichier))
    return Liste_y

def groupe_to_station_velo(Y_groupe,station):
    '''Y_groupe : liste de taux d'occupation pour un groupe de stations
    station : station du groupe format : 'Code Station'
    Retourne une liste de taux d'occupation pour une station du groupe'''
    ''' utilise : stat_cluster
        est utilisé par : utilisationgrille'''
    Ymeca,Yelec=[],[]
    try:
        Capa=df.loc[ (df["Code Station"]==int(station)) ]['Nombres de bornes en station'].values[0]
    except:
        Capa=df.loc[ (df["Code Station"]==station) ]['Nombres de bornes en station'].values[0]
    for i in Y_groupe:
        Ymeca.append(i*Capa)
        Yelec.append(i*Capa)
    return Ymeca,Yelec

def utilisationgrille_velo(grillenpy,n):
    '''grillenpy : fichier .npy contenant la grille
        de la forme : grille[i][j][k] : i : format de la grille, j : le groupe, k : le couple (groupe,liste prédiction)
                                    i:[0:3] pour les valeurs 24,44,64,99
                                    j:[0:nombre de groupe] le jieme groupe
                                    k:[0:1] 0 : liste des codes de station, 1 : liste de prédiction (taux d'occupation)
        n : le format de la grille. (24,44,64,99)
        retourne : une liste utilisable pour l'analyse, de la forme : resultats = [ [station,(liste de la prévision MECA ,liste de la prévision ELEC) ], ]
        '''     
    grilles=np.load("grilles/"+grillenpy,allow_pickle=True)
    resMECA, resELEC =[],[]
    dico={24:0,44:1,64:2,99:3}
    grille=grilles[dico[n]]
    for i in range(len(grille)):
        for station in grille[i][0] :
            resMECA.append( (Code_Station(station) , groupe_to_station_velo(grille[i][1],station)[0] ))
            resELEC.append( (Code_Station(station) , groupe_to_station_velo(grille[i][1],station)[1] ))
    return resMECA,resELEC

def affichage_prediction_cluster_velo(STATION,ngrille,fichierref='csvAux/vélib_données_ref.csv'):
    '''STATION : station au format 'Station'
    affiche la prédiction pour une station'''
    ''' utilise : utilisationgrille_velo
        est utilisé par : affichage_analyse_places_velo'''
    dfref=pd.read_csv(fichierref)
    dfrefstation=dfref.loc[ (dfref["Station"]==STATION) ]

    resultats=np.load('resultats/résultat_cluster_'+str(ngrille)+'_velo.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:
            X, Ym,Ye, Xref, Ymref,Yeref = [i for i in range(0,86401,5*60)], resultats[i][1],resultats[i][2], dfrefstation['heure en seconde'][:-1], dfrefstation['Nb vélo mécanique'][:-1], dfrefstation['Nb vélo électrique'][:-1]
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Ymref,label="réalité : Vélos mécaniques",color=vertclair)
            plt.plot(Xref_24h,Yeref,label="réalite : Vélos électriques",color=bleuclair)
            plt.plot(X_24h,Ym,label="prédiction : Vélos mécaniques",color=vertfonce)
            plt.plot(X_24h,Ye,label="prédiction : Vélos électriques",color=bleufonce)
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Vélos dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(ngrille)+" pour : "+STATION)
            plt.show()
            break

#----------------------Fin de la partie prévision par groupes/clusters----------------------#

#----------------------Partie prévision par Random Forest----------------------#

def prediction_rf_velo(STATION,fichierCible):
    '''STATION : nom de la station
    retourne : la liste des prédictions pour la station STATION
    '''
    ''' utilise : l'algo RF de la bibliothèque sklearn
        est utilisé par : affichage_prediction_rf_velo'''
    tdebut = time.time()

    #df = pd.read_csv('vélib_données.csv')
    data = df.loc[ df['Station']==STATION ]

    features = ['heure en seconde','jour','mois','precip','feelslike','windspeed','conditions_int']
    target1 = "Nb vélo mécanique"
    target2 = "Nb vélo électrique"

    X_train = data[features]
    y1_train = data[target1]
    y2_train = data[target2]

    rf1_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf1_model.fit(X_train, y1_train)

    rf2_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf2_model.fit(X_train, y2_train)

    dfcible = pd.read_csv(fichierCible)
    dfcibleStation = dfcible.loc[ dfcible['Station']==STATION ]

    y1_pred = rf1_model.predict(dfcibleStation[features])
    y2_pred = rf2_model.predict(dfcibleStation[features])
    print("Temps d'exécution : ",time.time()-tdebut)
    return y1_pred, y2_pred

def affichage_prediction_rf_velo(STATION,fichierCible,fichierref='csvAux/vélib_données_ref.csv'):
    '''STATION : nom de la station
    retourne : affiche le graphique de la prédiction
    '''
    ''' utilise : prediction_rf_velo
        est utilisé par : RIEN IL EST FINAL'''
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref['Station']==STATION ]

    y1_pred = prediction_rf_velo(STATION,fichierCible)[0]
    y2_pred = prediction_rf_velo(STATION,fichierCible)[1]
    dfcible = pd.read_csv(fichierCible)
    dfcibleStation = dfcible.loc[ dfcible['Station']==STATION ]

    X,Y1,Y2,Xref,Y1ref,Y2ref =  dfcibleStation['heure en seconde'] , y1_pred,y2_pred, dfrefstation['heure en seconde'], dfrefstation["Nb vélo mécanique"], dfrefstation['Nb vélo électrique']
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(X_24h,Y1,label="prédiction vélo mécaniques",color=vertfonce)
    plt.plot(X_24h,Y2,label="prédiction vélos électriques",color=bleufonce)
    plt.plot(Xref_24h,Y1ref,label="réalité vélos mécaniques",color=vertclair)
    plt.plot(Xref_24h,Y2ref,label="réalité vélos électriques",color=bleuclair)
    plt.title("Prévision avec Random Forest pour la station "+STATION)
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de vélos disponibles")
    plt.grid()
    plt.legend()
    plt.show()

#----------------------Fin de la partie prévision par Random Forest----------------------#

#Fonctions d'affichage

def affichage_tousrésultats(STATION,fichierref='vélib_données_refnew.csv'):
    '''STATION : nom de la station
    retourne : affiche les graphiques de la prédiction pour la station STATION'''
    ''' utilise : 
        est utilisé par : RIEN IL EST FINAL'''
    
    #moyenne ciblée sans jour
    res = np.load("resltats/résultat_moyenne_ciblée.npy",allow_pickle=True)
    dfref=pd.read_csv(fichierref)

    dfrefstation = dfref.loc[ dfref["Station"] == STATION ]

    for station in res :
        if station[0] == STATION :

            plt.subplot(2,3,1)
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],station[1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour dt<10 min")
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne pour : "+STATION)
            #plt.show()
            break
    
    #clusters

    resultats=np.load('resultats/résultat_cluster_24.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:
            plt.subplot(2,3,2)
            
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(24))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(24)+" pour : "+STATION)
            #plt.show()
            break
    
    resultats=np.load('resultats/résultat_cluster_'+str(44)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:
            plt.subplot(2,3,3)

            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(44))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(44)+" pour : "+STATION)
            #plt.show()
            break
    
    resultats=np.load('resultats/résultat_cluster_'+str(64)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:

            plt.subplot(2,3,4)
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(64))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(64)+" pour : "+STATION)
            #plt.show()
            break
    
    resultats=np.load('resultats/résultat_cluster_'+str(99)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:

            plt.subplot(2,3,5)
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(99))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(99)+" pour : "+STATION)
            #plt.show()
            break

    #random forest

    fichierCible = fichierref

    y_pred2 = prediction_rf(STATION,fichierCible)
    dfcible = pd.read_csv(fichierCible)

    X,Y,Xref,Yref = dfcible['heure en seconde'] , y_pred2, dfrefstation['heure en seconde'], dfrefstation['Nb bornes disponibles']
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.subplot(2,3,6)
    plt.plot(Xref_24h,Yref,label="réalité")
    plt.plot(X_24h,Y,label="prédiction pour la station "+STATION)
    plt.title("Prévision avec Random Forest pour la station "+STATION)
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de places disponibles")
    plt.grid()
    plt.legend()
    plt.show()


def affichage_cluster(STATION,fichierref):
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref["Station"] == STATION ]

    #clusters

    resultats=np.load('resultats/résultat_cluster_24.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:
            plt.subplot(2,2,1)
            
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(24))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(24)+" pour : "+STATION)
            #plt.show()
            break
    
    resultats=np.load('resultats/résultat_cluster_'+str(44)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:
            plt.subplot(2,2,2)

            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(44))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(44)+" pour : "+STATION)
            #plt.show()
            break
    
    resultats=np.load('resultats/résultat_cluster_'+str(64)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:

            plt.subplot(2,2,3)
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(64))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(64)+" pour : "+STATION)
            #plt.show()
            break
    
    resultats=np.load('resultats/résultat_cluster_'+str(99)+'.npy',allow_pickle=True)

    for i in range(len(resultats)):
        if resultats[i][0]==STATION:

            plt.subplot(2,2,4)
            X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],resultats[i][1],dfrefstation['heure en seconde'],dfrefstation['Nb bornes disponibles']
            X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

            plt.plot(Xref_24h,Yref,label="réalité")
            plt.plot(X_24h,Y,label="prédiction pour n="+str(99))
            plt.scatter([0],[0],s=0.1)
            plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
            plt.grid()
            plt.legend()
            plt.ylabel("Places dipsonibles")
            plt.xlabel("Heure dans la journée")
            plt.title("Moyenne avec cluster"+str(99)+" pour : "+STATION)
            #plt.show()
            break
    plt.show()


def affichagePPT(STATION):
    affichage_moyenne_ciblée_stock('Charonne - Bureau', 'vélib_données_16mai.csv')
    affichage_moyenne_ciblée_stock_velo('Charonne - Bureau', 'vélib_données_16mai.csv')
    #affichage_prediction_cluster_velo('Charonne - Bureau',44, 'vélib_données_16mai.csv')
    affichage_cluster('Charonne - Bureau', 'vélib_données_16mai.csv')
    affichage_prediction_rf('Charonne - Bureau', 'vélib_données_16mai.csv','vélib_données_16mai.csv')
    affichage_prediction_rf_velo('Charonne - Bureau', 'vélib_données_16mai.csv','vélib_données_16mai.csv')
