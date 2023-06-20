import multiprocessing as mp
import time
import numpy as np
from Main import *

def mini_ecarts_mp(liste):
    mini,station_mini = 70617.9572416587,"Bercy."
    for station in liste:
        if station[0] >= 0.1:
            if station[0]<mini:
                mini = station[0]
                station_mini = station[1]
    return mini,station_mini

def maxi_ecarts_mp(liste):
    maxi,station_maxi = 70617.9572416587,"Bercy."
    for station in liste:
        if station[0]>maxi:
            maxi = station[0]
            station_maxi = station[1]
    return maxi,station_maxi

def changement(liste):
    '''argument : list of 2-tuple
    return 2-tuple of list'''
    res1,res2=[],[]
    for tup in liste:
        res1.append(tup[0])
        res2.append(tup[1])
    return res1,res2

def moyenne_horsnuls(liste):
    res=[]
    for i in liste:
        if i >= 0.1:
            res.append(i)
    return moyenne(res)

def liste_nuls(tuple_liste):
    res=[]
    for i in range(len(tuple_liste[0])):
        if tuple_liste[0][i] <= 0.1:
            res.append((tuple_liste[0][i],tuple_liste[1][i]))
    return res

"""if __name__=='__main__':
    tdmp=time.time()
    p=mp.Pool(6)
    rl=p.starmap(ecarts_moyenne_ciblée_sansjour_df,[(station,fichier,"csvAux/vélib_données_ref.csv") for station in listeStation])
    p.close()
    p.join()
    stati_mini = mini_ecarts_mp(rl)
    stati_maxi = maxi_ecarts_mp(rl)
    print("temps total du calcul : ",time.time()-tdmp)
    print("station qui a l'écart minimum : ",stati_mini)
    print("station qui a l'écart maximum : ",stati_maxi)
    print("moyenne hors nuls = ",moyenne_horsnuls(changement(rl)[0])) 
    print("listes des stations qui semblent constantes : ", liste_nuls(changement(rl)))
    affichage_prediction_moyenne_ciblée_unique_sansjour_df(stati_mini[1],fichier,fichierref="csvAux/vélib_données_ref.csv")
    affichage_prediction_moyenne_ciblée_unique_sansjour_df(stati_maxi[1],fichier,fichierref="csvAux/vélib_données_ref.csv")"""

"""if __name__=='__main__':
    tdmp=time.time()
    p=mp.Pool(4)
    rl=p.starmap(stat_cluster_entier_velo ,[([grille]) for grille in listegrille])
    p.close()
    p.join()
    np.save("stat_cluster_entier_toutegrille_velo.npy",rl)"""

"""if __name__=='__main__':
    tdmp=time.time()
    p=mp.Pool(6)
    rl=p.starmap( prediction_moyenne_ciblée_sansjour_unique_df_velo ,[([station]) for station in listeStation])
    p.close()
    p.join()
    np.save("résultat_moyenne_ciblée_velo.npy",rl)"""
"""
if __name__=='__main__':
    '''fakepool2 = []
    for station in listeStation:
        fakepool2.append(prediction_moyenne_ciblée_sansjour_unique_df_velo(station))
        printProgressBar(listeStation.index(station),len(listeStation))
    np.save("résultat_moyenne_ciblée_velo.npy",fakepool2)'''

    fakepool = []
    for grille in listegrille:
        fakepool.append(stat_cluster_entier_velo(grille))
        print("grille terminée")
    np.save("stat_cluster_entier_toutegrille_velo.npy",fakepool)
"""

def try_rf(station,fichier):
    try:
        return (station,prediction_rf(station,fichier))
    except:
        pass

if __name__ == '__main__':
    tdmp=time.time()
    p=mp.Pool(3)
    lstation = listeStation('vélib_données_16mai.csv')
    rl=p.starmap( try_rf ,[([station,'vélib_données_16mai.csv']) for station in lstation])
    p.close()
    p.join()
    np.save("résultat_rf_2.npy",rl)
    print("temps total du calcul : ",time.time()-tdmp)