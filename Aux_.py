import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def listeStation(fichier):
    df=pd.read_csv(fichier)
    res = []
    for station in df["Station"]:
        if not(station in res):
            res.append(station)
    return res

def carrés_des_écarts(liste1,liste2):
    ''' les listes sont supposées de longueurs égales
    retourne la liste des carrés des écarts, ainsi que la somme des termes de la liste'''
    res=[]
    for i in range(min(len(liste1),len(liste2))):
        res.append((liste1[i]-liste2[i])**2)
    return res,sum(res)

def moyenne(liste):
    '''list of int -> float
    retourne la moyenne des valeurs d'un liste'''
    return sum(liste)/len(liste)

Code_station = np.load("npyAuxiliaires/code_station.npy",allow_pickle=True).item()
Station_code = np.load("npyAuxiliaires/station_code.npy",allow_pickle=True).item()

def Code_Station(code):
    return Code_station[code]
def Station_Code(station):
    return Station_code[station]

def retrouver_groupe(grille,station):
    '''grille : liste de la forme [ [[codes],[Y]] , [[codes],[Y]], ... ] (Y : liste des 288 points)
    station : nom de la station voulue de la forme "Saint-Antoine - Sévigné" 
    retourne :  l'indice du groupe dans lequel est contenue la station
                la liste des stations du groupe en question
                la liste Y du groupe'''
    code = Code_station(station)
    for i in range(len(grille)):
        groupe = grille[i]
        if code in groupe[0]:
            return i,groupe[0],groupe[1]
        
def Seconde_format_Timer(n):
    '''retourne l'heure en format Timer "2022-06-02 00:56:16" des secondes depuis 00:00'''
    h,m,s=(n//3600)%24,(n//60)%60,n%60
    res="2022-00-00 "
    if h==0 :
        res= res + "00:"
    elif h<10:
        res=res+"0"+str(h)+":"
    elif h>=10:
        res=res+str(h)+":"
    if m==0 :
        res= res + "00:"
    elif m<10:
        res=res+"0"+str(m)+":"
    elif m>=10:
        res=res+str(m)+":"
    if s==0 :
        res= res + "00"
    elif s<10:
        res=res+"0"+str(s)
    elif s>=10:
        res=res+str(s)
    return res

def heurejournéeTimer(Timer):
    '''str -> float
    Date de format : "2022-06-02 18:00:00"
    retourne l'heure en secondes
    '''
    dheures = int(Timer[11:13])
    dminutes= int(Timer[14:16])
    dsecondes=int(Timer[17:19])
    return dheures*3600+dminutes*60+dsecondes

def affichage_placesdispo(STATION,ListePoints,fichierref):
    dfref=pd.read_csv(fichierref)
    dfrefstation = dfref.loc[ dfref["Station"] == STATION ]

    X,Y,Xref,Yref = [i for i in range(0,86401,5*60)],ListePoints,dfrefstation['heure en seconde'][:-1],dfrefstation['Nb bornes disponibles'][:-1]
    X_24h,Xref_24h=[t/3600 for t in X],[t/3600 for t in Xref]

    plt.plot(Xref_24h,Yref,label="réalité")
    plt.plot(X_24h,Y,label="prédiction")
    plt.scatter([0],[0],s=0.1)
    plt.gca().xaxis.set_ticks(range(0,int(max(X_24h))+1,6))
    plt.grid()
    plt.legend()
    plt.ylabel("Places dipsonibles")
    plt.xlabel("Heure dans la journée")
    plt.title("résultats pour : "+STATION)
    plt.show()