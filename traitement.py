from os import remove
import pandas as pd
import numpy as np
import datetime

def remove_specific_row_from_csv(file, column_name, *args):
    '''
    trouvé sur https://stackoverflow.com/questions/29725932/deleting-rows-with-python-in-a-csv-file
    :param file: file to remove the rows from
    :param column_name: The column that determines which row will be 
           deleted (e.g. if Column == Name and row-*args
           contains "Gavri", All rows that contain this word will be deleted)
    :param args: Strings from the rows according to the conditions with 
                 the column
    '''
    row_to_remove = []
    for row_name in args:
        row_to_remove.append(row_name)
    try:
        df = pd.read_csv(file)
        for row in row_to_remove:
            df = df[eval("df.{}".format(column_name)) != row]
        df.to_csv(file, index=False)
    except Exception  as e:
        raise Exception("Error message....")

def heurejournéeTimer(Timer):
    '''str -> float
    Date de format : "2022-06-02 18:00:00"
    retourne l'heure en secondes
    '''
    dheures = int(Timer[11:13])
    dminutes= int(Timer[14:16])
    dsecondes=int(Timer[17:19])
    return dheures*3600+dminutes*60+dsecondes

def UsableTimer(Date): #format "2022-06-09T16:56:19+00:00"
    """str -> int*str*str
    on va tout passer en secondes,
    et prendre comme origine : 2022-10-01 00:00:00 c'est un samedi
    retourne le remps en sec depuis l'origine, le jour de la semaine, le mois
    return [secondes,"lun","jan"] """
    origine="2022-10-01 00:00:00"
    semaine=["lun","mar","mer","jeu","ven","sam","dim"]
    année=["jan","fev","mar","avr","mai","jun","jul","aou","sep","oct","nov","dec"]
    joursmois=[31,28,31,30,31,30,31,31,30,31,30,31]
    dannées= int(Date[0:4])-int(origine[0:4])
    dmois= (int(Date[5:7])-int(origine[5:7]))%12
    djours = (int(Date[8:10])-int(origine[8:10]))%joursmois[int(Date[5:7])-2]
    passagejoursmois=0
    for i in range(1,dmois+1):
        passagejoursmois += joursmois[int(Date[5:7])-(i+1)]
    joursemaine=semaine[(djours+5+ passagejoursmois +dannées*365)%7]
    moisannée=année[int(Date[5:7])-1]
    dheures = int(Date[11:13])-int(origine[11:13])
    dminutes= int(Date[14:16])-int(origine[14:16])
    dsecondes=int(Date[17:19])-int(origine[17:19])
    return [dsecondes + dminutes*60 + dheures*3600 + djours*24*3600 + passagejoursmois*24*3600 + dannées*365*24*3600,joursemaine,moisannée]

def UsableTimer_secondes(Timer):
    return UsableTimer(Timer)[0]

def UsableTimer_jour(date_string):
    date = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return date.strftime("%A"), date.weekday()

def UsableTimer_mois(date_string):
    date = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return date.strftime("%B"),date.month

def ajout_heure_jour(fichier):
    dof=pd.read_csv(fichier)
    dof["heure en seconde"]=dof["Timer"].apply(heurejournéeTimer)
    dof["secondes"]=dof["Timer"].apply(UsableTimer_secondes)
    dof["jour"]=dof["Timer"].apply(UsableTimer_jour)
    dof["mois"]=dof["Timer"].apply(UsableTimer_mois)
    dof.to_csv(fichier,mode='w',index=False,header=True)

def ajout_groupe(fichier):
    grille24 = np.load("grilles/grille24.npy",allow_pickle=True)
    grille44 = np.load("grilles/grille44.npy",allow_pickle=True)
    grille64 = np.load("grilles/grille64.npy",allow_pickle=True)
    grille99 = np.load("grilles/grille99.npy",allow_pickle=True)
    #grilles = [grille24,grille44,grille64,grille99]
    dico24,dico44,dico64,dico99=dict(),dict(),dict(),dict()
    for i in range(len(grille24)):
        for j in range(len(grille24[0])):
            for codestation in grille24[i][j]:
                dico24[codestation]=i*len(grille24[0])+j
    for i in range(len(grille44)):
        for j in range(len(grille44[0])):
            for codestation in grille44[i][j]:
                dico44[codestation]=i*len(grille44[0])+j
    for i in range(len(grille64)):
        for j in range(len(grille64[0])):
            for codestation in grille64[i][j]:
                dico64[codestation]=i*len(grille64[0])+j
    for i in range(len(grille99)):
        for j in range(len(grille99[0])):
            for codestation in grille99[i][j]:
                dico99[codestation]=i*len(grille99[0])+j
    print('fin de la création des dictionnaires')
    def determination_groupe(code,dico):
        return dico[str(code)]
    dof=pd.read_csv(fichier)
    dof["groupe24"]=dof["Code Station"].apply(determination_groupe,dico=dico24)
    dof["groupe44"]=dof["Code Station"].apply(determination_groupe,dico=dico44)
    dof["groupe64"]=dof["Code Station"].apply(determination_groupe,dico=dico64)
    dof["groupe99"]=dof["Code Station"].apply(determination_groupe,dico=dico99)
    dof.to_csv(fichier,mode='w',index=False,header=True)

def est_meme_date(dateVelib,dateMTO):
    """dateVelib format : '2022-10-23 18:25:51'
       dateMTO   format : '2022-10-23T00:00:00' 
       retourne un booleen qui traduit si les dates sont identiques"""
    return dateVelib[0:10] == dateMTO[0:10]

def est_meme_heure(dateVelib,dateMTO):
    """dateVelib format : '2022-10-23 18:25:51'
       dateMTO   format : '2022-10-23T00:00:00' 
       retourne un booleen qui traduit si les heures sont identiques"""
    return dateVelib[11:13] == dateMTO[11:13]

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

def ajout_Timer(fichier):
    dof = pd.read_csv(fichier)
    dof["Timer"] = dof["heure en seconde"].apply(Seconde_format_Timer)
    dof.to_csv(fichier,mode='w',index=False,header=True)

def ajout_meteo(fichier,fichierMTO):
    meteo=pd.read_csv(fichierMTO)
    def application_meteo_feelslike(dateVelib):
        dateutile = dateVelib[:10] + 'T' + dateVelib[11:13] + ':00:00'
        return meteo.loc[ meteo['datetime']==dateutile ]['feelslike'].values[0]
    def application_meteo_windspeed(dateVelib):
        dateutile = dateVelib[:10] + 'T' + dateVelib[11:13] + ':00:00'
        return meteo.loc[ meteo['datetime']==dateutile ]['windspeed'].values[0]
    def application_meteo_conditions(dateVelib):
        dateutile = dateVelib[:10] + 'T' + dateVelib[11:13] + ':00:00'
        return meteo.loc[ meteo['datetime']==dateutile ]['conditions'].values[0]
    def application_meteo_precip(dateVelib):
        dateutile = dateVelib[:10] + 'T' + dateVelib[11:13] + ':00:00'
        return meteo.loc[ meteo['datetime']==dateutile ]['precip'].values[0]
    dof=pd.read_csv(fichier)
    dof['feelslike']=dof["Timer"].apply(application_meteo_feelslike)
    dof['windspeed']=dof["Timer"].apply(application_meteo_windspeed)
    dof['conditions']=dof["Timer"].apply(application_meteo_conditions)
    dof['precip']=dof["Timer"].apply(application_meteo_precip)
    dof.to_csv(fichier,mode='w',index=False,header=True)

def ajout_dummy(fichier,colonne):
    df = pd.read_csv(fichier)
    dico = {}
    curseur = 1
    res = []
    for i in df[colonne].values :
        if i not in dico :
            dico[i] = curseur
            curseur += 1
        res.append(dico[i])
    df[f'{colonne}_int'] = res
    df.to_csv(fichier,mode='w',index=False,header=True)

def ajout_int_conditions(fichier):
    df = pd.read_csv(fichier)
    dico = {'Partially cloudy': 1, 'Clear': 2, 'Overcast': 3, 'Rain, Overcast': 4, 'Rain, Partially cloudy': 5, 'Rain': 6, 'Snow, Rain, Overcast': 7, 'Snow, Overcast': 8}
    def application_conditions(conditions):
        return dico[conditions]
    df["conditions_int"]=df["conditions"].apply(application_conditions)
    df.to_csv(fichier,mode='w',index=False,header=True)