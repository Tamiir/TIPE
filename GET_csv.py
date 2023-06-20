from threading import Timer
import requests
import pandas as pd
from time import localtime, strftime


def update():
#    if iteration <= limiteIter :
    getData()
    set_timer()
  
def set_timer():
    Timer(durationinsec, update).start()
  
def getData():
    global iteration
    nbrows = 1500
    url = "https://opendata.paris.fr/api/records/1.0/search/?dataset=velib-disponibilite-en-temps-reel%40parisdata&rows=" + str(nbrows) + "&facet=overflowactivation&facet=creditcard&facet=kioskstate&facet=station_state"
    mytime = strftime("%Y-%m-%d %H:%M:%S", localtime())
 
    resp = requests.get(url)
    if resp.status_code != 200:
        print(mytime, " - ", iteration, " - Erreur dans la récupération des données")
    else:
        data = resp.json()
        dff = pd.DataFrame(columns =['Timer','Station','Code Station','Coord','commune'])
        for rec in data['records']:
            dff.loc[len(dff)] = [mytime, 
                                 rec['fields']["name"],
                                 rec['fields']['stationcode'],
                                 rec['fields']['coordonnees_geo'],
                                 rec['fields']['nom_arrondissement_communes']]
  
        if int(data['nhits']) > 0:
            with open("illustration.csv", 'a') as f:
                dff.to_csv(f, header=True, index=False)
            print(mytime, " - ", iteration, " - Fin de la récupération, Nb de lignes récupérées: ", data['nhits'])
        else:
            print(mytime, " - ", iteration, " - Pas de données à récupérer.")
    iteration = iteration + 1
                
 
durationinsec = 5*60-3
iteration = 1   
#limiteIter = 288000 

update()

#programme adapté de datacorner.fr/velib/ (Benoit Cayla)