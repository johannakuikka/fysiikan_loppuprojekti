# Tutki, missä kiihtyvyyden komponentissa kävelyliike havaitaan parhaiten, valitse se analyysiin kiihtyvyyden osalta.

# Määrittele havainnoista kurssilla oppimasi perusteella seuraavat asiat ja esitä ne numeroina visualisoinnissasi:
# - Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta
# - Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella
# - Keskinopeus (GPS-datasta)
# - Kuljettu matka (GPS-datasta)
# - Askelpituus (lasketun askelmäärän ja matkan perusteella)

# Esitä seuraavat kuvaajat:
# - Suodatettu kiihtyvyysdata, jota käytit askelmäärän määrittelemiseen.
# - Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys
# - Reittisi kartalla

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt

path = "kiihtyvyysdata.csv"
df = pd.read_csv(path)

st.title('Fysiikan loppuprojekti')

# Piirretään kiihtyvyysdatan kuvaajat
# st.line_chart(df, x = 'Time (s)', y = 'Linear Acceleration x (m/s^2)', x_label = 'Time (s)', y_label = 'Linear Acceleration x (m/s^2)')
# st.line_chart(df, x = 'Time (s)', y = 'Linear Acceleration y (m/s^2)', x_label = 'Time (s)', y_label = 'Linear Acceleration y (m/s^2)')
# st.line_chart(df, x = 'Time (s)', y = 'Linear Acceleration z (m/s^2)', x_label = 'Time (s)', y_label = 'Linear Acceleration z (m/s^2)')

# Kuvaajien perusteella x-komponentti näyttää parhaiten jaksollisen liikkeen.

# Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta:

#Suodatetaan datasta selvästi kävelytaajuutta suurempitaajuuksiset vaihtelut pois
#Filtteri:
from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#Filttereiden parametrit:
T = df['Time (s)'][len(df['Time (s)'])-1] - df['Time (s)'][0] #Koko datan pituus
n = len(df['Time (s)']) #Datapisteiden lukumäärä
fs = n/T #Näytteenottotaajuus (olettaen jotakuinkin vakioksi)
nyq = fs/2 #Nyqvistin taajuus
order = 3 #Kertaluku
cutoff = 1/(0.5) #Cutt-off taajuus

filtered_signal = butter_lowpass_filter(df['Linear Acceleration x (m/s^2)'], cutoff, nyq, order)
df['Filtered Linear Acceleration x (m/s^2)'] = filtered_signal

# Piirretään kuvaaja
st.title('Suodatettu kiihtyvyysdata (x-komponentti)')
st.line_chart(df, x = 'Time (s)', y = 'Filtered Linear Acceleration x (m/s^2)', x_label = 'Aika (s)', y_label = 'Suodatettu kiihtyvyys (m/s^2)')

#Lasketaan jaksojen määrä signaalissa (ja sitä kautta askelten määrä) laskemalla signaalin nollakohtien ylitysten määrä.
#Nolla ylitetään kaksi kertaa jokaisen jakson aikana.
jaksot = 0
for i in range(len(filtered_signal)-1):
    if filtered_signal[i]/filtered_signal[i+1] < 0:
        jaksot = jaksot + 1
st.write('Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta:', np.floor(jaksot/2), ' askelta')

# Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella

# Lasketaan valitun komponentin Fourier-muunnos ja tehospektri.
f = df['Linear Acceleration x (m/s^2)'] #Valittu signaali
t = df['Time (s)'] #Aika
N = len(df) #Havaintojen määrä
dt = np.max(t)/len(t) #Oletetaan sämpläystaajuus vakioksi

fourier = np.fft.fft(f,N) #Fourier-muunnos
psd = fourier*np.conj(fourier)/N #Tehospektri
freq = np.fft.fftfreq(N,dt) #Taajuudet
L = np.arange(1,int(N/2)) #Rajataan pois nollataajuus ja negatiiviset taajuudet

# Luodaan DataFrame Altair-kuvaajaa varten
spectrum_df = pd.DataFrame({
    'Taajuus': freq[L],
    'Teho': psd[L].real
})

# Piirretään tehospektrin kuvaaja
st.title('Tehospektri')
chart = alt.Chart(spectrum_df).mark_line().encode(
    x = alt.X('Taajuus', scale = alt.Scale(domain=[0, 10]), title = 'Taajuus (Hz)'),
    y = alt.Y('Teho', scale = alt.Scale(domain=[0, 500]), title = 'Teho')
).properties(
    width = 700,
    height = 400
).interactive()  # Tekee kuvaajasta zoomattavan
st.altair_chart(chart)

# Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella:

# Kävelydatan tehokkain taajuus (Hz): freq[L][psd[L]==np.max(psd[L])][0]
# Askeleeseen kuluva aika (s): 1/freq[L][psd[L]==np.max(psd[L])][0]
st.write('Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella:', freq[L][psd[L]==np.max(psd[L])][0]*np.max(t), ' askelta')

# Reittisi kartalla
path = "gpsdata.csv"
df = pd.read_csv(path)

#Määritellään kartan keskipiste ja laajuus (mittakaava, zoomaus)
lat_mean = df['Latitude (°)'].mean()
long_mean = df['Longitude (°)'].mean()

#Luodaan kartta
my_map = folium.Map(location = [lat_mean,long_mean], zoom_start = 16)

#Piirretään reitti
folium.PolyLine(df[['Latitude (°)','Longitude (°)']], color = 'blue', opacity = 1).add_to(my_map)

# Näytetään kartta Streamlitissa
st.title("Reitti kartalla")
st_folium(my_map, width=700, height=500)

#Määritetään keskinopeus (GPS-datasta), kuljettu matka (GPS-datasta) ja askelpituus (lasketun askelmäärän ja matkan perusteella).

def haversine(lon1, lat1, lon2, lat2): #Kahden pisteen koordinaatit

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r #Palauttaa koordinaattien välisen etäisyyden maapallon pintaa pitkin. 

# Lasketaan kokonaismatka Haversinen kaavan avulla
lat = df['Latitude (°)']
lon = df['Longitude (°)']

df['dist'] = np.zeros(len(df))

for i in range(len(df)-1):
    df.loc[i,'dist'] = haversine(lon[i],lat[i],lon[i+1],lat[i+1]) #Peräkkäisten pisteiden välimatka

df['tot_dist'] = np.cumsum(df['dist']) #Kokonaismatka
# Kokonaismatka metreinä
total_distance = df['tot_dist'].iloc[-1] * 1000
total_distance_rounded = round(total_distance)

# Kokonaisaika sekunneissa
total_time = df['Time (s)'].iloc[-1] - df['Time (s)'].iloc[0]
# Keskinopeus (m/s)
if total_time > 0:
    avg_speed = total_distance / total_time
else:
    avg_speed = 0

# Askelpituus metreinä
steps = 219
step_length = total_distance / steps

# Tulostetaan Streamlitilla
st.write("Kokonaismatka:", f"{total_distance_rounded}", "metriä")
st.write("Keskinopeus:", f"{avg_speed:.2f}", "m/s")
st.write("Askelpituus:", f"{step_length:.2f}", "metriä")