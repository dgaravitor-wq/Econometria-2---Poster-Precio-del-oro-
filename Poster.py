# %% Importacion de librerias

#Manejar rutas relativas
from pathlib import Path

#
import pandas as pd #manejo de bases de datos
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, probplot # type: ignore

#
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# %% Base de datos

#Obtener ruta de directorio
Base_Dir = Path(__file__).resolve().parent    #__file__ hace referencia al archivo actual de pyhton

#Rutas de la base de datos
Ruta_Precio_Oro = Base_Dir / "Base de datos - Poster.xlsx"
# %% Data frame
precio_oro_df = pd.read_excel(Ruta_Precio_Oro, sheet_name="Datos", skiprows=[1])
print(type(precio_oro_df))

print(precio_oro_df.head())
print(precio_oro_df.info())

precio_oro_df.index = pd.to_datetime(precio_oro_df["Fecha"])
# %%Creacion Series de tiempo

#La serie de tiempo se va a llamar precio_oro_serie y se va a crear a partir de la columna "Precio" del DataFrame precio_oro_df.
precio_serie = precio_oro_df["Precio"].copy()
precio_serie = pd.to_numeric(precio_serie, errors="coerce")
precio_serie = precio_serie.dropna()

print(precio_serie.head())
print(type(precio_serie))
print(precio_serie.describe())
# %%=========================
#   PASO 1: Idetificacion
#===========================

#Grafico de la serie de tiempo
plt.figure(figsize=(12, 6))
plt.plot(precio_serie,color="darkblue", label="Precio del oro")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title("Serie de tiempo: Precio del oro")
plt.legend()
plt.show()
# %%FAC Y FACP

fig,axes = plt.subplots(1,2,figsize=(14,6))

plot_acf(
    precio_serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC")

plot_pacf(
    precio_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP")
# %%Test de Raiz Unitaria

adf_result = adfuller(precio_serie)

print(".\n----Test de Raiz Unitaria ADF----")
print("Estadistico ADF:", adf_result[0])
print("Valor-p:", adf_result[1])
print("Rezagos usados:", adf_result[2])
print("Observaciones usadas:", adf_result[3])
print("Valores criticos:")
for nivel, valor in adf_result[4].items():
    print(f"  {nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. La serie es estacionaria según el test ADF.")
else:
    print("ADF: No rechazamos H0. La serie no es estacionaria según el test ADF.")

# %% Diferenciar la serie de tiempo

precio_serie_diff = precio_serie.diff().dropna()

plt.figure(figsize=(12, 6))
plt.plot(precio_serie_diff, color="blue", label="Precio del oro (diferenciado)")
plt.xlabel("Fecha")     
plt.ylabel("Precio Diferenciado")
plt.title("Serie de tiempo: Precio del oro (diferenciado)")
plt.grid(True)
plt.show()

# %%Diferencia del logaritmo de la serie de tiempo

precio_serie_logdiff = np.log(precio_serie).diff().dropna()
precio_serie_logdiff.index.freq = 'MS'

plt.figure(figsize=(12, 6))
plt.plot(precio_serie_logdiff, color="blue", label="Precio del oro (log-diferenciado)")
plt.xlabel("Fecha")
plt.ylabel("Log(Precio) Diferenciado")
plt.title("Serie de tiempo: Precio del oro (logaritmo diferenciado)")
plt.grid(True)
plt.show()
# %% FAC y FACP de la serie log-diferenciada    

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_acf(
    precio_serie_logdiff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC")

plot_pacf(
    precio_serie_logdiff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP")
# %%adf a la serie log-diferenciada

adf_result_logdiff = adfuller(precio_serie_logdiff)

print("\n----Test de Raiz Unitaria ADF (Log-Diferenciado)----")
print("Estadistico ADF:", adf_result_logdiff[0])
print("Valor-p:", adf_result_logdiff[1])
print("Rezagos usados:", adf_result_logdiff[2]) 
print("Observaciones usadas:", adf_result_logdiff[3])
print("Valores criticos:")
for nivel, valor in adf_result_logdiff[4].items():
    print(f"  {nivel}: {valor}")

if adf_result_logdiff[1] < 0.05:
    print("ADF: Rechazamos H0. La serie log-diferenciada es estacionaria según el test ADF.")
else:
    print("ADF: No rechazamos H0. La serie log-diferenciada no es estacionaria según el test ADF.")
    #Dado que la FAC 
# %% Criterios de informacion AIC y BIC
modelos_candidatos = [(1,1,0),
                      (2,1,1),
                      (1,1,2),
                      (0,1,1),
                      (1,1,1),
                      (2,1,0),
                      (0,1,2)]

resultados = []

for order in modelos_candidatos:
    modelo = SARIMAX(np.log(precio_serie),order=order, trend="c")
    resultado = modelo.fit(disp=False)
    resultados.append({
        "Modelo":f"ARIMA{order}",
        "AIC": resultado.aic,
        "BIC": resultado.bic,
        "log-likelihood": resultado.llf
    })

resultados_IC = pd.DataFrame(resultados)
resultados_IC = resultados_IC.sort_values(by="AIC").reset_index(drop=True)

print("\nResultados de los modelos ARIMA:")
print(resultados_IC)
# %%==================
#    2. ESTIMACION
#======================

modelo_ARIMA110 = SARIMAX(
    np.log(precio_serie),
    order = (1,1,0),
    trend = "c",
    enforce_stationarity = False,
    enforce_invertibility = False
)
type(modelo_ARIMA110)

#Estimar el modelo
estimacion_ARIMA110 = modelo_ARIMA110.fit(disp=False)

type(estimacion_ARIMA110)

#Resultados estimacion
print(estimacion_ARIMA110.summary())
# %%==========================
#  3. VALIDACION DE SUPUESTOS
#=============================

residuales = estimacion_ARIMA110.resid.dropna()
type(residuales)

#Grafico de los resuduales

plt.figure(figsize=(12,6))
plt.plot(residuales.iloc[2:-1]) #PREGUNTAR A MONITOR O PROFESOR SI SE PUEDE EMPEZAR EN 2 PORQUE EN 1 Y 0 SE DISPARA
plt.title("Grafico Residuales modelo ARIMA(1,1,0)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

print(residuales.describe())
# %%FAC y FACP de los residuos

fig,axes = plt.subplots(1,2, figsize =(12,6))

plot_acf(
    residuales,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC de los residuales")

plot_pacf(
    residuales,
    lags=24,
    alpha = 0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales")

plt.tight_layout()
plt.show()
# %%Test de Supuestos

ljung_box = acorr_ljungbox(
    residuales,
    lags = [6,12,18,24],
    return_df = True
)

print("Prueba ljung box")
print(ljung_box)
# %% FAC y FACP de residuales al cuadrado

fig,axes = plt.subplots(1,2, figsize=(12,6))

plot_acf(
    residuales**2,
    lags =24,
    alpha = 0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuos al cuadrado")

plot_pacf(
    residuales**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP de los residuales al cuadrado")
# %%Test ARCH de heterocedasticidad
arch_test = het_arch(
    residuales,
    nlags=24
)

print(".\n Prueba Arch de Heterocedasticidad")
print("LM statistic:", arch_test[0])
print("LM p-Value:", arch_test[1])
# %%Q-Q plot de los residuos

plt.figure(figsize=(7,7))
probplot(residuales.iloc[2:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los resuduos")
plt.grid(True)
plt.show()

# %%Test Jarque-Bera

JB_test =  jarque_bera(residuales.iloc[2:-1])

print("Prueba Jarque-Bera")
print("Estadistico:", JB_test[0])
print("P-Value:", JB_test[1])
# %%==================
#   4. PRONOSTICO
#=====================

pronostico_log = estimacion_ARIMA110.get_forecast(steps=10)

pronostico_puntual_log = pronostico_log.predicted_mean
varianza_pronostico_log = pronostico_log.var_pred_mean
intervalos_log = pronostico_log.conf_int()

pronostico_nivel = np.exp(pronostico_puntual_log +0.5 * varianza_pronostico_log)
intervalo_nivel = np.exp(intervalos_log)


tabla_pronosticos= pd.DataFrame({
    "Pronostico Puntual":pronostico_nivel,
    "Limite inferior":intervalo_nivel.iloc[:,0],
    "Limite superior":intervalo_nivel.iloc[:,1]
})

print(tabla_pronosticos)
# %%Grafico Pronostico

plt.figure(figsize=(12,6))
plt.plot(precio_serie, label="Datos historicos")
plt.plot(pronostico_nivel, color="Red", label = "Pronostico")
plt.fill_between(
    pronostico_nivel.index,
    intervalo_nivel.iloc[:,0],
    intervalo_nivel.iloc[:,1],
    color="pink",
    alpha=0.3,
    label="Intervalo de confianza"
)
plt.title("Grafico Pronostico")
plt.show()
# %%Fin
