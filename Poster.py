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
Base_Dir = Path(__file__).resolve().parent  

#Rutas de la base de datos
Ruta_Precio_Oro = Base_Dir / "Base de datos - Poster.xlsx"
# %% Data frame
precio_oro_df = pd.read_excel(Ruta_Precio_Oro, sheet_name="Datos", skiprows=[1])
print(type(precio_oro_df))

print(precio_oro_df.head())
print(precio_oro_df.tail())
print(precio_oro_df.info())

precio_oro_df.index = pd.to_datetime(precio_oro_df["Fecha"])
# %%Creacion Series de tiempo

#La serie de tiempo se va a llamar precio_oro_serie y se va a crear a partir de la columna "Precio" del DataFrame precio_oro_df.
precio_serie = precio_oro_df["Precio"].copy()
precio_serie = pd.to_numeric(precio_serie, errors="coerce")
precio_serie = precio_serie.dropna()


precio_serie.index.freq = "MS"
precio_serie = pd.to_numeric(precio_oro_df["Precio"], errors="coerce").dropna()

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
# %%==================
#    2. ESTIMACION
#======================
#Candiadto 1

modelo_ARIMA110 = SARIMAX(
    np.log(precio_serie),
    order = (1,1,0),
    trend = "c",
    enforce_invertibility = False,
    enforce_stationarity = False
)
type(modelo_ARIMA110)

#Estimar el modelo
estimacion_ARIMA110 = modelo_ARIMA110.fit(disp=False)

type(estimacion_ARIMA110)

#Resultados estimacion
print(estimacion_ARIMA110.summary())

#%%
#Candiadto 2

modelo_ARIMA111 = SARIMAX(
    np.log(precio_serie),
    order = (1,1,1),
    trend = "c",
    enforce_invertibility = False,
    enforce_stationarity = False
)
type(modelo_ARIMA111)

#Estimar el modelo
estimacion_ARIMA111 = modelo_ARIMA111.fit(disp=False)

type(estimacion_ARIMA111)

#Resultados estimacion
print(estimacion_ARIMA111.summary())

#%%
#Candiadto 3

modelo_ARIMA011 = SARIMAX(
    np.log(precio_serie),
    order = (0,1,1),
    trend = "c",
    enforce_invertibility = False,
    enforce_stationarity = False
)
type(modelo_ARIMA110)

#Estimar el modelo
estimacion_ARIMA011 = modelo_ARIMA011.fit(disp=False)

type(estimacion_ARIMA011)

#Resultados estimacion
print(estimacion_ARIMA011.summary())
# %%==========================
#  3. VALIDACION DE SUPUESTOS
#=============================

residuales = estimacion_ARIMA110.resid.dropna()
type(residuales)

#Grafico de los resuduales

plt.figure(figsize=(12,6))
plt.plot(residuales.iloc[2:])
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
probplot(residuales.iloc[2:], dist="norm", plot=plt)
plt.title("Q-Q plot de los resuduos")
plt.grid(True)
plt.show()


# %%Test Jarque-Bera

JB_test =  jarque_bera(residuales.iloc[2:])

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
plt.legend()
plt.fill_between(
    pronostico_nivel.index,
    intervalo_nivel.iloc[:,0],
    intervalo_nivel.iloc[:,1],
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza"
)
#%%  BOOTSTRAP
N_SIM = 10_000 
H = len(pronostico_nivel)
rng = np.random.default_rng(2026)

# Extraer parámetros del ARIMA(1,1,0)
c = estimacion_ARIMA110.params["intercept"]
phi1 = estimacion_ARIMA110.params["ar.L1"]

# Residuales desde la observación 2 (omitimos las 2 primeras por la inicialización) y centrados en cero
residuales_boot = residuales.iloc[2:].to_numpy()
residuales_boot = residuales_boot - residuales_boot.mean()

# Condiciones iniciales históricas
y_T = np.log(precio_serie.iloc[-1])                          
dy_T = np.log(precio_serie.iloc[-1]) - np.log(precio_serie.iloc[-2])  

e = rng.choice(residuales_boot, size=(N_SIM, H), replace=True)
dy = np.zeros((N_SIM, H))
dy[:, 0] = c + phi1 * dy_T + e[:, 0]
for h in range(1, H):
    dy[:, h] = c + phi1 * dy[:, h-1] + e[:, h]

# 3) Reconstrucción de la serie en escala original (Nivel de precios)
trayectorias = np.exp(y_T + np.cumsum(dy, axis=1))

# Métricas puntuales y percentiles
boot_puntual = trayectorias.mean(axis=0)
boot_inf, boot_sup = np.percentile(trayectorias, [2.5, 97.5], axis=0)

fechas_pronostico = pronostico_nivel.index

# %% Gráfico del pronóstico con intervalos bootstrap vs. normal

plt.figure(figsize=(12, 6))
plt.plot(precio_serie, label="Datos históricos", color="darkblue")
plt.plot(fechas_pronostico, boot_puntual, color="red", label="Pronóstico (media bootstrap)")
plt.fill_between(
    fechas_pronostico, boot_inf, boot_sup,
    color="orange", alpha=0.3, label="Intervalo 95% bootstrap"
)
plt.plot(fechas_pronostico, intervalo_nivel.iloc[:, 0], color="gray",
         linestyle="--", linewidth=1, label="Intervalo 95% normal")
plt.plot(fechas_pronostico, intervalo_nivel.iloc[:, 1], color="gray",
         linestyle="--", linewidth=1)
plt.title(f"Pronóstico ARIMA(1,1,0) a {H} períodos: Bootstrap vs. Normal")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.grid(True)
plt.show()

# %% Tabla comparativa de pronósticos

tabla_bootstrap = pd.DataFrame({
    "Periodo": [d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d) for d in fechas_pronostico],
    "Pronóstico (Bootstrap)": boot_puntual,
    "Inferior (Bootstrap)": boot_inf,
    "Superior (Bootstrap)": boot_sup,
    "Inferior (Normal)": intervalo_nivel.iloc[:, 0].values,
    "Superior (Normal)": intervalo_nivel.iloc[:, 1].values,
}, index=fechas_pronostico)

print("\n--- Tabla Comparativa de Pronósticos ---")
print(tabla_bootstrap.round(2))

# %%Fin
