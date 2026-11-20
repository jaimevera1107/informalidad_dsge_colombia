# Resultados finales

Estimación base: modelo DSGE de dos sectores con sector informal que produce solo con trabajo,
costos no salariales legales del empleo formal, α = 0,354 y depreciación de 4% anual,
estimado con todos los datos de 1993T2 a 2026T2. Las series con banda se calculan con 400
extracciones del posterior y el suavizador de Kalman: `_mediana` es la mediana y `_p05` y `_p95`
delimitan el intervalo de 90%.

Todo lo de esta carpeta se regenera con `python scripts\resultados_finales.py`.

## Archivos

| Archivo | Contenido |
|---|---|
| `01_series_trimestrales.csv` | Todas las series que entrega el modelo, trimestre a trimestre (ver diccionario) |
| `02_series_anuales.csv` | Agregados anuales de las principales series (años completos: 1994–2025) |
| `03_parametros_posterior.csv` | Previa, mediana, intervalo de 90%, R-hat y ESS de cada parámetro estimado |
| `04_estado_estacionario.csv` | Objetos de largo plazo del modelo en la mediana del posterior |
| `05_descomposicion_varianza.csv` | Descomposición de la varianza del error de pronóstico (1, 4, 20 y 80 trimestres) |
| `06_ajuste_un_paso.csv` | Error de predicción a un paso del modelo frente a un AR(1) |
| `07_resumen_corrida.json` | Calibración, diagnósticos de convergencia y configuración de la corrida |
| `08_sensibilidad.csv` | Participación informal en 1993T2 y 2026T2 bajo especificaciones alternativas |
| `09_comparacion_literatura.csv` | Esta estimación frente a cada valor publicado para Colombia |
| `10_participacion_capital_alfa.csv` | Estimación de α (Cobb-Douglas por DOLS) con depreciación de 4%, 6% y 8% |
| `11_identificacion_perfil.csv` | Perfil de verosimilitud de θ de estado estacionario, con y sin ingresos |
| `12_identificacion_simulacion.csv` | Recuperación de un θ conocido en datos simulados |
| `13_identificacion_trayectorias_theta_fijo.csv` | Trayectoria de la participación con θ de estado estacionario fijo en 0,10–0,40 |
| `figuras/` | Figuras del artículo (PDF) |

## Diccionario de `01_series_trimestrales.csv`

| Variable | Unidad | Qué es |
|---|---|---|
| `pib_observado_mmillones_2015` | miles de millones de pesos de 2015 | PIB trimestral del DANE (volumen encadenado, sin desestacionalizar) |
| `ocupados_observados` | personas | Ocupados totales (ICML) |
| `tasa_informalidad_observada` | proporción | Tasa de informalidad laboral publicada (ICML) |
| `participacion_informal_pib` | proporción | **Participación del trabajo informal en el PIB** (resultado principal) |
| `tasa_informalidad_laboral` | proporción | Tasa de informalidad laboral sin ruido de encuesta |
| `razon_ingresos_R` | razón | Ingreso informal por trabajador / costo laboral de un asalariado formal |
| `productividad_relativa_informal` | razón | Valor agregado por trabajador informal / formal, (1 − α)·R |
| `pib_informal_mmillones_2015` | miles de millones de pesos de 2015 | PIB producido por trabajo informal |
| `pib_formal_mmillones_2015` | miles de millones de pesos de 2015 | PIB producido por el sector formal |
| `ocupados_informales`, `ocupados_formales` | personas | Ocupados de cada sector, con la tasa sin ruido |
| `va_por_trabajador_informal_millones` | millones de pesos de 2015 por trimestre | Valor agregado por trabajador informal |
| `va_por_trabajador_formal_millones` | millones de pesos de 2015 por trimestre | Valor agregado por trabajador formal |
| `componente_ciclico_pib` | desviación logarítmica | PIB sin tendencia estocástica, respecto a su estado estacionario |
| `choque_crecimiento_tendencia_mu` | desviación | Choque al crecimiento de la productividad tendencial |
| `choque_gasto_publico_zg` | desviación logarítmica | Choque al gasto público |
| `choque_eficiencia_inversion_lnnu` | desviación logarítmica | Choque a la eficiencia marginal de la inversión |
| `choque_oferta_laboral_lnpsi` | desviación logarítmica | Choque a la desutilidad del trabajo |
| `choque_cuna_formalidad_omega` | desviación logarítmica | Choque a la cuña de formalidad (costo relativo de ser formal) |
| `choque_absorcion_x` | proporción del PIB | Choque de absorción (demanda externa y no aditividad) |
| `choque_productividad_informal_zr` | desviación logarítmica | Choque a la productividad del sector informal |

Cada variable estimada viene con `_p05`, `_mediana` y `_p95`. En `02_series_anuales.csv` el PIB
informal y formal son sumas de los cuatro trimestres y la participación anual es PIB informal
anual / PIB anual; las demás son promedios anuales.

## Cómo leer los resultados

- **La trayectoria está identificada; el estado estacionario no.** La participación trimestre a
  trimestre cambia menos de 0,3 puntos desde 2007 cuando θ de estado estacionario se fija entre
  0,10 y 0,40 (archivo 13). El parámetro θ de `03_parametros_posterior.csv` depende en parte de
  las previas y no debe leerse como una medición independiente.
- **Las bandas son de incertidumbre de parámetros.** La incertidumbre de concepto (cómo se
  traduce el ingreso de la encuesta en costo laboral, el valor de α, la tecnología) está en
  `08_sensibilidad.csv` y es mayor que la estadística.
- **2020T2–2020T4** están excluidos de la verosimilitud; sus valores son reconstrucciones del
  suavizador.
- **1993–2000** usa la reconstrucción del mercado laboral del ICML, y **antes de 2007** no hay
  serie de ingresos: el nivel de esos años lo proyecta el modelo hacia atrás.
