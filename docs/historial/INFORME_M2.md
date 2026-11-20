# M2: estimación del PIB informal de Colombia con un DSGE estándar

**Muestra:** 2005T2–2026T2, 85 variaciones trimestrales (DANE original base 2015 + ICML/GEIH, registro Zenodo 24357243). 2020T2–2020T4 se tratan como datos faltantes.
**Estado:** el modelo converge y está verificado. El nivel del PIB informal no queda identificado con los seis agregados actuales. Agregar el ingreso laboral relativo informal/formal de la GEIH lo identifica (comprobado por simulación).

## 1. Por qué M1 no convergía ni ajustaba

| Problema en M1 | Evidencia | Cambio en M2 |
|---|---|---|
| El criterio de ajuste era inalcanzable | Sin COVID, la autocorrelación de los crecimientos es 0,02 en PIB, 0,14 en consumo y −0,12 en empleo. Ni un AR(1) baja de 0,99 en RMSE/desviación. | Se compara contra un AR(1) y se informa la tabla completa. |
| Persistencias acumuladas en el límite de 0,98 | La productividad tiene raíz casi unitaria y el prior estaba truncado en 0,98. | La tendencia es estocástica (raíz unitaria en la productividad agregada), con priors Beta sin truncamiento artificial. |
| COVID dentro de la verosimilitud | En 2020T2 el PIB cae 17 % y el empleo 21 %. | 2020T2–T4 se tratan como faltantes en el filtro de Kalman. |
| Muestreador por bloques con escalas ad hoc y solo 430 pasos | R-hat de 1,03 a 1,11 | Propuesta con la inversa del hessiano en el modo, escala adaptada solo durante el burn-in, 4 × 30 000 pasos. |
| Errores de medición fijados a mano | 36 % de la dispersión robusta | Se estiman y se modelan en niveles. Eso implica una MA(1) negativa en los crecimientos, igual a la de los datos (−0,43 en informalidad, −0,52 en FBCF). |
| 3 choques para 6 observables | El error de medida cargaba con el ajuste. | 6 choques estándar. |
| La productividad relativa ρ venía de M0 | El propio diagnóstico de M0 dice que no está identificada. | Se estima θ (participación informal del PIB) y se mide su identificación. |

## 2. Modelo

Es un RBC estándar de dos sectores sin rigideces nominales.

**Hogar.** Utilidad logarítmica con desutilidad separable del trabajo formal e informal. El ingreso formal soporta una cuña τ_t: impuestos, aportes y costos de regulación.

**Tecnología.** Las firmas formales e informales usan la misma Cobb-Douglas y el capital se mueve entre sectores. El trabajo informal tiene una productividad relativa ρ̃. De ahí sale la identidad:

θ_t = ρ̃ s_t / [(1 − s_t) + ρ̃ s_t] = participación informal en el PIB = participación informal en el ingreso laboral.

**Tendencia y acumulación.** La tendencia laboral es estocástica, con crecimiento γ + μ_t. El capital se acumula con un choque a la eficiencia marginal de la inversión.

**Recursos.** Se incluye un residuo de absorción, x̄ + zx_t, que recoge la no aditividad de los volúmenes encadenados.

**Choques.** Seis: tendencia, gasto público, eficiencia de la inversión, oferta de trabajo, cuña de formalidad y absorción.

**Calibración** (en `run_summary.json`):
- Fijados de la literatura: α = 0,33, β = 0,985, inversa de Frisch = 1.
- Sacados de los datos: γ, s̄, C/Y, I/Y y G/Y.
- δ implícito por I/Y.

## 3. Resultados de la estimación (`resultados_m2/base/`)

**Convergencia.** R-hat normalizado por rangos máximo de 1,016. ESS bulk de al menos 1 480 en 13 de los 16 parámetros; los mínimos son me_i 283, sig_x 399 y rho_x 548. Aceptación entre 23 % y 29 %.

**Ajuste.** El modelo queda al nivel del AR(1) en PIB, consumo y empleo, y lo supera en gasto público (0,93 frente a 0,99) y en FBCF (0,83 frente a 0,85). Es lo máximo esperable con crecimientos que casi no tienen autocorrelación.

**Identificación del tamaño:**
- **Posterior frente a prior.** La desviación posterior de θ es 0,81 veces la del prior (`fig2`).
- **Verosimilitud perfilada.** Va de 1 143,8 en θ = 0,05 a 1 135,8 en θ = 0,80 (`fig3`). Favorece levemente valores bajos, pero esa información es débil y sesgable: el experimento simulado lo muestra.
- **Experimento simulado** (`experimentos/identificacion_simulada.py`). Con θ verdadero = 0,45:
  - Con los seis agregados, en tres réplicas el modo cae entre 0,51 y 0,56 y la desviación es 0,08, casi igual a la del prior.
  - Agregando el ingreso laboral relativo, el modo cae entre 0,448 y 0,459 y la desviación es 0,0045.
- **Variante con informal solo trabajo.** Es la tecnología de Ihrig-Moe y Elgin-Öztunalı: Y_i = A_i X N_i^η. En ella la informalidad sí responde al capital y a la tendencia, pero la verosimilitud perfilada de θ sigue plana. Cambiar la tecnología no resuelve la identificación.

**Dinámica de la informalidad.** En el modelo base, el 100 % de la varianza del logit de la informalidad viene de la cuña de formalidad. Es una propiedad estructural: con la misma tecnología en los dos sectores y utilidad separable, los choques agregados mueven el empleo formal y el informal en la misma proporción.

**PIB informal (prior agnóstico, Beta(0,35; 0,10)).** La mediana pasa de 32 % en 2005 a 22 % en 2026T2. La banda del 90 % en 2026T2 va de 12 % a 35 %. La tendencia viene de los datos; el nivel viene del prior.

## 4. Qué falta para el artículo

1. **Serie trimestral de ingreso laboral relativo.** Ingreso laboral medio de los ocupados informales dividido por el de los formales, ambos con la definición DANE, a nivel nacional, desde 2007 (o desde 2021 con el marco 2018, empalmado), calculado con el pipeline GEIH propio.
   - Formato: un CSV con columnas `quarter,ratio`.
   - Se ejecuta con `--ingresos ruta.csv`. El modelo agrega el choque de productividad relativa zr y un error de medida.
   - Conviene una versión por hora trabajada como robustez.
2. **Validación externa del nivel.**
   - DANE estima que la economía no observada (ENO) es 22,4 % de la producción en base 2015. No es el mismo concepto, porque es producción y no PIB e incluye actividades ilegales y deficiencias estadísticas.
   - Arango, Misas y López (2006) miden la economía subterránea a partir de la demanda de efectivo.
3. **Sensibilidad.** Inversa de Frisch, α, muestra desde 1993 y otra ventana para el COVID.

## 5. Reproducción

```
python estimar_m2.py --datos datos/datos_dsge_estimacion.csv --salida resultados_m2/base --draws 30000 --burn 10000
python figuras_m2.py --resultados resultados_m2/base
python experimentos/identificacion_simulada.py
python -m pytest tests/test_m2.py -q
```

Las pruebas verifican:
- el estado estacionario;
- la condición de Blanchard-Kahn y la ecuación linealizada;
- la densidad del prior frente a SciPy;
- la verosimilitud de Kalman frente a la gaussiana exacta;
- la exclusión de los trimestres del COVID.

## 6. Estimación de largo plazo 1993–2026 (`resultados_m2/largo_plazo_1993/`)

**Qué entra a la verosimilitud.** Desde 1993 entran el PIB, el consumo, la FBCF y el gobierno. Antes de 2005 esas series son retropoladas, y la crisis de 1999 sí queda dentro de la muestra. El empleo y la informalidad de ICML entran solo desde 2001, porque para 1993–2000 son reconstrucciones y no observaciones. Para el nivel de la serie de PIB informal se usa en todo el período la tasa ICML, incluida la reconstruida.

**Por qué se excluyen.** Probé tres variantes y miré la verosimilitud perfilada de θ entre 0,05 y 0,40:

| Variante | Caída de la verosimilitud entre θ = 0,05 y θ = 0,40 | Mediana posterior de θ |
|---|---|---|
| Empleo e informalidad reconstruidos dentro | 10 puntos | 0,18 |
| Solo la informalidad reconstruida fuera | 14 puntos | 0,16 |
| Ambas fuera (versión base) | 1,4 puntos: plana, igual que la muestra desde 2001 | 0,29 |

Toda la información aparente sobre el tamaño venía de la reconstrucción ICML del empleo 1993–2000. No se puede presentar como identificación.

**Resultado.**
- **Convergencia:** R-hat máximo de 1,004 y ESS bulk mínimo de 1 563.
- **θ:** mediana 0,29, banda del 90 % de 0,17 a 0,44. La desviación posterior es 0,80 veces la del prior.
- **PIB informal (mediana):**

| Año | Mediana | Banda del 90 % |
|---|---|---|
| 1993 | 33 % | 21–46 % |
| 2000 | 34 % | — |
| 2010 | 31 % | — |
| 2026 | 21 % | 13–32 % |

La trayectoria (estable hasta 2003 y en descenso desde entonces) viene de los datos. El nivel sigue dependiendo del ancla.

**Para identificar el nivel desde 1993** se necesita la razón de ingresos laborales informal/formal. Con la GEIH existe desde 2007. Esa serie fija ρ̃ en 2007–2026, y el filtro extiende el nivel hacia 1993–2006 con su propia incertidumbre: no hace falta tener ingresos desde 1993 para estimar desde 1993. Extenderla hacia atrás requiere procesar la ECH (2001–2006) y la ENH (1993–2000).
