# ¿Qué tan grande es la economía informal de Colombia? Un DSGE bayesiano, 1993–2026

[![Pruebas](https://github.com/jaimevera1107/informalidad_dsge_colombia/actions/workflows/pruebas.yml/badge.svg)](https://github.com/jaimevera1107/informalidad_dsge_colombia/actions/workflows/pruebas.yml)
[![Licencia: MIT](https://img.shields.io/badge/c%C3%B3digo-MIT-blue.svg)](LICENSE)
[![Datos: CC BY 4.0](https://img.shields.io/badge/datos-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Este repositorio estima qué parte del PIB de Colombia produce el trabajo informal, trimestre a trimestre, de 1993 a 2026. Usa un modelo de equilibrio general dinámico estándar de dos sectores, estimado con métodos bayesianos.

El nivel se identifica con una serie nueva de ingresos de trabajadores informales frente a formales, construida con los microdatos de la Gran Encuesta Integrada de Hogares (GEIH). Contiene el código, los datos, todas las estimaciones y el artículo (en inglés).

**Autor:** Jaime A. Vera-Jaramillo · [ORCID 0000-0001-7611-6251](https://orcid.org/0000-0001-7611-6251)

---

## Resultados principales

| Año | Participación informal en el PIB | Intervalo 90% | PIB informal (billones de pesos de 2015) | Informalidad laboral | VA por trabajador informal / formal |
|---|---|---|---|---|---|
| 1994 | 30,5% | 29,2–31,2 | 120,0 | 68,8% | 0,20 |
| 2000 | 29,7% | 29,0–30,3 | 127,5 | 69,6% | 0,19 |
| 2005 | 27,8% | 27,5–28,1 | 143,3 | 67,3% | 0,19 |
| 2010 | 27,9% | 27,8–27,9 | 178,5 | 66,5% | 0,20 |
| 2015 | 25,7% | 25,6–25,9 | 207,2 | 62,1% | 0,21 |
| 2019 | 23,4% | 23,3–23,5 | 205,9 | 60,0% | 0,20 |
| 2022 | 21,8% | 21,2–22,4 | 212,2 | 57,8% | 0,20 |
| 2025 | 20,4% | 19,8–21,2 | 208,7 | 55,5% | 0,21 |

- **Nivel.** El trabajo informal produce alrededor del 30% del PIB en los años 90 y el 20% en 2025–2026. La caída sigue a la de la tasa de informalidad laboral, que pasó de 69% a 54% de los ocupados.
- **Por qué 55% de los trabajadores produce 20% del PIB.** El valor agregado por trabajador informal es cerca de una quinta parte del de un trabajador formal. Con la misma brecha, una informalidad laboral de 30% equivaldría a cerca de 9% del PIB.
- **Comparación con la literatura.** La estimación está dentro del rango de las estimaciones publicadas para Colombia (Schneider y coautores, Medina y Schneider, Asllani et al., Banco Mundial y Vuletin), que difieren entre sí hasta en 19 puntos.
- **Identificación.** Lo que los datos identifican es la trayectoria de la participación, no el parámetro de largo plazo del modelo. La trayectoria cambia menos de 0,3 puntos desde 2007 cuando ese parámetro se fija en cualquier valor entre 10% y 40%.

> **Tres medidas distintas.** La informalidad *laboral* cuenta personas (54% de los ocupados). La informalidad *empresarial* del DANE cuenta micronegocios (90% en 2024). Este proyecto estima una tercera cosa: la proporción del *producto* que sale del trabajo informal.

## Qué entrega el modelo

Todo está en [`resultados_finales/`](resultados_finales/), con su diccionario de variables en [`resultados_finales/LEEME.md`](resultados_finales/LEEME.md).

**Series trimestrales, 1993T2–2026T2, con mediana e intervalo de 90%:**
- participación informal en el PIB;
- PIB informal y PIB formal en pesos de 2015;
- tasa de informalidad laboral sin ruido de encuesta;
- ocupados informales y formales;
- razón de ingresos informal / formal y productividad relativa del trabajo informal;
- valor agregado por trabajador en cada sector;
- componente cíclico del PIB;
- los siete choques estructurales: crecimiento de la tendencia, gasto público, eficiencia de la inversión, oferta laboral, cuña de formalidad, absorción y productividad informal.

**Series anuales** de las principales variables.

**Parámetros y estructura:**
- posterior de los 21 parámetros estimados, con diagnósticos de convergencia;
- estado estacionario: relación capital/PIB, tasa de renta, productividad relativa;
- descomposición de varianza: qué choques mueven la informalidad, la participación informal y el PIB;
- ajuste a un paso frente a un AR(1);
- participación del capital (α) estimada con una Cobb-Douglas.

**Validación:**
- 13 especificaciones alternativas;
- comparación con todas las estimaciones publicadas para Colombia;
- pruebas de identificación: perfil de verosimilitud, recuperación en datos simulados e invariancia de la trayectoria.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `resultados_finales/` | **Resultados finales de la estimación base**, listos para usar |
| `m2/` | Paquete del modelo: ecuaciones, solución, espacio de estados, filtro de Kalman, previas, muestreador, diagnósticos, α y ecuación de ingresos; `m2/sources/` descarga DANE, Zenodo y GEIH, y `m2/update.py` arma la actualización |
| `scripts/` | Puntos de entrada en línea de comandos (ver el flujo) |
| `datos/` | Insumos: base trimestral, ingresos GEIH, costos laborales legales, referencias publicadas, tablas GRECO y series anuales 1905–2025 |
| `resultados_m2/` | Salidas completas de cada estimación (una carpeta por especificación) |
| `paper/` | Fuente LaTeX del artículo, figuras y tablas |
| `docs/` | Comandos de las especificaciones alternativas (`SENSIBILIDADES.md`); `docs/historial/` guarda notas de trabajo |
| `.github/workflows/` | Pruebas en cada cambio y actualización trimestral automática |
| `tests/` | Pruebas automáticas |

### Estimaciones en `resultados_m2/`

| Carpeta | Especificación |
|---|---|
| `solo_trabajo_cuna_legal` | **Base:** sector informal solo con trabajo, costos no salariales legales, α = 0,354, δ = 4% |
| `solo_trabajo_cuna_aportes` | Solo aportes, sin prestaciones sociales |
| `solo_trabajo` | Cuña constante de 25% |
| `solo_trabajo_d6`, `solo_trabajo_a050` | α = 0,33 con δ = 6%; α = 0,50 |
| `solo_trabajo_eta05`, `solo_trabajo_eta09` | Previa de η centrada en 0,5 y en 0,9 |
| `solo_trabajo_desde2007` | Muestra desde 2007 |
| `solo_trabajo_capital` | Razón capital/PIB observada en niveles |
| `con_ingresos`, `con_ingresos_sin_cuna`, `con_ingresos_por_hora` | Misma tecnología en ambos sectores (cota superior) |
| `desde_1993`, `largo_plazo_1993`, `desde_1993_s_observada`, `base` | Solo agregados, sin ingresos (problema de identificación) |
| `identificacion_solo_trabajo` | Perfil, simulación y trayectorias con θ fijo |
| `comparacion`, `alfa` | Comparación con la literatura; participación del capital |

## Instalación

Requiere Python 3.10 o superior.

```powershell
git clone https://github.com/jaimevera1107/informalidad_dsge_colombia
cd informalidad_dsge_colombia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q
```

Antes de las estimaciones largas, en cada ventana de PowerShell:

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
```

## Actualización trimestral

Un solo comando descarga las fuentes públicas, actualiza los datos y recalcula los resultados finales:

```powershell
python scripts\actualizar.py
```

| Paso | Fuente | Qué hace |
|---|---|---|
| Cuentas nacionales | DANE, anexo `anex-GastoConstantes` (PIB por el lado del gasto, volúmenes base 2015) | Reemplaza 2005 en adelante, incluidas las revisiones del DANE. |
| Ocupados e informalidad | Última versión del ICML en Zenodo | Reemplaza toda la serie. Los meses posteriores al corte del ICML se completan con los microdatos de la GEIH, ajustados al nivel del ICML en los tres meses comunes. |
| Ingresos relativos | Microdatos GEIH del catálogo del DANE | Calcula solo los trimestres nuevos y los agrega a los anteriores, que nunca se recalculan. |
| Costos no salariales | Ley vigente | Extiende la serie hasta el año en curso. |
| Resultados | `scripts\resultados_finales.py` | Vuelve a pasar el suavizador de Kalman con el posterior ya estimado (segundos). |

Garantías:
- **Nada se escribe si algo falla.** El script se detiene si una fuente no responde, si cambia el formato de un archivo, si el nivel de 2005 cambia más de 5% (señal de un nuevo año base) o si una serie tiene un salto interanual implausible.
- **Cada versión anterior se guarda** en `datos/versiones/` con fecha y hora, y `datos/actualizacion.json` registra fuentes, versiones y ajustes.
- **La historia es fija.** Las cuentas de 1993–2004 son la desagregación propia de la retropolación del DANE. Si el DANE revisa el nivel de 2005, ese bloque se reescala con la revisión y conserva sus tasas de crecimiento.

Opciones útiles:

```powershell
python scripts\actualizar.py --simular             # descarga y valida, no escribe nada
python scripts\actualizar.py --sin-resultados      # solo los datos
python scripts\actualizar.py --validar-ingresos    # recalcula también el último trimestre fijo y compara
```

La clasificación de informalidad de los microdatos (`m2/sources/informality.py`) replica la regla del marco 2018 del procesamiento propio de la GEIH. Con los meses de abril a junio de 2026 reproduce la razón de ingresos fija con una diferencia de 0,003%.

**Actualización automática en GitHub.** El flujo `.github/workflows/actualizar.yml` corre `actualizar.py` el 22 de febrero, mayo, agosto y noviembre (unos días después de que el DANE publica el PIB trimestral) y abre un *pull request* con los datos y resultados nuevos. Nada llega a la rama principal sin revisión, y si no hay datos nuevos no abre nada. También se puede lanzar a mano desde la pestaña *Actions* (*Run workflow*). Para que pueda abrir *pull requests*, en el repositorio hay que activar *Settings → Actions → General → Allow GitHub Actions to create and approve pull requests*.

**Cuándo reestimar.** La actualización trimestral usa los parámetros ya estimados. Conviene volver a correr la estimación completa (paso 5 del flujo) una vez al año, o antes si el DANE cambia el año base de las cuentas, si la GEIH cambia la definición de informalidad o si cambian los aportes a cargo del empleador. Los cambios de ley en los aportes se agregan en la clase `StatutorySchedule` de `scripts\costo_no_salarial.py`.

## Flujo para reproducir todo

Todos los comandos se corren desde la raíz del repositorio.

1. **Ingresos GEIH por grupo** (requiere el pipeline propio de la GEIH y los microdatos del DANE):
   ```powershell
   python scripts\geih_ingresos_relativos.py --raiz "G:\My Drive\SENA\Procesamiento\GEIH" --salida datos\geih\ratio_ingresos_geih_v3.csv
   ```
2. **Costos no salariales legales** (ya incluidos en `datos/costos_laborales/`):
   ```powershell
   python scripts\costo_no_salarial.py
   ```
3. **Participación del capital α** (Cobb-Douglas por DOLS):
   ```powershell
   python scripts\estimar_alfa.py --participacion resultados_m2\con_ingresos\informal_output_share.csv --salida resultados_m2\alfa
   ```
4. **Razón de valor agregado:**
   ```powershell
   python scripts\construir_ratio_va.py --geih datos\geih\ratio_ingresos_geih_v3.csv --alfa 0.354 --salida datos\geih\ratio_va_0354.csv
   ```
5. **Estimación base** (unos 20 minutos con 8 núcleos):
   ```powershell
   python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --alfa 0.354 --depreciacion 0.0102 --ingresos datos\geih\ratio_va_0354.csv --columna-ingresos ratio_va --ajuste-ingresos-archivo datos\costos_laborales\costo_no_salarial.csv --columna-ajuste offset_log --salida resultados_m2\solo_trabajo_cuna_legal --cadenas 8 --draws 40000 --burn 15000 --workers 8
   ```
6. **Sensibilidades:** todos los comandos están en [`docs/SENSIBILIDADES.md`](docs/SENSIBILIDADES.md).
7. **Identificación:**
   ```powershell
   python scripts\identificacion_solo_trabajo.py --workers 8
   python scripts\identificacion_solo_trabajo.py --base resultados_m2\solo_trabajo_cuna_legal --ajuste-ingresos-archivo datos\costos_laborales\costo_no_salarial.csv --solo trayectorias --workers 4
   ```
8. **Comparación con la literatura:**
   ```powershell
   python scripts\comparar_referencias.py --base resultados_m2\solo_trabajo_cuna_legal --cota resultados_m2\con_ingresos --salida resultados_m2\comparacion
   ```
9. **Resultados finales:**
   ```powershell
   python scripts\resultados_finales.py --draws 400
   ```
10. **Tablas, figuras y paquete de arXiv:**
    ```powershell
    python scripts\tablas_paper.py --base resultados_m2\solo_trabajo_cuna_legal --out paper
    python scripts\tabla_sensibilidad.py --results resultados_m2 --out paper
    python scripts\figuras_paper.py --results resultados_m2 --out paper
    python scripts\preparar_arxiv.py
    ```

**Otros scripts:**
- `scripts\series_largas.py`: series anuales 1905–2025 (GRECO empalmado en 1993).
- `scripts\extraer_greco.py`: audita la extracción de las tablas GRECO.
- `scripts\identificacion_simulada.py`: experimento original con tecnología común.

## El modelo en una página

- **Hogar.** Un hogar representativo reparte su trabajo entre un sector formal, gravado con una cuña (aportes, impuestos a la nómina y costos de cumplimiento), y un sector informal.
- **Sector formal.** Las empresas formales usan capital y trabajo con una Cobb-Douglas.
- **Sector informal.** Produce solo con trabajo y retornos decrecientes. Esto corresponde a lo que mide la encuesta: el ingreso de cuenta propias y patrones informales es ingreso mixto y ya incluye el retorno a su propio capital.
- **Choques y estimación.** Hay siete choques estructurales y una tendencia con raíz unitaria. El modelo se linealiza, se resuelve con QZ (Klein) y se estima con el filtro de Kalman y Metropolis-Hastings de paseo aleatorio.
- **Identidad clave (Proposición 1).**

  θ_t = (1 − α)·R_t·x_t / (1 + (1 − α)·R_t·x_t),   con x_t = s_t / (1 − s_t)

  - s_t: tasa de informalidad laboral, observada;
  - R_t: ingreso informal por trabajador sobre el costo laboral formal, observado en la GEIH y llevado a costo laboral con los costos no salariales legales;
  - α: participación del capital, estimada.

Las condiciones de primer orden, el estado estacionario, la solución y las ecuaciones de medida están desarrolladas paso a paso en la sección 4 y el apéndice A del artículo.

## Datos

| Serie | Fuente | Periodo |
|---|---|---|
| PIB, consumo, inversión y gasto público (volúmenes, base 2015) | DANE; retropolación propia 1993–2004 | 1993–2026 |
| Ocupados y tasa de informalidad | ICML 2.6.1 ([Zenodo](https://doi.org/10.5281/zenodo.24357243)) | 1993–2026 |
| Ingresos informal / formal por posición y tamaño de empresa | Microdatos GEIH: procesamiento propio hasta 2026T2; desde ahí, descarga directa del catálogo del DANE | 2007–2026 |
| Costos no salariales legales | Ley 100/1993, Ley 797/2003, Ley 1122/2007, Ley 1607/2012 | 2007–2026 |
| Estimaciones publicadas | Schneider et al. (2010), Medina y Schneider (2019), Asllani et al. (2024), Elgin et al. (2021), Vuletin (2008) | varios |
| Series históricas | GRECO, Banco de la República (Borradores 121 y 134) | 1905–1997 |

## Artículo

*Measuring Colombia's Informal Output: Relative Earnings and Identification in a Bayesian Two-Sector Model, 1993–2026.* La fuente está en `paper/`; el PDF se compila con `latexmk -pdf main.tex` dentro de esa carpeta.

## Cómo citar

GitHub muestra la cita en formato APA y BibTeX con el botón *Cite this repository* (archivo [`CITATION.cff`](CITATION.cff)).

> Vera-Jaramillo, J. A. (2026). *Measuring Colombia's Informal Output: Relative Earnings and Identification in a Bayesian Two-Sector Model, 1993–2026.* Repositorio de código y datos.

## Licencia

- **Código** (`m2/`, `scripts/`, `tests/`): [MIT](LICENSE).
- **Datos construidos, resultados y artículo** (`datos/`, `resultados_m2/`, `resultados_finales/`, `paper/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es). Se pueden usar y adaptar citando la fuente.
- **Fuentes de terceros:** las cifras del DANE, el Banco de la República (GRECO) y las estimaciones publicadas conservan los términos de sus autores; aquí solo se incluyen series derivadas, con su referencia. Los documentos GRECO no se redistribuyen: `scripts\extraer_greco.py` los descarga del Banco de la República.
