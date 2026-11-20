# Especificaciones alternativas

Comandos de las especificaciones de la tabla de sensibilidad y de las pruebas de identificación. Se corren desde la raíz del repositorio, en PowerShell, con el entorno activado y los hilos de BLAS limitados:

    $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1

Cada estimación tarda entre 20 y 60 minutos con 8 núcleos; ajustar `--workers` a los núcleos físicos. Las salidas quedan en `resultados_m2\<nombre>`.

## Insumos

    python scripts\costo_no_salarial.py
    python scripts\construir_ratio_va.py --geih datos\geih\ratio_ingresos_geih_v3.csv --alfa 0.354 --salida datos\geih\ratio_va_0354.csv
    python scripts\construir_ratio_va.py --geih datos\geih\ratio_ingresos_geih_v3.csv --alfa 0.33 --salida datos\geih\ratio_va_033.csv
    python scripts\construir_ratio_va.py --geih datos\geih\ratio_ingresos_geih_v3.csv --alfa 0.50 --salida datos\geih\ratio_va_050.csv

## Panel A: concepto del salario formal

Base, costos no salariales legales (aportes y prestaciones):

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --ajuste-ingresos-archivo datos\costos_laborales\costo_no_salarial.csv --columna-ajuste offset_log --salida resultados_m2\solo_trabajo_cuna_legal

Solo aportes (cadenas más largas para converger):

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 80000 --burn 30000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --ajuste-ingresos-archivo datos\costos_laborales\costo_no_salarial.csv --columna-ajuste offset_log_solo_aportes --salida resultados_m2\solo_trabajo_cuna_aportes

Cuña constante de 25%:

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --salida resultados_m2\solo_trabajo

## Panel B: otros supuestos (con cuña constante)

Depreciación de 6% anual (α = 0,33):

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.015 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.33 --ingresos datos\geih\ratio_va_033.csv --salida resultados_m2\solo_trabajo_d6

α = 0,50:

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.50 --ingresos datos\geih\ratio_va_050.csv --salida resultados_m2\solo_trabajo_a050

Previa de η centrada en 0,5 y en 0,9:

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --eta-media 0.5 --eta-sd 0.1 --salida resultados_m2\solo_trabajo_eta05
    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --eta-media 0.9 --eta-sd 0.05 --salida resultados_m2\solo_trabajo_eta09

Muestra desde 2007:

    python scripts\estimar_m2.py --desde 2007Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --salida resultados_m2\solo_trabajo_desde2007

Capital/PIB observado en niveles (inventario perpetuo):

    python scripts\estimar_m2.py --desde 1993Q1 --modelo solo_trabajo --informalidad-en-niveles --depreciacion 0.0102 --columna-ingresos ratio_va --cadenas 8 --draws 40000 --burn 15000 --workers 8 --alfa 0.354 --ingresos datos\geih\ratio_va_0354.csv --capital-en-niveles --capital-inicial-archivo datos\series_anuales_largas.csv --salida resultados_m2\solo_trabajo_capital

## Identificación

Perfil de verosimilitud y simulación con parámetros conocidos (15 a 25 minutos):

    python scripts\identificacion_solo_trabajo.py --workers 8

Trayectorias con θ fijo (invariancia):

    python scripts\identificacion_solo_trabajo.py --base resultados_m2\solo_trabajo_cuna_legal --ajuste-ingresos-archivo datos\costos_laborales\costo_no_salarial.csv --solo trayectorias --workers 4

## Tablas, figuras y resultados

    python scripts\tabla_sensibilidad.py --results resultados_m2 --out paper
    python scripts\comparar_referencias.py --base resultados_m2\solo_trabajo_cuna_legal --cota resultados_m2\con_ingresos --salida resultados_m2\comparacion
    python scripts\resultados_finales.py --draws 400
