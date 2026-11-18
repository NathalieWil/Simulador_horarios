# Simulador híbrido de asignación de salones (AG + eventos discretos + dinámica continua)

Proyecto académico para *Modelos de Simulación*. **Todos los datos son sintéticos** y los parámetros físicos
y estocásticos de `ParamsSim` son **supuestos** a calibrar; nada de lo que aquí se reporta son datos reales de la institución.

## Estructura
| Módulo | Contenido | Rol de la guía |
|---|---|---|
| `rng.py` | Generador U(0,1) Park-Miller + auditoría estadística | Fase 1 / Rol B |
| `datos.py` | Esquema, generador sintético, carga de CSV reales | Rol C |
| `modelo.py` | Restricciones duras/blandas, fitness, validador independiente | Rol A/B |
| `ag.py` | AG adaptativo, líneas base | Rol B |
| `simulacion.py` | Motor de eventos discretos + EDO de CO2 (RK4) | Fase 2 / Roles A y B |
| `experimentos.py` | Escenarios, tablas CSV, figuras | Rol C |
| `cli.py` | Comandos de ejecución y «defensa en caliente» | Fase 3 |
| `gui.py` + `web/index.html` | **Interfaz gráfica** local (navegador): optimizar, horario, animación, defensa, resultados | Fase 3 / Rol C |

## Ejecución (Python ≥ 3.10, `numpy` no es necesario; `matplotlib` solo para figuras)
```
python -m unittest discover -s tests -v           # 13 pruebas
python -m simhorarios.cli rng                      # auditoría U(0,1)
python -m simhorarios.cli experimentos             # corrida completa (~3-4 min) -> resultados/
python -m simhorarios.cli experimentos --rapido    # prueba de humo
python -m simhorarios.cli defensa --ach 3 --n 30 --cap 40   # cambia un coeficiente y compara con el equilibrio analítico
```
## Interfaz gráfica
```
python -m simhorarios.cli gui          # abre http://127.0.0.1:8765 en el navegador (o doble clic en iniciar_gui.bat)
```
Pestañas: 1 Optimizar (AG), 2 Horario (Gantt), 3 Simulación (réplicas + animación del día), 4 Defensa en caliente, 5 Resultados.
Para datos reales: `python -m simhorarios.cli gui --salones salones.csv --clases clases.csv`.

## Datos reales
Reemplazar `generar_dataset` por `cargar_csv(salones.csv, clases.csv)` (esquema en `datos.py`).
