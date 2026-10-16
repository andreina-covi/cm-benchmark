# Notas para presentar el avance — cm-benchmark

Documento editable. No es el paper ni el README: es un guion de diseño para explicar **qué se midió, por qué, y qué queda fuera a propósito**.

Puedes borrar, acortar o pegar cifras de tus corridas en los recuadros `[…]`.

---

## 1. Qué es y qué avanza hoy

Benchmark de **cognición espacial** para modelos visión–lenguaje (VLM).
La pregunta de investigación no es “¿el modelo describe la escena?”, sino
**¿construye y usa un mapa interno del espacio a lo largo del tiempo?**

Los ítems son **secuencias de frames** (AI2-THOR / SPOC) + pregunta.
Clases 1–3: opción múltiple. Clase 4: secuencia libre de acciones
(`move_ahead`, `rotate_left`, `rotate_right`, `move_back`).

**Estado del loop (hoy):** carpeta exportada SPOC → ground truth de episodio →
ítems generados por código. Aún no: validador GT, *vision-necessity*, FREEZE
inmutable, ni evaluación de VLMs.

**Regla madre:** el generador **nunca inventa** hechos espaciales. Cada respuesta
sale de metadatos (poses, FOV, eventos). `unsupported` no es un bug: es “esta
escena no puede probar ese constructo sin inventar”.

---

## 2. Taxonomía (eje de capacidad)

Cuatro clases, ocho constructos, **pipeline parcialmente ordenado**: las etapas
tardías subsumen a las tempranas (no son independientes).
En cada ítem se etiqueta además el **marco de referencia** (egocéntrico vs
alocéntrico). El análisis previsto es por constructo y por
constructo × marco.

| Clase | Nombre | Constructos | Qué exige |
|------:|--------|-------------|-----------|
| 1 | Codificación espacial | `egocentric_encoding` · `allocentric_encoding` | Todo está en la vista actual |
| 2 | Memoria espacial | `spatial_working_memory` · `invisible_displacement` | La info ya no está en la vista final |
| 3 | Operación sobre el mapa | `spatial_updating` · `perspective_taking` | Hay que transformar el punto de vista |
| 4 | Navegación | `route_knowledge` · `survey_based_route_planning` | Camino de acciones, no MCQ |

Definiciones canónicas: `configs/taxonomy.yaml`.

### Distinciones que conviene decir en voz alta

- **SWM vs desplazamiento invisible.** SWM: el objeto **sigue donde estaba**;
  recuerdas su rumbo de hace *k* pasos. ID: el objeto **se movió mientras
  estaba oculto**; la pregunta nombra el destino y pide el rumbo **ahora**.
- **Updating vs perspective-taking.** Updating: el agente **sí se movió**.
  Perspective-taking: te piden **imaginar** estar en A mirando a B; el agente
  no camina hasta A.
- **Route vs survey.** Route: rehacer un camino **ya caminado**.
  Survey: planear un camino **nunca caminado** a partir de lo visto (layout).
- **Egocéntrico vs alocéntrico.** Relativo a ti vs relativo a objetos / layout.
  Perspective-taking y survey se etiquetan alocéntricos **sin** necesitar el
  “frente intrínseco” de un objeto (ver §5).

---

## 3. Cómo se decide “adelante / derecha / atrás / izquierda”

No se usa norte geográfico. El marco es el de la **cámara del agente**
(AI2-THOR / SPOC): **+Z adelante, +X derecha**. El rumbo es
`atan2(dx, dz)`: 0° = adelante, 90° = derecha, 180° = atrás, 270° = izquierda.

Las respuestas MCQ son **cuatro etiquetas exclusivas** (nunca “adelante-izquierda”).
Se mapean con un **sector de adelanto** de semiancho *w*:

| Etiqueta | Intervalo |
|----------|-----------|
| adelante | \([360-w, 360) \cup [0, w)\) |
| a tu derecha | \([w, 180-w)\) |
| detrás | \([180-w, 180+w)\) |
| a tu izquierda | \([180+w, 360-w)\) |

**Dos valores de *w*, por constructo:**

- **20°** (sector adelante de 40°) — ítems donde el objeto **está en el FOV
  actual** (`egocentric_encoding`, disambiguadores, encode de SWM).
  Un FOV típico ~90° pondría casi todo lo visible dentro de un semiancho de 45°,
  y la respuesta correcta colapsaría a “adelante”.
  En estos ítems **“detrás de ti” no puede ser la verdad** (solo distractor).
- **45°** (cuatro cuadrantes de 90°) — ítems **de círculo completo** después de
  un cambio de pose (`spatial_updating`, query de invisible displacement).

**Perspective-taking es otro esquema:** no hay clase “adelante”.
Ángulo con signo de A→C respecto de A→B:

- derecha: \(-135° < \theta < 0°\)
- izquierda: \(0° < \theta < +135°\)
- detrás: \(|\theta| \ge 135°\)
- **se descarta** el ítem si \(\theta\) cae a menos de **10°** de 0° o ±135°
  (frontera ambigua).

La **distancia** (`within reach` / `nearby` / `far` / `beyond`) es un eje
aparte. “Beyond + frente” = lejos **pero delante**, no detrás.

### ¿Está basado en un benchmark SOTA?

**No es una copia de las bins de GQA, Visual Spatial Reasoning, CLEVR, ni de
un modelo psicolingüístico publicado.** Es un contrato de laboratorio:

- Los **cuadrantes de 45°** son el 4-way compass habitual en robótica / QSR.
- El corte de **20°** es una decisión de ingeniería para que, en FOV, aún
  existan ítems “izquierda/derecha” y no todo sea “adelante”.
- Las fronteras de PT se **rechazan** en vez de forzar un empate.
- La etiqueta se **recalcula desde poses**; no se confía en triples viejos
  tipo “frente-izquierda”.

Es **reproducible y determinista**. No es un estándar humano validado
(“la gente corta el espacio exactamente a 20°”). Conviene decirlo así:
*alineado con el eje de la cámara del simulador, no importado de un dataset 2D*.

---

## 4. Cómo se define que un objeto es “visible” para preguntar

La colección SPOC exporta detecciones FOV con métricas (`bbox-area`, `min-side`,
`visible-pixels`, `occupancy-ratio`, distancia). Eso **no** basta: un cup de
25 px en el borde del frame “está en FOV” pero un humano no lo identifica.

**Catálogo y logs de desplazamiento no se filtran.** El filtro es solo para
objetos **preguntables** (Q&A).

**Preferido:** árbol de decisión entrenado con etiquetas humanas
(`distinguible` / `no distinguible` / `ambiguo`), bundle
`analysis/dt_tune/visibility_filter.joblib`. Se usa `predict_proba` y bandas
de probabilidad; no se retoca el código cada vez que aparece un objeto malo.

**Respaldo si no hay modelo:** umbrales estáticos
`min_bbox_area=100`, `min_side=8`, `min_visible_pixels=40`.

**Más estricto para landmarks / encode de memoria:** si no hay árbol, pisos
más altos (p. ej. área 1200, lado 32, píxeles 300) + rechazo de recortes
finos en el borde. El encode de ID (objetos pequeños movibles) usa pisos
más blandos, pero el objeto debe estar **fuera de vista** desde el movimiento
hasta la pregunta.

**Nombres:** si `obj-type` es `Undefined` o el id es `ObjaScooter|…`, la
pregunta dice **Scooter**, no “ObjaScooter”.

Etiquetado y reajuste: `docs/generation.md`.

---

## 5. Por qué no se generan preguntas de `allocentric_encoding`

El constructo pide la relación **objeto–objeto en el marco del objeto de
referencia** (“a la izquierda **del sofá**”, usando el **frente intrínseco**
del sofá). Eso exige `edges_object_frame` / facing confiable.

En SPOC/AI2-THOR la rotación exportada del mesh **no es** un frente semántico
de confianza (una silla puede “mirar” según el asset, no según cómo se usa).
Inventar el frente violaría el invariante “no inventar hechos espaciales”.

Por eso el planner **devuelve siempre `unsupported`**
(`no_trusted_reference_facing_edges_object_frame_empty`) hasta que exista
esa metadata. **No es que la generación falle:** es un hueco de datos
declarado.

**Qué sí cubre el alocéntrico hoy:** `perspective_taking` (rumbo A→B a partir
de **dos posiciones**, sin frente intrínseco) y, cuando la escena lo permite,
`survey_based_route_planning` (layout visto, no “frente del objeto”).

---

## 6. Por qué a menudo no hay `survey_based_route_planning`

A diferencia del encoding alocéntrico, **survey sí está implementado**.
Si en *esta* corrida sale `unsupported`, la escena no cumple las puertas;
no hay que “arreglar” el generador para forzar ítems.

Survey pide un par origen→meta tal que:

1. **No** hay camino en `traversed_edges` (nunca se caminó ese par).
2. **Sí** hay camino en `viewed_edges` (se puede responder con lo visto;
   `viewed_edges` es un radio alrededor de la trayectoria, no raycasting).
3. El agente **no** está encima del origen al verlo (≥ 1 m; zona de éxito
   ObjectNav / GOAT).
4. Geodésica en el grafo completo en **1–30 m** (banda HM3D-OVON / GOAT-Bench /
   HSSD-200, 2024) **y** al menos la mediana de pares de *esa* escena.
5. Camino no casi recto: ratio geo/eucl ≥ ~1.05 (HSSD: &lt;1.05 es “casi línea”).

En **casas ProcTHOR chicas / planta abierta**, casi todo par de landmarks
o ya se caminó, o es un atajo corto, o el “visto” coincide con lo recorrido.
El resultado honesto es `unsupported` con conteo de motivos
(`path_on_traversed`, `no_viewed_path`, `geodesic_below_min`, etc.).

`route_knowledge` es el hermano: exige camino **sí** caminado, con **giros
reales** (2–12). Un pasillo recto de una habitación también queda
`unsupported`. Eso está alineado con Habitat (no premiar trayectos
adivinables a ciegas).

---

## 7. Desplazamiento invisible (por si preguntan “¿por qué hay tan pocos?”)

No basta con “lo vi en \(n-t\) y en \(n\) ya no está” (eso sería SWM u
oclusión). Hace falta un **evento de relocación oculta** en
`displacement_events` (`hidden_during=true`), encode distinguible, objeto
invisible del movimiento a la query, destino nombrable en los frames
(el `Floor` desnudo no vale), y cuatro opciones de rumbo distintas.

Las caminatas SPOC típicas **casi no recogen** esos eventos. Pocas filas
en el CSV → pocos ítems. Es un tema de **colección**, no de los sectores
de 20°/45°.

---

## 8. Arquitectura que hay que defender (una diapositiva)

```
colección SPOC → GT de episodio → generación de ítems → FREEZE → evaluación
```

- Respuestas y `answer_source`: **solo código**.
- El VLM **no** juzga si el ítem es correcto.
- Lo observado en evaluación **no** puede reeditar ítems (muro FREEZE).
- Clase 4: la vocabularios de acciones se dice **una vez** en el harness
  (`SYSTEM_INSTRUCTION`), no en cada pregunta.
- Scoring clase 4: simulación métrica (éxito / validez / SPL), no
  coincidencia exacta con la secuencia guardada.

**Hecho:** generador de episodios + ítems, filtro de visibilidad, taxonomía,
protocolo de formato.
**Pendiente:** validador GT, *vision-necessity* (ciego y caption deben
fallar), FREEZE, runner de modelos.

---

## 9. Cifras de la corrida (rellenar)

Escenas: `[p. ej. house_001030, house_006068, house_007514]`

| Constructo | ok | unsupported | nota de una línea |
|------------|---:|------------:|-------------------|
| egocentric_encoding | | | |
| allocentric_encoding | | | siempre unsupported (facing) |
| spatial_working_memory | | | |
| invisible_displacement | | | suele faltar `displacement_events` |
| spatial_updating | | | |
| perspective_taking | | | |
| route_knowledge | | | rectos → unsupported |
| survey_based_route_planning | | | casa chica / todo ya caminado |

---

## 10. Frases útiles si recortas a 5 minutos

1. El benchmark mide mapa cognitivo, no priors de lenguaje.
2. Ocho constructos en pipeline; FoR es eje transversal.
3. Las direcciones son sectores de cámara, 20° vs 45°, no un estándar SOTA 2D.
4. Preguntamos solo objetos que un humano distinguiría (árbol etiquetado).
5. `unsupported` es honestidad de datos: sin frente de objeto no hay encoding
   alocéntrico; sin camino inédito-pero-visto no hay survey.
6. Siguiente muro: validar → visión necesaria → congelar → evaluar VLMs.

---

*Fuente de verdad en el repo: `configs/taxonomy.yaml`, `docs/generation.md`,
`src/cm_benchmark/generation/constructs.py` (sectores),
`src/cm_benchmark/generator/visibility_filters.py` (visibilidad).*
