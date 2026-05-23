# WMH segmentation con FIS e Genetic Algorithm

## File necessari

Struttura consigliata:

```text
data/                 dataset e feature .npy
python/               script Python
matlab/               script MATLAB
wmh_preprocessing.py  preprocessing condiviso
```

## Come eseguire

Se usi Google Colab, carica tutti i file nella stessa cartella e poi esegui:

```python
!python python/wmh_fis_ga.py --max-slices 3 --population-size 40 --generations 30
```

Se usi il computer locale, installa prima:

```bash
pip install -r requirements.txt
```

Poi esegui:

```bash
python python/wmh_fis_ga.py --max-slices 3 --population-size 40 --generations 30
```

Per usare una soglia fissa diversa da `0.5`:

```bash
python python/wmh_fis_ga.py --threshold 0.6
```

Per cercare automaticamente la soglia migliore tra `0.1` e `0.9`:

```bash
python python/wmh_fis_ga.py --tune-threshold
```

## Versione MATLAB

Per usare il FIS MATLAB con il Genetic Algorithm MATLAB, esegui:

```matlab
run('matlab/wmh_fis_ga_matlab.m')
```

Il file `wmh_fis_ga_matlab.m` usa:

- `wmh_build_fis.m` per creare il FIS MATLAB;
- gli stessi file `.npy` del progetto;
- `evalfis` per applicare le regole fuzzy;
- un Genetic Algorithm implementato in MATLAB per ottimizzare i pesi delle regole.

Nota: per leggere i file `.npy`, MATLAB deve avere accesso a Python con NumPy installato.

## Preprocessing Python + FIS MATLAB

Per salvare le feature preprocessate in un formato che MATLAB puo caricare:

```bash
python python/export_preprocessed_features_for_matlab.py
```

Questo crea un file `.mat` nella cartella:

```text
matlab_preprocessed_features/
```

Poi in MATLAB puoi applicare le regole fuzzy con:

```matlab
run('matlab/wmh_apply_fis_to_preprocessed_features.m')
```

Questa pipeline usa Python solo per preprocessing e MATLAB per `evalfis`.

## Cosa fa il codice

1. Carica immagini FLAIR, maschere manuali e feature già estratte.
2. Normalizza le feature slice per slice con min-max classico.
3. Applica un Fuzzy Inference System con 12 regole.
4. Usa un Genetic Algorithm per ottimizzare i pesi delle regole.
5. Applica una soglia allo score fuzzy per ottenere una maschera binaria.
6. Valuta la segmentazione con Dice score.
7. Salva una figura finale in `wmh_fis_ga_result.png`.

Oltre alle feature già estratte, il codice calcola anche:

- `pixel intensity`: intensità del pixel FLAIR normalizzata;
- `local contrast`: differenza tra intensità del pixel e media locale;
- `local range`: differenza tra massimo e minimo in una finestra locale `5x5`.

La normalizzazione min-max viene applicata con `MinMaxScaler` di scikit-learn:

```python
from sklearn.preprocessing import MinMaxScaler
```

## Regole fuzzy usate

1. High intensity AND high local mean -> WMH
2. High intensity AND high local contrast -> WMH
3. High intensity AND medium local std -> WMH
4. High intensity AND high local std -> WMH
5. High local mean AND high kurtosis -> WMH
6. High intensity AND high skewness -> WMH
7. Medium intensity AND high mean AND high std -> WMH
8. High intensity AND high local range -> WMH
9. Medium intensity AND high mean AND high local range -> WMH
10. Low intensity -> non-WMH
11. Low local mean -> non-WMH
12. Low std AND low contrast AND low local range -> non-WMH

## Cosa mettere nella relazione

Mostra:

- una FLAIR originale;
- la maschera manuale;
- la mappa fuzzy score;
- la maschera predetta;
- i pesi finali delle regole;
- il Dice score prima e dopo il Genetic Algorithm.
