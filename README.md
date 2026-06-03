# Baseline Performance Modelli NPU

Questo repository fa parte di un progetto di tirocinio curriculare svolto presso l'**Università degli Studi di Trento**, all'interno del **DISI** (Dipartimento di Ingegneria e Scienza dell'Informazione), sotto la supervisione dei professori **Vella** e **Yildirim**.

## Descrizione del Progetto

Il progetto si concentra sull'ottimizzazione di operatori AI su microcontrollore **STM32N6**.

Nello specifico, il repository confronta:
- Le prestazioni reali misurate su hardware rispetto alle stime statiche fornite dai compilatori).
- Due generazioni di compilatori e ambienti software STMicroelectronics: **AI Core v2.2** (tramite STM32CubeMX - X-CUBE-AI e ST DevCloud) e **AI Core v4.0** (tramite STM32 AI Studio e ST DevCloud).
- Due piattaforme di esecuzione: la scheda fisica locale **STM32N6 Nucleo** e la scheda **Discovery Kit (DK)** tramite l'ambiente cloud **ST DevCloud**.

## Struttura del Repository

Il repository è organizzato come segue:

- **`Models/`**: Contiene i file dei modelli neurali (in formato compresso o quantizzato) utilizzati per i benchmark prestazionali delle singole operazioni.
- **`Output/`**: Contiene i log raw, i report e i file di configurazione generati dai compilatori locali (CubeMX) e dai test eseguiti su ST DevCloud.
- **`Results/`**: Contiene i dataset in formato CSV strutturati per il confronto:
  - `baseline/`: Dati prestazionali misurati direttamente sull'hardware per ciascun tool.
  - `estimated/`: Stime statiche di memoria e complessità fornite dai compilatori.
- **`Report/`**: Contiene i deliverable dell'analisi delle prestazioni:
  - `NPU_Performance_Analysis.ipynb`: Notebook Jupyter contenente le tabelle riassuntive e i grafici di confronto.
  - `NPU_Performance_Analysis.html` & `NPU_Performance_Analysis.pdf`: Report esportati pronti per la consultazione e la stampa.
  - `plot/`: Grafici esportati (tempi di inferenza, confronto RAM, complessità MACC e distribuzioni statistiche).
- **`generate_nb.py`**: Script Python utilizzato per generare programmaticamente la struttura del notebook Jupyter e compilare i report finali in HTML e PDF.

## Guida all'Uso

Per rigenerare il notebook di report e compilare le versioni HTML e PDF:

1. **Creare e attivare l'ambiente virtuale Python:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Installare le dipendenze richieste:**
   ```bash
   pip install nbformat nbconvert pandas seaborn matplotlib ipykernel jinja2 ipython
   ```

3. **Eseguire lo script di generazione:**
   ```bash
   python3 generate_nb.py
   ```
   Lo script salverà il notebook vuoto in `Report/NPU_Performance_Analysis.ipynb`.

4. **Eseguire e compilare i report:**
   Per eseguire le celle del notebook e salvare i grafici all'interno del file `.ipynb`:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace Report/NPU_Performance_Analysis.ipynb
   ```
   Per esportare il report finale in HTML e stamparlo in PDF (tramite Chrome headless):
   ```bash
   jupyter nbconvert --to html Report/NPU_Performance_Analysis.ipynb
   ```
   Generazione del PDF da Chrome:
   ```bash
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --no-sandbox --print-to-pdf="Report/NPU_Performance_Analysis.pdf" "Report/NPU_Performance_Analysis.html"
   ```
