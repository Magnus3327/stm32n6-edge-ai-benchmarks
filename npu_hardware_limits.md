# 🛑 Analisi dei Limiti Hardware e Compiler (STM32N657 NPU)

> **Progetto:** Baseline Performance NPU
> **Target:** STM32N657 (Neural Art Accelerator)
> **Toolchain:** STM32Cube.AI (ST Edge AI Core)
> **Data Analisi:** 2026-06-16

Durante il porting dei modelli di Computer Vision (Object Detection e Segmentation) sulla NPU di test, abbiamo riscontrato e documentato dei limiti architetturali fondamentali che impediscono l'esecuzione di modelli ad alta risoluzione (es. 640x640) sul dispositivo.

---

## 1. Limite Architetturale della SRAM Interna (npuRAM)

La NPU richiede fisicamente che i tensori di attivazione (gli input e gli output intermedi dei singoli layer convoluzionali) risiedano nei suoi banchi di memoria ultra-veloce dedicata, denominati **`npuRAM`**.

*   **Capacità Hardware:** La scheda in uso possiede 4 banchi `npuRAM` da **448 KB** ciascuno, per un totale di circa **1.78 MB**.
*   **Il problema dei Modelli Standard:** Un modello come YOLOv11-n (o v8-n) a risoluzione nativa 640x640, anche quantizzato in INT8, genera un tensore massimo di attivazione (es. 320x320x16) pari a circa **1.6 MB**. Se sommiamo questo al tensore dell'immagine di input (1.17 MB), il fabbisogno supera ampiamente la capienza fisica totale e/o il limite del singolo blocco contiguo allocabile dalla NPU.
*   **Errore riscontrato:** `Model does not fit in Board. RAM Size too low`.

> [!IMPORTANT]
> **Conclusione Tecnica:** L'hardware NPU rifiuta di compilare ed eseguire qualsiasi rete neurale in cui l'operazione del singolo layer richieda più SRAM interna di quella fisicamente disponibile. Non esegue "tiling" automatico per dividere i tensori.

---

## 2. Inutilizzabilità della HyperRAM per il Calcolo Attivo

La scheda di sviluppo è dotata di **32 MB di HyperRAM** (memoria esterna). Sebbene sia configurabile in AI Studio per "Activations and weights", abbiamo dimostrato che la NPU non può usarla come "memoria di lavoro principale" per bypassare i limiti della `npuRAM`.
Il bus di comunicazione verso l'esterno è verosimilmente troppo lento per i requisiti della Neural Art Accelerator, forzando il compilatore a esigere l'allocazione nella SRAM interna.

---

## 3. Limiti di Fallback su CPU (Out of Memory)

Nel tentativo di aggirare il limite della NPU disabilitandola (forzando l'esecuzione sul core ARM Cortex), abbiamo riscontrato un ulteriore collo di bottiglia del compilatore.

*   Per eseguire il layer più pesante di YOLOv11-n (640x640) su CPU, il toolchain richiede un blocco contiguo di **~6.4 MB** (`Minimal sizes needed are: 6405160`).
*   La SRAM interna dedicata alla CPU è divisa in `cpuRAM1` (624 KB) e `cpuRAM2` (1024 KB).
*   Anche in questo caso, il compilatore fallisce (`UnfeasibleAllocation`) perché non è in grado o si rifiuta di allocare i 6.4 MB dinamici nella lenta HyperRAM esterna da 32 MB.

---

## 💡 Workaround e Best Practices Emerse

Per poter procedere con l'analisi delle performance (baseline) su questa NPU, è obbligatorio adattare il carico di lavoro ai vincoli del chip:

1.  **Downscaling della Risoluzione:** Per modelli heavy come YOLO (Detection), l'unica via praticabile per il deployment Edge è ridurre l'input. Portando l'input a **224x224** o **320x320**, il tensore massimo crolla a **~200 KB**, rientrando comodamente in un singolo banco `npuRAM` da 448 KB.
2.  **Modelli Nativamente Compatibili:** Modelli di image classification (es. **ResNet34**) o vision encoder (es. **MobileCLIP-B Image**) che nascono con input di `224x224` o `256x256` si sposano perfettamente con questa architettura hardware e compileranno senza errori.
3.  **Ottimizzazione Pesi:** I pesi (modello) non soffrono di questo limite dinamico, in quanto possono essere agevolmente caricati nella OctoFlash esterna (114 MB) in modalità read-only.
