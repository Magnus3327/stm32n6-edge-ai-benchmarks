#!/bin/bash

# Vai nella cartella Report
cd "$(dirname "$0")" || exit

echo "🔧 Creazione ambiente virtuale..."
python3 -m venv .venv_html
source .venv_html/bin/activate

echo "📦 Installazione dipendenze..."
# Usiamo solo jupyter e nbconvert. 
# NOTA: Abbiamo rimosso pyppeteer/playwright perché generavano un errore fatale del compilatore C++ su Python 3.14 (Homebrew).
pip install -q jupyter nbconvert

echo "🚀 Esecuzione Notebook ed esportazione in HTML..."
# Esportiamo in HTML. È l'unico formato stabile al momento sul tuo Mac.
jupyter nbconvert --execute --to html NPU_Performance_Analysis.ipynb

echo "🧹 Pulizia in corso..."
deactivate
rm -rf .venv_html

echo ""
echo "=========================================================="
echo "✅ FATTO! È stato creato il file: NPU_Performance_Analysis.html"
echo "⚠️ PER AVERE IL PDF: apri il file HTML generato con Safari o Chrome, premi (Cmd + P) e clicca 'Salva come PDF'."
echo "=========================================================="
