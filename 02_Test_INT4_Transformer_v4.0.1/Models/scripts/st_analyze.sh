#!/bin/bash
# Esegue stedgeai analyze per STM32N6 su un modello e stampa un verdetto categorizzato.
# Uso: st_analyze.sh <modello.onnx|.tflite> [tag_output]
# Categorie:
#   RAM        -> "RAM too low"/"does not fit": limite fisico (ESIBIBILE)
#   LAYER      -> "NOT IMPLEMENTED: <op>": limite compiler netto (ESIBIBILE) se non e' artefatto
#   COMPILER   -> "INTERNAL ERROR"/"TOOL ERROR"/E1xx: bug/limite compiler (ESIBIBILE, da segnalare a ST)
#   ARTIFACT   -> allowzero/NMS/malformed/shape inference/list index/broadcast: da FIXARE (export)
#   OK         -> compila: candidato per esecuzione su scheda
ST=/Applications/ST/STEdgeAI/4.0
MODEL="$1"
TAG="${2:-$(basename "$MODEL" | sed 's/\.[^.]*$//')}"
OUT="/tmp/st_sweep/$TAG"
mkdir -p "$OUT"
LOG="$OUT/analyze.log"

"$ST/Utilities/mac/stedgeai" analyze --model "$MODEL" --target stm32n6 --st-neural-art -o "$OUT" \
  > "$LOG" 2>&1
CLEAN=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG")

verdict() { echo "$1|$2|$3"; }  # TAG|CATEGORIA|dettaglio

low=$(echo "$CLEAN" | tr 'A-Z' 'a-z')
detail=$(echo "$CLEAN" | grep -iE "not implemented|tool error|internal error|ram size too low|does not fit|E[0-9]{3}\(" | head -1 | sed 's/^ *//;s/  */ /g' | cut -c1-160)

if echo "$low" | grep -q "allowzero"; then
    verdict "$TAG" "ARTIFACT" "allowzero (export FP32 bug) - $detail"
elif echo "$low" | grep -q "nonmaxsuppression"; then
    verdict "$TAG" "ARTIFACT" "NMS nel grafo (post-processing) - $detail"
elif echo "$low" | grep -qE "malformed|shape inference fails|list index out of range|non-numerical|could not be broadcast|unkonwn layer format|unknown layer format"; then
    verdict "$TAG" "ARTIFACT" "gen onnx/tflite sporca - $detail"
elif echo "$low" | grep -qE "ram size too low|does not fit"; then
    verdict "$TAG" "RAM" "$detail"
elif echo "$low" | grep -q "not implemented"; then
    verdict "$TAG" "LAYER" "$detail"
elif echo "$low" | grep -qE "internal error|tool error|e[0-9]{3}\("; then
    verdict "$TAG" "COMPILER" "$detail"
elif echo "$low" | grep -qE "report generated|complete|generated .*network|ram size|c-model"; then
    ram=$(echo "$CLEAN" | grep -iE "activations|ram" | head -3 | tr '\n' ' ' | cut -c1-160)
    verdict "$TAG" "OK" "compila. $ram"
else
    verdict "$TAG" "UNKNOWN" "$(echo "$CLEAN" | tail -3 | tr '\n' ' ' | cut -c1-160)"
fi
