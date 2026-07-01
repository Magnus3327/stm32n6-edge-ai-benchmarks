#!/bin/bash
# UN modello + UN profilo -> UNA riga TSV. Workspace isolato (parallel-safe).
# Uso: st_one.sh <modello_abs> <nome> <profilo>
# TSV: nome  profilo  VERDETTO  input_shape  attivazioni  unalloc  dettaglio
ST=/Applications/ST/STEdgeAI/4.0
CFG="$ST/scripts/N6_scripts/user_neuralart.json"
BIN="$ST/Utilities/mac/stedgeai"
MODEL="$1"; NAME="$2"; P="$3"
RES=/tmp/st_matrix/results.tsv
# skip se questa coppia (modello,profilo) e' gia' nei risultati (resume)
if [ -f "$RES" ] && awk -F'\t' -v n="$NAME" -v p="$P" '$1==n&&$2==p{f=1} END{exit !f}' "$RES"; then
  exit 0
fi
OUT="/tmp/st_matrix/${NAME}__${P}"
rm -rf "$OUT"; mkdir -p "$OUT"
# esegui DENTRO OUT cosi' st_ai_ws e' isolato per ogni run (no collisioni in parallelo)
( cd "$OUT" && "$BIN" analyze --model "$MODEL" --target stm32n6 --st-neural-art "${P}@${CFG}" -o "$OUT" ) > "$OUT/log.txt" 2>&1
C=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/log.txt")
low=$(echo "$C" | tr 'A-Z' 'a-z')
ishape=$(echo "$C" | grep -iE "input 1/1" | head -1 | grep -oE "int8\([0-9x]+\)|float[0-9]*\([0-9x]+\)" | head -1)
act=$(echo "$C" | grep -iE "activations \(rw\)" | head -1 | grep -oE "[0-9.]+ MiB" | head -1)
unalloc=$(echo "$C" | grep -iE "unallocated" | grep -oE "unallocated=[0-9]+" | head -1 | cut -d= -f2)
det=$(echo "$C" | grep -iE "not implemented|tool error|internal error|E[0-9]{3}\(|too low|does not fit" | head -1 | sed 's/^ *//;s/  */ /g' | cut -c1-110)

if   echo "$low" | grep -q "allowzero"; then V=ARTIFACT
elif echo "$low" | grep -q "nonmaxsuppression"; then V=ARTIFACT
elif echo "$low" | grep -qE "malformed|shape inference fails|list index out of range|non-numerical|could not be broadcast|unkonwn layer format|unknown layer format"; then V=ARTIFACT
elif echo "$low" | grep -qE "unallocated|ram size too low|does not fit"; then V=RAM_NOFIT
elif echo "$low" | grep -q "not implemented"; then V=LAYER
elif echo "$low" | grep -qE "internal error|tool error|e[0-9]{3}\("; then V=COMPILER
elif echo "$C" | grep -qiE "report file|report generated|ram \(total\)"; then V=OK
else V=UNKNOWN; fi

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$P" "$V" "${ishape:-–}" "${act:-–}" "${unalloc:-–}" "${det:-–}"
