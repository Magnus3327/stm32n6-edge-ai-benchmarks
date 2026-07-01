#!/bin/bash
# Testa UN modello su TUTTI i profili memoria STM32N6 e stampa una riga TSV per profilo.
# Uso: st_matrix.sh <modello.onnx|.tflite> <nome> [profili...]
# Output TSV: nome  profilo  VERDETTO  attivazioni  unalloc  dettaglio
# Verdetti:
#   OK            compila e alloca -> candidato board (con quella memoria)
#   RAM_NOFIT     E103/unallocated: non entra in quella memoria (limite fisico ESIBIBILE)
#   LAYER         NOT IMPLEMENTED: op non supportato (limite compiler ESIBIBILE)
#   COMPILER      INTERNAL/TOOL ERROR: bug/limite compiler (ESIBIBILE, segnalare a ST)
#   ARTIFACT      allowzero/NMS/malformed/broadcast: da FIXARE (export)
ST=/Applications/ST/STEdgeAI/4.0
CFG="$ST/scripts/N6_scripts/user_neuralart.json"
BIN="$ST/Utilities/mac/stedgeai"
MODEL="$1"; NAME="$2"; shift 2
PROFILES=("$@")
if [ ${#PROFILES[@]} -eq 0 ]; then
  PROFILES=(n6-noextmem n6-extram n6-extflash n6-nointmem n6-allmems-O1 n6-allmems-O2 n6-allmems-O3 n6-allmems-Oauto profile_O3)
fi

for P in "${PROFILES[@]}"; do
  OUT="/tmp/st_matrix/${NAME}__${P}"
  mkdir -p "$OUT"
  "$BIN" analyze --model "$MODEL" --target stm32n6 --st-neural-art "${P}@${CFG}" -o "$OUT" > "$OUT/log.txt" 2>&1
  C=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/log.txt")
  low=$(echo "$C" | tr 'A-Z' 'a-z')
  act=$(echo "$C" | grep -iE "activations \(rw\)" | head -1 | grep -oE "[0-9,]+ B \([0-9.]+ MiB\)" | head -1)
  unalloc=$(echo "$C" | grep -iE "unallocated" | grep -oE "unallocated=[0-9]+" | head -1 | cut -d= -f2)
  det=$(echo "$C" | grep -iE "not implemented|tool error|internal error|E[0-9]{3}\(|too low|does not fit" | head -1 | sed 's/^ *//;s/  */ /g' | cut -c1-120)

  if echo "$low" | grep -q "allowzero"; then V=ARTIFACT
  elif echo "$low" | grep -q "nonmaxsuppression"; then V=ARTIFACT
  elif echo "$low" | grep -qE "malformed|shape inference fails|list index out of range|non-numerical|could not be broadcast|unkonwn layer format|unknown layer format"; then V=ARTIFACT
  elif echo "$low" | grep -qE "unallocated|ram size too low|does not fit"; then V=RAM_NOFIT
  elif echo "$low" | grep -q "not implemented"; then V=LAYER
  elif echo "$low" | grep -qE "internal error|tool error|e[0-9]{3}\("; then V=COMPILER
  elif echo "$C" | grep -qiE "report file|report generated|ram \(total\)"; then V=OK
  else V=UNKNOWN; fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$P" "$V" "${act:-–}" "${unalloc:-–}" "${det:-–}"
done
