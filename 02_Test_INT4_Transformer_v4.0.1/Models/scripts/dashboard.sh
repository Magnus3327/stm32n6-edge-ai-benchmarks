#!/bin/bash
# Dashboard live della campagna N6. Eseguila nel TUO terminale:
#   bash Models/scripts/dashboard.sh
# Si aggiorna ogni 2s. Ctrl-C per uscire.
RES=/tmp/st_matrix/results.tsv
KEYP=(n6-noextmem n6-extram n6-allmems-O3)

while true; do
  clear
  echo "══════════════════════════════════════════════════════════════════════"
  echo "  CAMPAGNA STM32N6 — DASHBOARD LIVE      $(date '+%H:%M:%S')"
  echo "══════════════════════════════════════════════════════════════════════"

  # --- job compilatore attivi ---
  running=$(pgrep -fl "atonn|stedgeai" 2>/dev/null | grep -c atonn)
  echo "  atonn (compilatore) attivi ora : $running / 4 worker"
  jobs_matrix=$(pgrep -fl "run_matrix.sh\|st_one.sh" 2>/dev/null | wc -l | tr -d ' ')
  echo "  processi sweep                 : $jobs_matrix"
  echo ""

  # --- progresso matrice ---
  if [ -f "$RES" ]; then
    n=$(wc -l < "$RES" | tr -d ' ')
    echo "  MATRICE COMPILAZIONE           : $n / 180 run"
    ok=$(awk -F'\t' '$3=="OK"{c++}END{print c+0}' "$RES")
    nofit=$(awk -F'\t' '$3=="RAM_NOFIT"{c++}END{print c+0}' "$RES")
    lay=$(awk -F'\t' '$3=="LAYER"{c++}END{print c+0}' "$RES")
    comp=$(awk -F'\t' '$3=="COMPILER"{c++}END{print c+0}' "$RES")
    art=$(awk -F'\t' '$3=="ARTIFACT"{c++}END{print c+0}' "$RES")
    echo "    OK:$ok  RAM_NOFIT:$nofit  LAYER:$lay  COMPILER:$comp  ARTIFACT:$art"
    echo ""
    printf "  %-26s %-9s %-9s %-9s\n" "MODELLO" "interna" "+extRAM" "tutte-O3"
    printf "  %-26s %-9s %-9s %-9s\n" "--------------------------" "-------" "-------" "--------"
    for name in $(cut -f1 "$RES" | sort -u); do
      a=$(awk -F'\t' -v n="$name" '$1==n && $2=="n6-noextmem"{print $3}' "$RES")
      b=$(awk -F'\t' -v n="$name" '$1==n && $2=="n6-extram"{print $3}' "$RES")
      c=$(awk -F'\t' -v n="$name" '$1==n && $2=="n6-allmems-O3"{print $3}' "$RES")
      printf "  %-26s %-9s %-9s %-9s\n" "$name" "${a:-·}" "${b:-·}" "${c:-·}"
    done
  else
    echo "  matrice non ancora avviata (results.tsv assente)"
  fi
  echo "══════════════════════════════════════════════════════════════════════"
  sleep 2
done
