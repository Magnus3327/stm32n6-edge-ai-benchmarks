#!/bin/bash
# Matrice completa: tutti i modelli x tutti i profili N6. Concorrenza 4, parallel-safe.
# Output: /tmp/st_matrix/results.tsv
ROOT="/Users/matteo/Universita/Tesi/Baseline Performance/Modelli"
PRES="$ROOT/01_Presentazione"           # INT8 CNN baseline (riusati, non rigenerati)
BASE="$ROOT/02_Test_INT4_Transformer_v4.0.1"  # campagna corrente
ONE="$BASE/Models/scripts/st_one.sh"
RES="/tmp/st_matrix/results.tsv"
mkdir -p /tmp/st_matrix
touch "$RES"   # RESUME: non troncare, la skip-logic in st_one.sh evita i doppioni

# 4 profili significativi (gli altri 5 sono ridondanti/edge). Override: RUN_ALL9=1
if [ "${RUN_ALL9:-0}" = "1" ]; then
  PROFILES=(n6-noextmem n6-extram n6-extflash n6-nointmem n6-allmems-O1 n6-allmems-O2 n6-allmems-O3 n6-allmems-Oauto profile_O3)
else
  PROFILES=(n6-noextmem n6-extram n6-extflash n6-allmems-O3)
fi

# nome  ->  path assoluto
declare -a MODELS=(
  # CNN INT8 ONNX (CV_Models: yolo@640, resnet34@224, pidnet@2048x1024, rtmdet@640)
  "yolov7_int8|$PRES/Models/1_ram_fail/int8/yolov7_int8.onnx"
  "resnet34_int8|$PRES/Models/1_ram_fail/int8/resnet34_fixed_int8.onnx"
  "pidnet_s_int8|$PRES/Models/1_ram_fail/int8/pidnet_s_int8.onnx"
  "rtmdet_l_int8|$PRES/Models/1_ram_fail/int8/rtmdet_l_int8.onnx"
  "rtmdet_l_int8_nonms|$BASE/Models/1_ram_fail/rtmdet_l_int8_nonms.onnx"
  "yolov8n_int8|$PRES/Models/1_ram_fail/int8/yolov8n_int8.onnx"
  "yolov11n_int8|$PRES/Models/1_ram_fail/int8/yolov11n_int8.onnx"
  "yolov11s_int8|$PRES/Models/1_ram_fail/int8/yolov11s_int8.onnx"
  "yolov11m_int8|$PRES/Models/1_ram_fail/int8/yolov11m_int8.onnx"
  "yolov11l_int8|$PRES/Models/1_ram_fail/int8/yolov11l_int8.onnx"
  "yolov11x_int8|$PRES/Models/1_ram_fail/int8/yolov11x_int8.onnx"
  # Transformer TFLite NATIVO (litert-torch)
  "mobileclip_s0_img_tfl|$BASE/tflite_native/mobileclip_s0_image_int8.tflite"
  "mobileclip_b_img_tfl|$BASE/tflite_native/mobileclip_b_image_int8.tflite"
  "uniformer_small_tfl|$BASE/tflite_native/uniformer_small_int8.tflite"
  "uniformer_base_tfl|$BASE/tflite_native/uniformer_base_int8.tflite"
  "uniformer_small_qkvsplit_tfl|$BASE/tflite_native/uniformer_small_qkvsplit_int8.tflite"
  # Transformer ONNX (per documentare artefatti export vs nativo)
  "mobileclip_s0_img_onnx|$BASE/Models/Test_v4.0.1/to_windows/fase2_transformer/onnx_int8/mobileclip_s0_image_int8.onnx"
  "mobileclip_b_img_onnx|$BASE/Models/Test_v4.0.1/to_windows/fase2_transformer/onnx_int8/mobileclip_b_image_int8.onnx"
  "uniformer_small_onnx|$BASE/Models/Test_v4.0.1/to_windows/fase2_transformer/onnx_int8/uniformer_small_int8.onnx"
  "uniformer_base_onnx|$BASE/Models/Test_v4.0.1/to_windows/fase2_transformer/onnx_int8/uniformer_base_int8.onnx"
  # baseline transformer block
  "baseline_tf_block|$BASE/Models/Test_v4.0.1/to_windows/fase3_baseline/transformer/baseline_transformer_block_int8.onnx"
)

# loop con gate di concorrenza (robusto a spazi nei path: IFS=tab)
maxjobs=4
run_pairs() {
  for m in "${MODELS[@]}"; do
    name="${m%%|*}"; path="${m#*|}"
    [ -f "$path" ] || { echo "MANCA: $name ($path)" >&2; continue; }
    for p in "${PROFILES[@]}"; do
      printf "%s\t%s\t%s\n" "$path" "$name" "$p"
    done
  done
}
while IFS=$'\t' read -r path name profile; do
  bash "$ONE" "$path" "$name" "$profile" >> "$RES" &
  while [ "$(jobs -r | wc -l)" -ge "$maxjobs" ]; do wait -n; done
done < <(run_pairs)
wait

echo "DONE. Righe: $(wc -l < "$RES")"
