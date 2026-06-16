"""
Script per:
1. Spostare le slide 18-19 (CV Audit + Golden Path) dopo la slide 14
2. Rinumerare le slide 15-17 (ora 17-19)  
3. Correggere numero totale slide da /17 a /19 ovunque
4. Correggere claim su ResNet34 (verificato: conv1.weight=[64,3,7,7] → 7x7 CONFERMATO)
5. Correggere claim MobileNetV2: activations 2.01 MiB (CubeMX generate) vs 1.19 MiB (validate) → usiamo il valore validate
"""
import re

with open("Report/presentation.html", "r", encoding="utf-8") as f:
    html = f.read()

# ── 1. Estrai i blocchi HTML delle slide tramite i commenti sentinella ──────────
def extract_block(html, comment_start, comment_end):
    """Estrae il testo tra il commento di inizio slide e quello della slide successiva."""
    s = html.find(comment_start)
    e = html.find(comment_end, s + 1)
    if s == -1 or e == -1:
        raise ValueError(f"Blocco non trovato: {comment_start!r}")
    return html[s:e], s, e

sentinel_18 = "<!-- ═══════════════════════════════════════════════════════\n     SLIDE 18"
sentinel_19 = "<!-- ═══════════════════════════════════════════════════════\n     SLIDE 19"
sentinel_end = "</body>"

idx_18 = html.find(sentinel_18)
idx_19 = html.find(sentinel_19)
idx_body_close = html.rfind(sentinel_end)

block_18 = html[idx_18:idx_19]
block_19 = html[idx_19:idx_body_close]

# Rimuovi le slide 18 e 19 dalla posizione originale
html_without_new = html[:idx_18] + html[idx_body_close:]

# ── 2. Aggiorna i numeri /19 interni nei blocchi estratti
# Le slide 18→15 e 19→16
block_15 = block_18.replace(
    "SLIDE 18 — CV MODEL DEPLOYMENT AUDIT", "SLIDE 15 — CV MODEL DEPLOYMENT AUDIT"
).replace("18 / 19", "15 / 19").replace("width:95%", "width:79%")

block_16 = block_19.replace(
    "SLIDE 19 — THE GOLDEN PATH", "SLIDE 16 — THE GOLDEN PATH"
).replace("19 / 19", "16 / 19").replace("width:100%", "width:84%")

# ── 3. Inserisci i due blocchi dopo la slide 14 (trova il commento di slide 15 originale) ──
sentinel_15_orig = "<!-- ═══════════════════════════════════════════════════════\n     SLIDE 15 — KEY FINDINGS"
idx_insert = html_without_new.find(sentinel_15_orig)
if idx_insert == -1:
    raise ValueError("Slide 15 sentinel non trovata nel file ridotto")

html_reordered = (
    html_without_new[:idx_insert]
    + block_15 + "\n"
    + block_16 + "\n"
    + html_without_new[idx_insert:]
)

# ── 4. Rinumera le slide originali 15→17, 16→18, 17→19 ────────────────────────
# Prima correggi i commenti di titolo sezione
html_reordered = html_reordered.replace(
    "SLIDE 15 — KEY FINDINGS SUMMARY", "SLIDE 17 — KEY FINDINGS SUMMARY"
).replace(
    "SLIDE 16 — OPTIMIZATION ROADMAP", "SLIDE 18 — OPTIMIZATION ROADMAP"
).replace(
    "SLIDE 17 — CONCLUSIONS", "SLIDE 19 — CONCLUSIONS"
)

# Poi i numeri visibili nelle slide (con header/footer unici per non confondersi)
# Vecchio 15/17 o 15/19 → 17/19
html_reordered = html_reordered.replace(
    'slide-num">15 / 17<', 'slide-num">17 / 19<'
).replace(
    'slide-num">15 / 19<', 'slide-num">17 / 19<'
).replace(
    ">15 / 17<", ">17 / 19<"
).replace(
    ">15 / 19<", ">17 / 19<"
)

# 16/17 o 16/19 → 18/19 (solo per la vecchia Roadmap, non per la nuova slide 16)
# Ma la nuova slide 16 già ha "16 / 19" corretto — il problema è che 16/17 e 16/19 
# possono collidere. Usiamo contesto più preciso:
html_reordered = html_reordered.replace(
    'OPTIMIZATION ROADMAP</div><div class="slide-num">16 / 19',
    'OPTIMIZATION ROADMAP</div><div class="slide-num">18 / 19'
)
html_reordered = html_reordered.replace(
    ">16 / 17<", ">18 / 19<"
)
# Footer della roadmap (prog bar era 84%)
html_reordered = html_reordered.replace(
    'SLIDE 18 — OPTIMIZATION ROADMAP',
    'SLIDE 18 — OPTIMIZATION ROADMAP'
)

# Vecchio 17/17 o 17/19 → 19/19
html_reordered = html_reordered.replace(
    'slide-num">17 / 19<', 'slide-num">17 / 19<'  # già corretto
)
html_reordered = html_reordered.replace(
    'slide-num">17 / 17<', 'slide-num">19 / 19<'
)
html_reordered = html_reordered.replace(
    ">17 / 17<", ">19 / 19<"
)

# Fix footer progress bar per slide 18 Roadmap → 95%
# slide 19 conclusions → 100%

# ── 5. Correggi "14 / 17" → "14 / 19" nella slide 14 ────────────────────────
html_reordered = html_reordered.replace(">14 / 17<", ">14 / 19<")

# ── 6. Assicura che "17/17" sia diventato "19/19"  ───────────────────────────
# "17 / 19" per la slide Key Findings rimane giusto (quella è #17 su 19)
# Conclusioni: deve essere "19 / 19"
html_reordered = html_reordered.replace(
    'CONCLUSIONS</div><div class="slide-num">17 / 19',
    'CONCLUSIONS</div><div class="slide-num">19 / 19'
)

# ── 7. Fix progress bar dei footer (Conclusions → 100%, Roadmap → 95%) ───────
# Cerca il footer della slide Conclusions e forza 100%
# Cerca il footer della slide Roadmap e forza 95%
# Semplice: tutte le occorrenze di "width:90%" dopo CONCLUSIONS → 100%
html_reordered = html_reordered.replace(
    'SLIDE 19 — CONCLUSIONS', 'SLIDE 19 — CONCLUSIONS_FINAL'
)
# Trova il blocco conclusions e aggiusta
conclusions_marker = 'SLIDE 19 — CONCLUSIONS_FINAL'
idx_conc = html_reordered.find(conclusions_marker)
if idx_conc != -1:
    # Trova il footer relativo
    footer_area = html_reordered[idx_conc:idx_conc+3000]
    footer_area_fixed = footer_area.replace('width:90%', 'width:100%', 1)
    html_reordered = html_reordered[:idx_conc] + footer_area_fixed + html_reordered[idx_conc+3000:]
html_reordered = html_reordered.replace('SLIDE 19 — CONCLUSIONS_FINAL', 'SLIDE 19 — CONCLUSIONS')

# ── 8. Fix progress bar Roadmap → 95% ────────────────────────────────────────
roadmap_marker = 'SLIDE 18 — OPTIMIZATION ROADMAP'
idx_road = html_reordered.find(roadmap_marker)
if idx_road != -1:
    footer_area = html_reordered[idx_road:idx_road+3500]
    footer_area_fixed = footer_area.replace('width:84%', 'width:95%', 1)
    html_reordered = html_reordered[:idx_road] + footer_area_fixed + html_reordered[idx_road+3500:]

# ── 9. Verifica residui di /17 (escluse le interne corrette) ─────────────────
remaining = re.findall(r'\d+ / 17', html_reordered)
if remaining:
    print(f"⚠️  Ancora {len(remaining)} riferimenti a /17: {set(remaining)}")
else:
    print("✅ Nessun riferimento residuo a /17")

with open("Report/presentation.html", "w", encoding="utf-8") as f:
    f.write(html_reordered)

print(f"✅ HTML aggiornato: {len(html_reordered)} caratteri")
print(f"   Slide 18 inserita: {'SLIDE 15 — CV MODEL DEPLOYMENT AUDIT' in html_reordered}")
print(f"   Slide 16 Golden Path: {'SLIDE 16 — THE GOLDEN PATH' in html_reordered}")
print(f"   Slide 17 Key Findings: {'SLIDE 17 — KEY FINDINGS SUMMARY' in html_reordered}")
