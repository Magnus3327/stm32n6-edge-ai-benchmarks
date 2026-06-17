import re

with open("Report/presentation.html") as f:
    html = f.read()

total = 19

# Trova la posizione di ogni commento SLIDE N
slide_markers = list(re.finditer(r"SLIDE (\d+) \u2014", html))
print("Slide trovate nell'ordine attuale:")
for m in slide_markers:
    print(f"  pos={m.start()}: SLIDE {m.group(1)}")

# Costruiamo un elenco (pos_inizio_blocco, pos_fine_blocco, numero_corretto)
blocks = []
for i, m in enumerate(slide_markers):
    slide_num = int(m.group(1))
    start = m.start()
    end = slide_markers[i+1].start() if i+1 < len(slide_markers) else len(html)
    blocks.append((start, end, slide_num))

# Ora ricostruiamo l'HTML correggendo i numeri blocco per blocco
new_html_parts = []
prev_end = 0

for (start, end, slide_num) in blocks:
    # Parte prima del blocco (non cambia)
    new_html_parts.append(html[prev_end:start])
    block = html[start:end]

    # Correggi slide-num nel header
    block = re.sub(
        r'class="slide-num">\d+ / \d+',
        f'class="slide-num">{slide_num} / {total}',
        block
    )
    # Correggi il <span>N / T</span> nel footer (l'ultimo span numerico)
    # Pattern: <span>X / Y</span> alla fine del footer
    block = re.sub(
        r'<span>(\d+ / \d+)</span>\s*\n\s*</div>\s*\n</div>',
        f'<span>{slide_num} / {total}</span>\n  </div>\n</div>',
        block
    )

    new_html_parts.append(block)
    prev_end = end

# Aggiungi la parte finale dopo l'ultimo blocco
new_html_parts.append(html[prev_end:])

new_html = "".join(new_html_parts)

# Verifica
all_nums = re.findall(r'class="slide-num">(\d+) / (\d+)', new_html)
print("\nSlide-num dopo fix:")
for n, t in all_nums:
    print(f"  {n} / {t}")

with open("Report/presentation.html", "w") as f:
    f.write(new_html)

print(f"\n✅ Salvato. {len(all_nums)} slide-num aggiornati.")
