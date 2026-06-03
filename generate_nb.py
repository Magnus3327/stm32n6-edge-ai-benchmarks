import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # ---------------------------------------------------------
    # Cell 1: Setup and Imports (Custom Color Palette)
    # ---------------------------------------------------------
    cell_setup = nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from IPython.display import display, display_html

# Set style & palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style='whitegrid')

# Distinct palette colors
CB_PALETTE = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
sns.set_palette(CB_PALETTE)

# Custom color mapping for consistency across all plots
COLOR_MAP = {
    'CubeMX v2.2 (Nucleo)': '#0072B2',
    'AI Studio v4.0 (Nucleo)': '#009E73',
    'DevCloud v2.2 (DK)': '#D55E00',
    'DevCloud v4.0 (DK)': '#CC79A7'
}

# Resolved relative to the notebook's location (Report/)
PLOT_DIR = 'plot'
os.makedirs(PLOT_DIR, exist_ok=True)""")
    
    # ---------------------------------------------------------
    # Cell 2: Data Loading and Name Normalization
    # ---------------------------------------------------------
    cell_load = nbf.v4.new_code_cell("""# Load Baseline files (relative to Report/ directory)
df_base_aistudio = pd.read_csv('../Results/baseline/performance_baseline_results_aistudio.csv')
df_base_cubemx = pd.read_csv('../Results/baseline/performance_baseline_results_cubemx.csv')
df_base_cloud2 = pd.read_csv('../Results/baseline/performance_baseline_results_devcloudaicore2.2.csv')
df_base_cloud4 = pd.read_csv('../Results/baseline/performance_baseline_results_devcloudaicore4.0.csv')

# Load Estimated files (relative to Report/ directory)
df_est_aistudio = pd.read_csv('../Results/estimated/performance_estimated_results_aistudio.csv')
df_est_cubemx = pd.read_csv('../Results/estimated/performance_estimated_results_cubemx.csv')
df_est_cloud2 = pd.read_csv('../Results/estimated/performance_estimated_results_devcloudaicore2.2.csv')
df_est_cloud4 = pd.read_csv('../Results/estimated/performance_estimated_results_devcloudaicore4.0.csv')

# Standardize Model Names across datasets to prevent mismatches
for df in [df_base_aistudio, df_base_cubemx, df_base_cloud2, df_base_cloud4,
           df_est_aistudio, df_est_cubemx, df_est_cloud2, df_est_cloud4]:
    if 'Model Name' in df.columns:
        df['Model Name'] = df['Model Name'].str.replace('conv2d_poll', 'conv2d_pool')

# Data cleaning: convert columns to numeric for baseline data
for df in [df_base_aistudio, df_base_cubemx, df_base_cloud2, df_base_cloud4]:
    if 'Inference Time (ms)' in df.columns:
        df['Inference Time (ms)'] = pd.to_numeric(df['Inference Time (ms)'], errors='coerce')
    if 'RAM Usage (KiB)' in df.columns:
        df['RAM Usage (KiB)'] = pd.to_numeric(df['RAM Usage (KiB)'], errors='coerce')
    if 'Flash Usage (KiB)' in df.columns:
        df['Flash Usage (KiB)'] = pd.to_numeric(df['Flash Usage (KiB)'], errors='coerce')

# Map raw model names to formatted labels for presentation
def get_clean_name(model_name):
    if not isinstance(model_name, str):
        return model_name
    name = model_name.replace('baseline_', '').replace('_int8', '')
    if name == 'conv2d_pool':
        return 'Conv2D + MaxPool'
    if name == 'depthwise_3x3':
        return 'Depthwise Conv2D 3x3'
    if name == 'conv2d_3x3':
        return 'Conv2D 32f 3x3'
    if name == 'conv2d_16f_3x3':
        return 'Conv2D 16f 3x3'
    if name == 'conv2d_64f_3x3':
        return 'Conv2D 64f 3x3'
    if name == 'conv2d_1x1':
        return 'Conv2D 32f 1x1'
    if name == 'conv2d_32f_5x5':
        return 'Conv2D 32f 5x5'
    if name == 'conv2d_32f_7x7':
        return 'Conv2D 32f 7x7'
    return name""")

    # ---------------------------------------------------------
    # Cell 3: Tested Models & Platforms
    # ---------------------------------------------------------
    cell_md_platforms = nbf.v4.new_markdown_cell("""## 1. Tested Models and Platforms
This section describes the single-layer models selected for benchmarking and the hardware/software configurations tested.

### Benchmark Platforms & Configurations:
- **AI Studio v4.0 (Nucleo):** STM32 AI Studio running **AI Core v4.0** compiler on the STM32N6 Nucleo hardware board.
- **CubeMX v2.2 (Nucleo):** STM32CubeMX with X-CUBE-AI running **AI Core v2.2** compiler on the STM32N6 Nucleo hardware board.
- **DevCloud v2.2 (DK):** ST DevCloud running **AI Core v2.2** compiler on the STM32N6 Discovery Kit (DK) board.
- **DevCloud v4.0 (DK):** ST DevCloud running **AI Core v4.0** compiler on the STM32N6 Discovery Kit (DK) board.""")

    cell_platforms = nbf.v4.new_code_cell("""# Compile Table of Tested Models
overview_models = df_base_aistudio[['Model Name', 'Layer Type', 'Filters', 'Kernel Size', 'Input Shape', 'Quantization', 'MACC']].copy()
overview_models['Clean Name'] = overview_models['Model Name'].apply(get_clean_name)

# Reorder columns
overview_models = overview_models[['Model Name', 'Clean Name', 'Layer Type', 'Filters', 'Kernel Size', 'Input Shape', 'Quantization', 'MACC']]

# Style and display
styled_models = overview_models.style.set_properties(**{
    'text-align': 'center',
    'font-size': '11pt',
    'border': '1px solid lightgrey'
}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).set_caption("Summary of Tested Models and Operators")

display(styled_models)""")

    # ---------------------------------------------------------
    # Cell 4: Inference Time Table & Bar Plot
    # ---------------------------------------------------------
    cell_md_inference = nbf.v4.new_markdown_cell("""## 2. Inference Time Comparison (Table & Bar Plot)
This section evaluates the execution speed across all four compilers and platforms. 
- **FAIL** denotes hardware timeout, out of memory, or model compilation failure.
- The plots use highly distinct, custom colors for visual clarity.""")

    cell_inference = nbf.v4.new_code_cell("""# Consolidated Inference Time Table
dfs_base = {
    'CubeMX v2.2 (Nucleo)': df_base_cubemx,
    'AI Studio v4.0 (Nucleo)': df_base_aistudio,
    'DevCloud v2.2 (DK)': df_base_cloud2,
    'DevCloud v4.0 (DK)': df_base_cloud4
}

inference_data = []
all_models = df_base_aistudio['Model Name'].unique()

for m in all_models:
    row_data = {'Model': get_clean_name(m)}
    for tool_name, df in dfs_base.items():
        row_match = df[df['Model Name'] == m]
        if not row_match.empty:
            val = row_match.iloc[0]['Inference Time (ms)']
            row_data[tool_name] = val
        else:
            row_data[tool_name] = np.nan
    inference_data.append(row_data)

df_inference_table = pd.DataFrame(inference_data).set_index('Model')

# Reorder columns
col_order_tools = ['CubeMX v2.2 (Nucleo)', 'AI Studio v4.0 (Nucleo)', 'DevCloud v2.2 (DK)', 'DevCloud v4.0 (DK)']
df_inference_table = df_inference_table.reindex(columns=col_order_tools)

# Sort index logically by model size
model_order = [
    'Depthwise Conv2D 3x3', 'Conv2D 16f 3x3', 'Conv2D + MaxPool',
    'Conv2D 32f 1x1', 'Conv2D 32f 3x3', 'Conv2D 32f 5x5',
    'Conv2D 64f 3x3', 'Conv2D 32f 7x7'
]
df_inference_table = df_inference_table.reindex([m for m in model_order if m in df_inference_table.index])

# Formatted table for printing
df_inference_print = df_inference_table.apply(lambda col: col.map(lambda x: f"{x:.3f} ms" if not pd.isna(x) else "FAIL"))

styled_inference = df_inference_print.style.set_properties(**{
    'text-align': 'center',
    'font-size': '11pt',
    'border': '1px solid lightgrey'
}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#2980b9'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).set_caption("Measured Inference Time (ms) across Compilers and Boards")

display(styled_inference)

# Plotting the inference times as a bar plot
plt.figure(figsize=(14, 7))
data_list = []
for tool, df in dfs_base.items():
    for _, row in df.iterrows():
        clean_model = get_clean_name(row['Model Name'])
        val = row['Inference Time (ms)']
        data_list.append({
            'Tool': tool,
            'Model Name': clean_model,
            'Inference Time (ms)': val
        })
df_all = pd.DataFrame(data_list)

sns.barplot(data=df_all, x='Model Name', y='Inference Time (ms)', hue='Tool', palette=COLOR_MAP, order=model_order)
plt.title('Inference Time (ms) per Layer across Tools & Platforms', fontweight='bold', fontsize=14)
plt.ylabel('Inference Time (ms) [Linear Scale]', fontsize=12)
plt.xlabel('Layer Name', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Platform & Tool', frameon=True)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/11_inference_time_bar_plot.png', dpi=300)
plt.show()""")

    # ---------------------------------------------------------
    # Cell 5: Side-by-Side RAM Comparison
    # ---------------------------------------------------------
    cell_md_ram = nbf.v4.new_markdown_cell("""## 3. Memory Footprint: Compiler Estimated vs. Actual Hardware RAM Usage
Here we compare the **Estimated RAM** against the **Actual RAM** using distinct color scales:
- **Logarithmic scale** is used to handle large variations (e.g. from 12 KiB to 1548 KiB).
- **Shared Y-axis scale** is applied to directly expose discrepancies, highlighting cases where compilers severely underestimate actual memory limits (e.g. 7x7 layer on CubeMX).""")

    cell_ram = nbf.v4.new_code_cell("""# Prepare RAM data
ram_data_est = []
ram_data_act = []

dfs_est = {
    'CubeMX v2.2 (Nucleo)': df_est_cubemx,
    'AI Studio v4.0 (Nucleo)': df_est_aistudio,
    'DevCloud v2.2 (DK)': df_est_cloud2,
    'DevCloud v4.0 (DK)': df_est_cloud4
}

for m in all_models:
    clean_m = get_clean_name(m)
    
    # Estimated
    est_row = {'Model': clean_m}
    for tool_name, df in dfs_est.items():
        row_match = df[df['Model Name'] == m]
        val = row_match.iloc[0]['Estimated RAM Usage (KiB)'] if not row_match.empty else np.nan
        est_row[tool_name] = val
    ram_data_est.append(est_row)
    
    # Actual
    act_row = {'Model': clean_m}
    for tool_name, df in dfs_base.items():
        row_match = df[df['Model Name'] == m]
        val = row_match.iloc[0]['RAM Usage (KiB)'] if not row_match.empty else np.nan
        act_row[tool_name] = val
    ram_data_act.append(act_row)

df_ram_est = pd.DataFrame(ram_data_est).set_index('Model').reindex(columns=col_order_tools)
df_ram_act = pd.DataFrame(ram_data_act).set_index('Model').reindex(columns=col_order_tools)

df_ram_est = df_ram_est.reindex([m for m in model_order if m in df_ram_est.index])
df_ram_act = df_ram_act.reindex([m for m in model_order if m in df_ram_act.index])

# Plotting side-by-side with custom colors
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
colors_ram = [COLOR_MAP[col] for col in df_ram_est.columns]

df_ram_est.plot(kind='bar', ax=axes[0], width=0.8, color=colors_ram)
axes[0].set_title('Estimated RAM Usage (KiB) - Compiler Prediction', fontweight='bold', fontsize=12)
axes[0].set_ylabel('RAM (KiB) [Log Scale]', fontsize=11)
axes[0].set_yscale('log')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
axes[0].grid(True, which="both", ls="--", alpha=0.5)

df_ram_act.plot(kind='bar', ax=axes[1], width=0.8, color=colors_ram)
axes[1].set_title('Actual RAM Usage (KiB) - Measured on Hardware', fontweight='bold', fontsize=12)
axes[1].set_yscale('log')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
axes[1].grid(True, which="both", ls="--", alpha=0.5)

# Set same y-limit
max_val = max(df_ram_est.max().max(), df_ram_act.max().max())
axes[0].set_ylim(1, max_val * 1.5)
axes[1].set_ylim(1, max_val * 1.5)

plt.suptitle('RAM Footprint: Compiler Estimated vs. Actual Hardware Usage (Log Scale)', fontweight='bold', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/9_ram_estimated_vs_actual_side_by_side.png', dpi=300)
plt.show()

# Display tables stacked vertically for proper PDF formatting (avoid horizontal clipping)
html_est = df_ram_est.style.set_caption("Estimated RAM Usage (KiB)").format(precision=2).set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#8e44ad'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).to_html()

html_act = df_ram_act.style.set_caption("Actual RAM Usage (KiB)").format(precision=2).set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).to_html()

display_html(html_est, raw=True)
display_html("<br>", raw=True)
display_html(html_act, raw=True)""")

    # ---------------------------------------------------------
    # Cell 6: Side-by-Side MACC Comparison
    # ---------------------------------------------------------
    cell_md_macc = nbf.v4.new_markdown_cell("""## 4. Operation Complexity: Compiler Estimated vs. Actual Hardware MACC
Below is the comparison of **Estimated MACC** vs. **Actual MACC** count for all models across compilers. Linear scale is used for readability.""")

    cell_macc = nbf.v4.new_code_cell("""# Prepare MACC data
macc_data_est = []
macc_data_act = []

for m in all_models:
    clean_m = get_clean_name(m)
    
    # Estimated
    est_row = {'Model': clean_m}
    for tool_name, df in dfs_est.items():
        row_match = df[df['Model Name'] == m]
        val = row_match.iloc[0]['Estimated MACC'] if not row_match.empty else np.nan
        est_row[tool_name] = pd.to_numeric(val, errors='coerce')
    macc_data_est.append(est_row)
    
    # Actual
    act_row = {'Model': clean_m}
    for tool_name, df in dfs_base.items():
        row_match = df[df['Model Name'] == m]
        val = row_match.iloc[0]['MACC'] if not row_match.empty else np.nan
        act_row[tool_name] = pd.to_numeric(val, errors='coerce')
    macc_data_act.append(act_row)

df_macc_est = pd.DataFrame(macc_data_est).set_index('Model').reindex(columns=col_order_tools)
df_macc_act = pd.DataFrame(macc_data_act).set_index('Model').reindex(columns=col_order_tools)

df_macc_est = df_macc_est.reindex([m for m in model_order if m in df_macc_est.index])
df_macc_act = df_macc_act.reindex([m for m in model_order if m in df_macc_act.index])

# Plotting MACC side-by-side with custom colors
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
colors_macc = [COLOR_MAP[col] for col in df_macc_est.columns]

df_macc_est.plot(kind='bar', ax=axes[0], width=0.8, color=colors_macc)
axes[0].set_title('Estimated MACC (Compiler)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('MAC Operations [Linear Scale]', fontsize=11)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
axes[0].grid(True, which="both", ls="--", alpha=0.5)

df_macc_act.plot(kind='bar', ax=axes[1], width=0.8, color=colors_macc)
axes[1].set_title('Actual MACC (Hardware)', fontweight='bold', fontsize=12)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
axes[1].grid(True, which="both", ls="--", alpha=0.5)

# Set same y-limit
max_macc = max(df_macc_est.max().max(), df_macc_act.max().max())
axes[0].set_ylim(0, max_macc * 1.1)
axes[1].set_ylim(0, max_macc * 1.1)

plt.suptitle('Operator Complexity: Estimated vs. Actual MACC (Linear Scale)', fontweight='bold', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/10_macc_estimated_vs_actual_side_by_side.png', dpi=300)
plt.show()

# Display tables stacked vertically for proper PDF formatting
html_macc_est = df_macc_est.style.set_caption("Estimated MACC").format(precision=0).set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#16a085'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).to_html()

html_macc_act = df_macc_act.style.set_caption("Actual MACC").format(precision=0).set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).to_html()

display_html(html_macc_est, raw=True)
display_html("<br>", raw=True)
display_html(html_macc_act, raw=True)""")

    # ---------------------------------------------------------
    # Cell 7: Hardware Epochs Table
    # ---------------------------------------------------------
    cell_md_epochs = nbf.v4.new_markdown_cell("""## 5. Hardware Epochs Partitioning
On the STM32N6 NPU, models can be split into multiple "hardware epochs" to fit within the internal SRAM. A higher number of epochs indicates that intermediate activations or weights had to be tiled and swapped back and forth. The table below lists the epoch partitioning strategy for each tool/compiler.""")

    cell_epochs = nbf.v4.new_code_cell("""# Extract hardware epochs from notes
epochs_data = []

def parse_epochs(notes):
    if pd.isna(notes):
        return 'N/A'
    notes_lower = str(notes).lower()
    if 'fail' in notes_lower or 'timeout' in notes_lower:
        return 'FAIL'
    
    if '1 single' in notes_lower or '1 hardware' in notes_lower or 'single hardware' in notes_lower:
        return '1'
    if '2' in notes_lower:
        return '2'
    if '3' in notes_lower:
        return '3'
    if '4' in notes_lower:
        return '4'
        
    import re
    nums = re.findall(r'\\d+', notes)
    if nums:
        return nums[0]
        
    return notes

for tool, df in dfs_base.items():
    for _, row in df.iterrows():
        clean_model = get_clean_name(row['Model Name'])
        epoch_str = parse_epochs(row['Notes'])
        epochs_data.append({
            'Tool': tool,
            'Model Name': clean_model,
            'Hardware Epochs': epoch_str
        })

df_epochs = pd.DataFrame(epochs_data)
df_epochs_pivot = df_epochs.pivot(index='Model Name', columns='Tool', values='Hardware Epochs')
df_epochs_pivot = df_epochs_pivot.reindex(columns=col_order_tools)
df_epochs_pivot = df_epochs_pivot.reindex(index=[c for c in model_order if c in df_epochs_pivot.index])

styled_epochs = df_epochs_pivot.style.set_properties(**{
    'text-align': 'center',
    'font-size': '11pt',
    'border': '1px solid lightgrey'
}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#d35400'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).set_caption("Hardware Epochs Partitioning by Layer & Tool")

display(styled_epochs)""")

    # ---------------------------------------------------------
    # Cell 8: Detailed Platform Statistics & Distribution Plots
    # ---------------------------------------------------------
    cell_md_stats = nbf.v4.new_markdown_cell("""## 6. Comprehensive Platform Statistics & Distributions
This section provides a detailed statistical comparison to identify the overall fastest and most memory-efficient compiler and hardware platform.
- **Summary Tables:** Tables computing Count, Mean, Std Dev, Variance, Min, Medians, Percentiles (25%, 75%, 90%), Max, and Success Rate.
- **Side-by-Side Distribution Box Plots:** Displays the actual variability, standard deviation, and any performance outliers for both Latency (linear scale) and RAM Memory allocation (logarithmic scale) with custom palettes and individual data points overlaid.""")

    cell_stats = nbf.v4.new_code_cell("""# Compute detailed statistics for successful runs
stats_time_list = []
stats_ram_list = []

for tool, df in dfs_base.items():
    times = df['Inference Time (ms)'].dropna()
    rams = df['RAM Usage (KiB)' ].dropna()
    
    # Calculate Latency Stats
    t_stats = times.describe()
    t_stats['variance'] = times.var()
    t_stats['90%'] = times.quantile(0.9)
    t_stats['success_rate (%)'] = (df['Inference Time (ms)'].notna().sum() / len(df)) * 100
    stats_time_list.append(pd.DataFrame(t_stats).rename(columns={'Inference Time (ms)': tool}))
    
    # Calculate RAM Stats
    r_stats = rams.describe()
    r_stats['variance'] = rams.var()
    r_stats['90%'] = rams.quantile(0.9)
    stats_ram_list.append(pd.DataFrame(r_stats).rename(columns={'RAM Usage (KiB)': tool}))

df_stats_time = pd.concat(stats_time_list, axis=1).reindex(columns=col_order_tools)
df_stats_ram = pd.concat(stats_ram_list, axis=1).reindex(columns=col_order_tools)

stats_rename = {
    'count': 'Successful Runs (count)',
    'mean': 'Mean / Media (avg)',
    'std': 'Std Dev / Deviaz. Std',
    'variance': 'Variance / Varianza',
    'min': 'Minimum / Minimo',
    '25%': '25th Percentile (P25)',
    '50%': 'Median / Mediana (P50)',
    '75%': '75th Percentile (P75)',
    '90%': '90th Percentile (P90)',
    'max': 'Maximum / Massimo',
    'success_rate (%)': 'Success Rate (%)'
}

df_stats_time.rename(index=stats_rename, inplace=True)
df_stats_ram.rename(index=stats_rename, inplace=True)

# Styled HTML tables displayed stacked vertically for proper PDF width compliance
html_stats_time = df_stats_time.style.set_caption("Latency (Inference Time) Summary Statistics").format(precision=4).set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).to_html()

html_stats_ram = df_stats_ram.style.set_caption("RAM Memory Allocation Summary Statistics").format(precision=2).set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]}
]).to_html()

display_html(html_stats_time, raw=True)
display_html("<br>", raw=True)
display_html(html_stats_ram, raw=True)

# Plotting side-by-side boxplots for distributions
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Latency Boxplot (linear scale)
sns.boxplot(
    data=df_all.dropna(subset=['Inference Time (ms)']),
    x='Tool',
    y='Inference Time (ms)',
    ax=axes[0],
    palette=COLOR_MAP,
    order=col_order_tools
)
sns.stripplot(
    data=df_all.dropna(subset=['Inference Time (ms)']),
    x='Tool',
    y='Inference Time (ms)',
    ax=axes[0],
    color='black',
    alpha=0.6,
    size=6,
    jitter=0.15,
    order=col_order_tools
)
axes[0].set_title('Inference Time Distribution (ms) - Lower is Faster', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Inference Time (ms)', fontsize=11)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha='right')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Prepare RAM list for plotting
ram_plot_data = []
for tool, df in dfs_base.items():
    for _, row in df.iterrows():
        ram_plot_data.append({
            'Tool': tool,
            'RAM Usage (KiB)': row['RAM Usage (KiB)']
        })
df_ram_all = pd.DataFrame(ram_plot_data)

# RAM Boxplot (log scale for high range visibility)
sns.boxplot(
    data=df_ram_all.dropna(subset=['RAM Usage (KiB)']),
    x='Tool',
    y='RAM Usage (KiB)',
    ax=axes[1],
    palette=COLOR_MAP,
    order=col_order_tools
)
sns.stripplot(
    data=df_ram_all.dropna(subset=['RAM Usage (KiB)']),
    x='Tool',
    y='RAM Usage (KiB)',
    ax=axes[1],
    color='black',
    alpha=0.6,
    size=6,
    jitter=0.15,
    order=col_order_tools
)
axes[1].set_title('RAM Allocation Distribution (KiB) - Lower is More Efficient', fontweight='bold', fontsize=12)
axes[1].set_ylabel('RAM (KiB) [Log Scale]', fontsize=11)
axes[1].set_yscale('log')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha='right')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle('Platform Benchmark Distributions: Latency vs. Memory Footprint', fontweight='bold', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/12_latency_vs_ram_distributions.png', dpi=300)
plt.show()""")

    # Assemble the notebook cells
    nb['cells'] = [
        cell_setup,
        cell_load,
        cell_md_platforms,
        cell_platforms,
        cell_md_inference,
        cell_inference,
        cell_md_ram,
        cell_ram,
        cell_md_macc,
        cell_macc,
        cell_md_epochs,
        cell_epochs,
        cell_md_stats,
        cell_stats
    ]
    
    # Save the notebook to the Report folder
    os.makedirs('Report', exist_ok=True)
    with open('Report/NPU_Performance_Analysis_Updated.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print("Notebook 'Report/NPU_Performance_Analysis_Updated.ipynb' generated successfully!")

if __name__ == '__main__':
    create_notebook()