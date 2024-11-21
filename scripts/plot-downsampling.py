benchmark_data = {
    "Avg.": {
        "eval_setting": "",
        "sft_5": 57.69,
        "sft_10": 58.06,
        "sft_25": 58.64,
        "sft_50": 59.18,
        "sft_75": 59.57,
        "sft_full": 60.08
    },
    "MMLU": {
        "eval_setting": "5 shot",
        "sft_5": 64.1,
        "sft_10": 63.9,
        "sft_25": 63.4,
        "sft_50": 62.3,
        "sft_75": 62.1,
        "sft_full": 62.1
    },
    "TruthfulQA": {
        "eval_setting": "6 shot",
        "sft_5": 51.0,
        "sft_10": 50.4,
        "sft_25": 49.9,
        "sft_50": 48.9,
        "sft_75": 46.4,
        "sft_full": 46.8
    },
    "PopQA": {
        "eval_setting": "15 shot",
        "sft_5": 30.8,
        "sft_10": 30.8,
        "sft_25": 29.8,
        "sft_50": 30.1,
        "sft_75": 29.6,
        "sft_full": 29.3
    },
    # TODO: BBH IS NOT UP TO DATE!!!
    "BigBenchHard": {
        "eval_setting": "3 shot, CoT",
        "sft_5": 67.5,
        "sft_10": 68.2,
        "sft_25": 68.5,
        "sft_50": 67.6,
        "sft_75": 69.7,
        "sft_full": 68.8
    },
    "HumanEval": {
        "eval_setting": "pass@10",
        "sft_5": 81.5,
        "sft_10": 81.5,
        "sft_25": 81.4,
        "sft_50": 84.4,
        "sft_75": 86.7,
        "sft_full": 86.2
    },
    "HumanEval+": {
        "eval_setting": "pass@10",
        "sft_5": 76.1,
        "sft_10": 77.4,
        "sft_25": 75.5,
        "sft_50": 78.3,
        "sft_75": 79.5,
        "sft_full": 81.4
    },
    "GSM8K": {
        "eval_setting": "8 shot, CoT",
        "sft_5": 66.0,
        "sft_10": 66.3,
        "sft_25": 72.1,
        "sft_50": 73.8,
        "sft_75": 74.4,
        "sft_full": 76.2
    },
    "DROP": {
        "eval_setting": "3 shot",
        "sft_5": 60.7,
        "sft_10": 60.7,
        "sft_25": 59.4,
        "sft_50": 60.7,
        "sft_75": 59.9,
        "sft_full": 61.3
    },
    "MATH": {
        "eval_setting": "4 shot CoT, Flex",
        "sft_5": 29.3,
        "sft_10": 28.7,
        "sft_25": 30.0,
        "sft_50": 30.9,
        "sft_75": 31.7,
        "sft_full": 31.5
    },
    "IFEval": {
        "eval_setting": "Strict",
        "sft_5": 65.4,
        "sft_10": 68.6,
        "sft_25": 70.6,
        "sft_50": 68.2,
        "sft_75": 70.6,
        "sft_full": 72.8
    },
    "AlpacaEval 2": {
        "eval_setting": "LC % win",
        "sft_5": 11.1,
        "sft_10": 10.2,
        "sft_25": 11.7,
        "sft_50": 13.3,
        "sft_75": 12.4,
        "sft_full": 12.4
    },
    "Safety": {
        "eval_setting": "",
        "sft_5": 89.8,
        "sft_10": 90.9,
        "sft_25": 92.3,
        "sft_50": 92.6,
        "sft_75": 92.8,
        "sft_full": 93.1
    }
}

import matplotlib.pyplot as plt
import numpy as np

# Create x-axis values (SFT percentages)
x_values = [5, 10, 25, 50, 75, 100]  # 100 represents full SFT

# # Create figure and axis with a larger size
# plt.figure(figsize=(12, 8))

# # Color palette for different lines
# colors = plt.cm.tab20(np.linspace(0, 1, len(benchmark_data)))

# # Plot each benchmark
# for (benchmark, data), color in zip(benchmark_data.items(), colors):
#     if benchmark != "Avg.":  # Skip the average for now
#         y_values = [
#             data["sft_5"],
#             data["sft_10"],
#             data["sft_25"],
#             data["sft_50"],
#             data["sft_75"],
#             data["sft_full"]
#         ]
#         plt.plot(x_values, y_values, marker='o', label=benchmark, color=color, linewidth=2)

# # Add the average line with higher emphasis
# avg_values = [
#     benchmark_data["Avg."]["sft_5"],
#     benchmark_data["Avg."]["sft_10"],
#     benchmark_data["Avg."]["sft_25"],
#     benchmark_data["Avg."]["sft_50"],
#     benchmark_data["Avg."]["sft_75"],
#     benchmark_data["Avg."]["sft_full"]
# ]
# plt.plot(x_values, avg_values, 'k--', label='Average', linewidth=3, marker='s')

# # Customize the plot
# plt.xlabel('SFT Training Data Size', fontsize=12)
# plt.ylabel('Performance', fontsize=12)
# plt.title('Benchmark Performance Across Different SFT Percentages', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# # Set x-axis ticks
# plt.xticks(x_values)

# # Adjust layout to prevent label cutoff
# plt.tight_layout()

# # Show the plot
# plt.show()

# Optional: Create a second plot focusing on specific benchmarks of interest
# plt.figure(figsize=(20, 8))

# Define benchmarks and SFT percentages
benchmarks = [
    'Avg.',
    'GSM8K',
    'HumanEval+',
    'Safety',
    'TruthfulQA',
]
sft_percentages = ['5%', '10%', '25%', '50%', '75%', '100%']
# colors = ['#0A2B35', '#0fcb8c', '#105257', '#f0529c', '#838383', '#0a3235']  # One color for each percentage
colors = [
    '#FAC4DD',  # 10%
    '#F8ADD0',  # 20%
    '#F697C3',  # 40%
    '#F480B6',  # 60%
    '#F269A9',  # 80%
    '#F0529C',  # 100% - original pink
]

colors = [
    "#E7EEEE",  # RGB(231, 238, 238)
    "#CEDCDD",  # RGB(206, 220, 221)
    "#B7CBCC",  # RGB(183, 203, 204)
    "#9FB9BB",  # RGB(159, 185, 187)
    "#88A8AB",  # RGB(136, 168, 171)
    "#F0529C", # PINK
    "#6E979A",  # RGB(110, 151, 154)
    "#588689",  # RGB(88, 134, 137)
    "#3F7478",  # RGB(63, 116, 120)
    "#105257",  # RGB(16, 82, 87)
    "#0A3235",  # RGB(10, 50, 53)
]

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 8))

# Width of bars and positions
width = 0.12
n_percentages = len(sft_percentages)

# Create bars for each benchmark
for i, benchmark in enumerate(benchmarks):
    data = benchmark_data[benchmark]
    values = [
        data["sft_5"],
        data["sft_10"],
        data["sft_25"],
        data["sft_50"],
        data["sft_75"],
        data["sft_full"]
    ]
    
    # Calculate positions for this benchmark's group of bars
    x = i
    for j in range(n_percentages):
        bar_position = x - (n_percentages-1)*width/2 + j*width
        bar = ax.bar(bar_position, values[j], width, 
                    label=sft_percentages[j] if i == 0 else "", 
                    color=colors[j],
                    edgecolor="black")
        
        # Add value labels on top of bars
        # ax.text(bar_position, values[j], f'{values[j]:.1f}', ha='center', va='bottom', fontsize=8)

# Customize the plot
# ax.set_xlabel('Benchmarks', fontsize=14)
ax.set_ylabel('Performance', fontsize=18)
plt.tick_params(axis='y', labelsize=18)
# ax.set_title('Performance by Benchmark and SFT Percentage', fontsize=14)

# Set x-axis ticks and labels
ax.set_xticks(range(len(benchmarks)))
ax.set_xticklabels(benchmarks, ha="center", fontsize=18)

ax.spines[["right", "top"]].set_visible(False)

# Add legend
# ax.legend(title='SFT Sample Size', loc='center', bbox_to_anchor=(0.885, 0.8))

# Add grid
# ax.grid(True, linestyle='--', alpha=0.3, axis='y')

# Adjust layout to accommodate legend
# plt.subplots_adjust(right=0.85)

# Save and show the plot
plt.savefig('downsampling_bars.pdf', bbox_inches='tight', dpi=300)
plt.show()

# # Define specific benchmarks and their colors
# plot_config = {
#     'Avg.': '#0a3235',         # Black for average
#     'TruthfulQA': '#b11bE8',   # Coral red
#     'HumanEval+': '#f0529c',   # Turquoise
#     'Safety': '#105257',       # Light blue
#     'GSM8K': '#0fcb8c'         # Sage green
# }

# # Plot each benchmark with its specified color
# for benchmark, color in plot_config.items():
#     data = benchmark_data[benchmark]
#     y_values = [
#         data["sft_5"],
#         data["sft_10"],
#         data["sft_25"],
#         data["sft_50"],
#         data["sft_75"],
#         data["sft_full"]
#     ]
#     # Make average line dashed and thicker
#     if benchmark == 'Avg.':
#         plt.plot(x_values, y_values, '--', marker='s', label=benchmark, 
#                 color=color, linewidth=3)
#     else:
#         plt.plot(x_values, y_values, marker='o', label=benchmark, 
#                 color=color, linewidth=2)

# # Customize the focused plot
# plt.xlabel('SFT Percentage', fontsize=12)
# plt.ylabel('Performance', fontsize=12)
# # plt.title('Selected Benchmark Performance Trends', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.xticks(x_values)

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# # plt.show()

# plt.savefig('downsampling.pdf', bbox_inches='tight', dpi=300)
# plt.close()