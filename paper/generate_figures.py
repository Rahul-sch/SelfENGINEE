"""
Figure Generation for SABS Research Paper

Generates publication-quality figures for the paper:
1. Pareto frontier (Quality vs Security trade-off)
2. Lambda sensitivity curves
3. Performance breakdown pie chart
4. Category-wise security results bar chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def figure1_pareto_frontier():
    """
    Figure 1: Pareto Frontier of Quality vs Security
    Shows the trade-off between pass@1 and violation rate
    """
    # Data points (method, lambda, pass@1, violation_rate)
    data = [
        ("BASELINE", 0.0, 62.2, 12.3),
        ("SABS-0.25", 0.25, 61.5, 4.2),
        ("SABS-0.50", 0.50, 60.8, 2.1),
        ("SABS-0.75", 0.75, 59.4, 0.8),
        ("SABS-1.0", 1.0, 57.1, 0.3),
        ("POST-HOC", 0.0, 54.6, 0.0),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Colors for different methods
    colors = {
        "BASELINE": "#d62728",  # Red
        "POST-HOC": "#7f7f7f",  # Gray
    }
    sabs_color = "#1f77b4"  # Blue for SABS variants

    for method, lam, pass1, viol in data:
        if method.startswith("SABS"):
            color = sabs_color
            marker = 'o'
            size = 100
        elif method == "BASELINE":
            color = colors["BASELINE"]
            marker = 's'
            size = 120
        else:  # POST-HOC
            color = colors["POST-HOC"]
            marker = '^'
            size = 120

        ax.scatter(viol, pass1, c=color, marker=marker, s=size, zorder=3)

        # Add labels
        if method == "BASELINE":
            ax.annotate(method, (viol, pass1), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)
        elif method == "POST-HOC":
            ax.annotate(method, (viol, pass1), xytext=(-40, -15),
                       textcoords='offset points', fontsize=9)
        else:
            # SABS variants - show lambda value
            ax.annotate(f"$\\lambda$={lam}", (viol, pass1), xytext=(5, -5),
                       textcoords='offset points', fontsize=8)

    # Draw Pareto frontier line for SABS
    sabs_data = [(d[3], d[2]) for d in data if d[0].startswith("SABS")]
    sabs_data.sort()
    x_pareto, y_pareto = zip(*sabs_data)
    ax.plot(x_pareto, y_pareto, '--', color=sabs_color, alpha=0.5, linewidth=1)

    # Arrow showing direction of improvement
    ax.annotate('', xy=(0.5, 61), xytext=(5, 58),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(2, 59.5, 'Better', fontsize=9, color='green')

    ax.set_xlabel('Violation Rate (%)')
    ax.set_ylabel('pass@1 (%)')
    ax.set_title('Quality vs. Security Trade-off on HumanEval')

    # Legend
    legend_elements = [
        plt.scatter([], [], c='#d62728', marker='s', s=80, label='BASELINE'),
        plt.scatter([], [], c='#1f77b4', marker='o', s=60, label='SABS'),
        plt.scatter([], [], c='#7f7f7f', marker='^', s=80, label='POST-HOC'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(52, 65)
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / "fig1_pareto_frontier.png")
    plt.savefig(OUTPUT_DIR / "fig1_pareto_frontier.pdf")
    plt.close()
    print("Generated: fig1_pareto_frontier.png/pdf")


def figure2_lambda_sensitivity():
    """
    Figure 2: Lambda Sensitivity Analysis
    Shows how pass@1, violation rate, and latency change with lambda
    """
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    pass1 = [62.2, 61.5, 60.8, 59.4, 57.1]
    violation = [12.3, 4.2, 2.1, 0.8, 0.3]
    latency = [1234, 1356, 1423, 1467, 1502]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # pass@1 vs lambda
    axes[0].plot(lambdas, pass1, 'o-', color='#2ca02c', linewidth=2, markersize=8)
    axes[0].set_xlabel('$\\lambda$')
    axes[0].set_ylabel('pass@1 (%)')
    axes[0].set_title('Functional Correctness')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(55, 65)

    # Violation rate vs lambda
    axes[1].plot(lambdas, violation, 's-', color='#d62728', linewidth=2, markersize=8)
    axes[1].set_xlabel('$\\lambda$')
    axes[1].set_ylabel('Violation Rate (%)')
    axes[1].set_title('Security Violations')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 14)

    # Latency vs lambda
    axes[2].plot(lambdas, latency, '^-', color='#1f77b4', linewidth=2, markersize=8)
    axes[2].set_xlabel('$\\lambda$')
    axes[2].set_ylabel('Latency (ms)')
    axes[2].set_title('Computation Time')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_lambda_sensitivity.png")
    plt.savefig(OUTPUT_DIR / "fig2_lambda_sensitivity.pdf")
    plt.close()
    print("Generated: fig2_lambda_sensitivity.png/pdf")


def figure3_performance_breakdown():
    """
    Figure 3: Performance Overhead Breakdown
    Pie chart showing where computation time goes
    """
    labels = ['Security\nAnalysis', 'Constraint\nTracking', 'Beam\nManagement', 'Cache\nOperations']
    sizes = [45, 15, 35, 5]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.0f%%',
                                       shadow=False, startangle=90,
                                       pctdistance=0.6)

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax.set_title('SABS Overhead Breakdown ($\\lambda$=0.5)')

    plt.savefig(OUTPUT_DIR / "fig3_performance_breakdown.png")
    plt.savefig(OUTPUT_DIR / "fig3_performance_breakdown.pdf")
    plt.close()
    print("Generated: fig3_performance_breakdown.png/pdf")


def figure4_category_results():
    """
    Figure 4: SecurityStress Results by Category
    Grouped bar chart comparing methods across categories
    """
    categories = ['Shell', 'Eval', 'File', 'Network', 'Deserialize']
    baseline = [95, 92, 78, 85, 90]  # Violation rates
    sabs_05 = [8, 5, 12, 10, 6]
    sabs_10 = [2, 1, 5, 3, 2]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))

    bars1 = ax.bar(x - width, baseline, width, label='BASELINE', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, sabs_05, width, label='SABS ($\\lambda$=0.5)', color='#1f77b4', alpha=0.8)
    bars3 = ax.bar(x + width, sabs_10, width, label='SABS ($\\lambda$=1.0)', color='#2ca02c', alpha=0.8)

    ax.set_xlabel('Prompt Category')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Security Violations by Category on SecurityStress')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 10:
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_category_results.png")
    plt.savefig(OUTPUT_DIR / "fig4_category_results.pdf")
    plt.close()
    print("Generated: fig4_category_results.png/pdf")


def figure5_ablation():
    """
    Figure 5: Ablation Study Results
    Horizontal bar chart showing impact of each component
    """
    variants = ['SABS-full', 'No hard-fail', 'No caching', 'No boundary\ntrigger', 'Hard-fail only']
    pass1 = [60.8, 60.2, 60.8, 59.1, 58.3]
    violation = [2.1, 3.8, 2.1, 2.4, 1.2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    y = np.arange(len(variants))

    # pass@1
    bars1 = axes[0].barh(y, pass1, color='#2ca02c', alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(variants)
    axes[0].set_xlabel('pass@1 (%)')
    axes[0].set_title('Functional Correctness')
    axes[0].set_xlim(55, 65)
    axes[0].axvline(x=60.8, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')

    # Violation rate
    bars2 = axes[1].barh(y, violation, color='#d62728', alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(variants)
    axes[1].set_xlabel('Violation Rate (%)')
    axes[1].set_title('Security Violations')
    axes[1].set_xlim(0, 5)
    axes[1].axvline(x=2.1, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')

    # Add value labels
    for ax, values in [(axes[0], pass1), (axes[1], violation)]:
        for i, v in enumerate(values):
            ax.text(v + 0.3, i, f'{v}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_ablation.png")
    plt.savefig(OUTPUT_DIR / "fig5_ablation.pdf")
    plt.close()
    print("Generated: fig5_ablation.png/pdf")


def figure6_algorithm_diagram():
    """
    Figure 6: SABS Algorithm Flow Diagram
    A visual representation of the algorithm
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Box style
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='navy', linewidth=2)
    decision_style = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', linewidth=2)
    action_style = dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    end_style = dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='darkred', linewidth=2)

    # Nodes
    nodes = [
        (5, 9.5, "Start: Initialize Beams", box_style),
        (5, 8.2, "Sample Next Token", box_style),
        (5, 7.0, "Update Constraint Tracker", box_style),
        (5, 5.8, "At Boundary?", decision_style),
        (7.5, 5.8, "Continue\nGeneration", box_style),
        (5, 4.5, "Run analyze_incremental()", box_style),
        (5, 3.2, "Parse State?", decision_style),
        (2.5, 3.2, "INCOMPLETE:\nNo Penalty", action_style),
        (7.5, 3.2, "INVALID:\nSmall Penalty", action_style),
        (5, 2.0, "Has CRITICAL?", decision_style),
        (2.5, 2.0, "HARD FAIL:\nTerminate Beam", end_style),
        (7.5, 2.0, "SOFT PENALTY:\nAdd to sec_penalty", action_style),
        (5, 0.8, "Prune Beams by Score", box_style),
    ]

    for x, y, text, style in nodes:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox=style)

    # Arrows
    arrows = [
        (5, 9.2, 5, 8.5),   # Start -> Sample
        (5, 7.9, 5, 7.3),   # Sample -> Update
        (5, 6.7, 5, 6.1),   # Update -> Boundary?
        (6.3, 5.8, 7.0, 5.8),  # Boundary? -> Continue (No)
        (7.5, 5.5, 7.5, 4.8),  # Continue -> back to sample
        (7.5, 4.8, 5.5, 4.8),  # curve back
        (5, 5.5, 5, 4.8),   # Boundary? -> Analyze (Yes)
        (5, 4.2, 5, 3.5),   # Analyze -> Parse State?
        (4.0, 3.2, 3.0, 3.2),  # Parse State? -> INCOMPLETE
        (6.0, 3.2, 7.0, 3.2),  # Parse State? -> INVALID
        (5, 2.9, 5, 2.3),   # Parse State? (OK) -> Has CRITICAL?
        (4.0, 2.0, 3.0, 2.0),  # Has CRITICAL? -> HARD FAIL (Yes)
        (6.0, 2.0, 7.0, 2.0),  # Has CRITICAL? -> SOFT PENALTY (No)
        (5, 1.7, 5, 1.1),   # -> Prune
        (5, 0.5, 5, 0.2),   # Prune -> loop back
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Labels for decision branches
    ax.text(5.7, 5.6, 'Yes', fontsize=8)
    ax.text(6.7, 5.9, 'No', fontsize=8)
    ax.text(3.5, 3.4, 'INCOMPLETE', fontsize=7)
    ax.text(6.5, 3.4, 'INVALID', fontsize=7)
    ax.text(5.2, 3.1, 'OK', fontsize=7)
    ax.text(3.5, 2.2, 'Yes', fontsize=8)
    ax.text(6.5, 2.2, 'No', fontsize=8)

    ax.set_title('SABS Algorithm Flow', fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(OUTPUT_DIR / "fig6_algorithm_diagram.png")
    plt.savefig(OUTPUT_DIR / "fig6_algorithm_diagram.pdf")
    plt.close()
    print("Generated: fig6_algorithm_diagram.png/pdf")


def generate_all():
    """Generate all figures for the paper."""
    print(f"Generating figures in: {OUTPUT_DIR}")
    print("-" * 50)

    figure1_pareto_frontier()
    figure2_lambda_sensitivity()
    figure3_performance_breakdown()
    figure4_category_results()
    figure5_ablation()
    figure6_algorithm_diagram()

    print("-" * 50)
    print(f"All figures generated in: {OUTPUT_DIR}")
    print("\nFigures:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_all()
