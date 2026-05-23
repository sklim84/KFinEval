#!/usr/bin/env python3
"""
Toxicity Evaluation Results Visualization Script

Creates:
1. Radar charts for each category with A-G checklist items
2. Radar chart for attack methods with mean scores
3. Bar charts for toxicity_levels and score_distribution
"""

import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform


class _SnsShim:
    @staticmethod
    def color_palette(name, n):
        if name == "husl":
            return [plt.cm.hsv(i / n) for i in range(n)]
        if name == "viridis":
            return [plt.cm.viridis(i / max(n - 1, 1)) for i in range(n)]
        return [plt.cm.tab10(i % 10) for i in range(n)]


sns = _SnsShim()


# Set up Korean font
def setup_korean_font():
    """Configure matplotlib to display Korean characters."""
    if platform.system() == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=path
                ).get_name()
                break
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()

# JSON file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(
    SCRIPT_DIR, "..", "eval", "_results", "3_fin_toxicity"
)
JSON_FILES = [
    "3_fin_toxicity_google_gemini-3-flash-preview_eval_stats.json",
    "3_fin_toxicity_gpt-4.1_eval_stats.json",
    "3_fin_toxicity_claude-haiku-4-5_eval_stats.json",
    "3_fin_toxicity_claude-sonnet-4-5_eval_stats.json",
    "3_fin_toxicity_Ministral-3-14B-Instruct-2512_eval_stats.json",
    "3_fin_toxicity_Mistral-Small-3.2-24B-Instruct-2506_eval_stats.json",
    "3_fin_toxicity_gpt-oss-20b_eval_stats.json",
    "3_fin_toxicity_EXAONE-4.0-1.2B_eval_stats.json",
    "3_fin_toxicity_gpt-5.2_eval_stats.json",
    "3_fin_toxicity_gpt-5.2_reasoning_eval_stats.json",
    "3_fin_toxicity_gpt-5-mini_eval_stats.json",
    "3_fin_toxicity_gpt-5-mini_reasoning_eval_stats.json",
]

# Short model names for display
MODEL_SHORT_NAMES = {
    "google_gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-4.1": "GPT-4.1",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "Ministral-3-14B-Instruct-2512": "Ministral 3 14B",
    "Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small 24B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "EXAONE-4.0-1.2B": "EXAONE 4.0 1.2B",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2_reasoning": "GPT-5.2 (Reas)",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-mini_reasoning": "GPT-5 Mini (Reas)",
}

# Colors for each model
MODEL_COLORS = sns.color_palette("husl", len(JSON_FILES))

# Checklist item labels (A-G)
CHECKLIST_LABELS = [
    "A_명시적_거절",
    "B_법적_경고",
    "C_출처없는_주장",
    "D_실행가능_방법",
    "E_긴급성_공포",
    "F_정치편향",
    "G_수락의사",
]

CHECKLIST_SHORT = [
    "Explicit\nRefusal",
    "Legal\nWarning",
    "Uncited\nClaim",
    "Actionable\nMethod",
    "Urgency/Fear",
    "Political\nBias",
    "Acceptance",
]


def load_all_data():
    """Load all JSON files and return data dict."""
    data = {}
    for fname in JSON_FILES:
        fpath = os.path.join(BASE_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
            model_name = d.get("model", fname.replace(".json", ""))
            data[model_name] = d
    return data


def create_radar_chart(ax, labels, values_dict, title, show_legend=True):
    """Create a radar chart on the given axes."""
    num_vars = len(labels)
    # 각도 계산 (라디안)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_with_closure = angles + angles[:1]

    ax.set_theta_offset(np.pi / 2)  # 시작 위치를 12시 방향으로
    ax.set_theta_direction(-1)  # 시계 방향으로 진행
    ax.set_xticks(angles)

    # 먼저 라벨을 설정합니다.
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")

    # --- 라벨 회전 로직 추가 ---
    # tick_params의 pad를 조절해 원과의 간격을 띄웁니다.
    ax.tick_params(axis="x", pad=13)

    labels_objs = ax.get_xticklabels()
    for i, label in enumerate(labels_objs):
        # 현재 항목의 각도 (라디안 -> 도)
        angle_rad = angles[i]
        angle_deg = np.rad2deg(angle_rad)

        # 텍스트가 거꾸로 뒤집히지 않게 처리 (하단 부분은 180도 회전)
        if 90 < angle_deg < 270:
            rotation = 180 - angle_deg
        else:
            rotation = -angle_deg

        label.set_rotation(rotation)
        # 라벨의 정렬 방향을 중앙으로 설정하여 원을 따라 깔끔하게 배치
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")
    # --------------------------

    lines = []
    for idx, (model_name, values) in enumerate(values_dict.items()):
        vals = values + values[:1]
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        (line,) = ax.plot(
            angles_with_closure,
            vals,
            "-",
            linewidth=1.5,
            label=short_name,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )
        lines.append(line)
        ax.fill(
            angles_with_closure,
            vals,
            alpha=0.1,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=40)  # 제목 간격도 약간 넓힘

    return lines


CATEGORY_TRANSLATION = {
    "불법 부정행위 조언": "Illegal/Unethical Advice",
    "허위정보 생성": "Misinformation Generation",
    "정치 선동 / 여론 조작": "Political Agitation",
    "공포 불안 조장": "Fearmongering",
}


def plot_category_radar_charts(all_data):
    """Create radar charts for each category (1x4 layout) with tighter spacing."""
    first_model = list(all_data.values())[0]
    categories = [cat["category"] for cat in first_model["by_category"]]

    # 1. 전체 피규어 너비를 줄입니다 (32 -> 22)
    fig = plt.figure(figsize=(22, 6))
    from matplotlib.gridspec import GridSpec

    # 2. wspace를 작게 설정하여 그래프 간 간격을 좁힙니다.
    # width_ratios에서 범례 공간(마지막 열) 비중을 조절합니다.

    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.5], wspace=0.5)

    csv_data = []

    for cat_idx, category in enumerate(categories):
        ax = fig.add_subplot(gs[0, cat_idx], polar=True)
        values_dict = {}

        for model_name, model_data in all_data.items():
            cat_data = next(
                (
                    cat
                    for cat in model_data["by_category"]
                    if cat["category"] == category
                ),
                None,
            )
            if cat_data:
                checklist = cat_data["checklist_stats"]
                values = [
                    checklist.get(label, {}).get("Y_count", 0)
                    for label in CHECKLIST_LABELS
                ]
                values_dict[model_name] = values

                # Add to CSV data
                row = {
                    "Category": CATEGORY_TRANSLATION.get(category, category),
                    "Model": MODEL_SHORT_NAMES.get(model_name, model_name),
                }
                for label, val in zip(CHECKLIST_LABELS, values):
                    row[label] = val
                csv_data.append(row)

        # 레이더 차트 생성 (이전의 라벨 회전 로직이 적용된 create_radar_chart 호출)
        create_radar_chart(
            ax,
            CHECKLIST_SHORT,
            values_dict,
            f"{CATEGORY_TRANSLATION.get(category, category)}",
            show_legend=False,
        )

    # 범례 영역
    ax_legend = fig.add_subplot(gs[0, 4])
    ax_legend.axis("off")

    legend_handles = []
    for idx, model_name in enumerate(all_data.keys()):
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        handle = plt.Line2D(
            [0],
            [0],
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            linewidth=3,
            label=short_name,
        )
        legend_handles.append(handle)

    ax_legend.legend(
        handles=legend_handles,
        loc="center left",
        fontsize=12,
        title="Model",
        frameon=True,
    )

    # 3. tight_layout 대신 구체적인 여백 조정을 사용하거나,
    # tight_layout 사용 시 pad를 줄입니다.
    plt.tight_layout(pad=1.0)

    plt.savefig("toxicity_category_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: toxicity_category_radar.png")

    # Save to CSV
    csv_file = "toxicity_radar_data.csv"
    fieldnames = ["Category", "Model"] + CHECKLIST_LABELS
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Saved: {csv_file}")


def plot_toxicity_distribution_bar_chart(all_data):
    """Create stacked bar chart for toxicity distribution."""
    # Get toxicity levels keys
    first_model = list(all_data.values())[0]
    toxicity_levels = list(first_model["toxicity_levels"].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    tox_short = [
        "Full Defense",
        "Passive\nRefusal",
        "Indirect\nHint",
        "Specific\nInfo",
        "Immediate\nExec",
    ]

    # Prepare data for stacked bar chart
    model_names = list(all_data.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m) for m in model_names]

    data = np.zeros((len(toxicity_levels), len(model_names)))  # (levels, models)

    csv_bar_data = []
    for i, model_name in enumerate(model_names):
        model_data = all_data[model_name]
        row = {"Model": MODEL_SHORT_NAMES.get(model_name, model_name)}
        for j, level in enumerate(toxicity_levels):
            percentage = model_data["toxicity_levels"][level]["percentage"]
            data[j, i] = percentage
            row[tox_short[j]] = percentage
        csv_bar_data.append(row)

    # Plot stacked bars
    bottom = np.zeros(len(model_names))
    colors = sns.color_palette("viridis", len(toxicity_levels))

    for i, level in enumerate(toxicity_levels):
        ax.bar(
            short_names,
            data[i],
            bottom=bottom,
            label=tox_short[i],
            color=colors[i],
            width=0.6,
        )
        bottom += data[i]

    ax.set_title(
        "Distribution by Toxicity Level", fontsize=25, fontweight="bold", pad=20
    )
    ax.set_ylabel("Percentage (%)", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, title="Toxicity Level"
    )
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            SCRIPT_DIR,
            "toxicity_distribution_bar.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: toxicity_distribution_bar.png")

    # Save to CSV
    csv_file = "toxicity_bar_data.csv"
    fieldnames = ["Model"] + tox_short
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_bar_data)
    print(f"Saved: {csv_file}")


def main():
    """Main function to generate all visualizations."""
    print("Loading data...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} models: {list(all_data.keys())}")

    print("\nGenerating visualizations...")
    plot_category_radar_charts(all_data)
    plot_toxicity_distribution_bar_chart(all_data)

    print("\nAll visualizations saved to:", BASE_DIR)


if __name__ == "__main__":
    main()
