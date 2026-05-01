
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


print("Loading embedding model (downloads ~90 MB on first run)...")
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model ready. Embedding dimension: 384\n")
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("Run:  pip install sentence-transformers")
    sys.exit(1)


def embed(text: str) -> np.ndarray:
    """Return a 384-dim unit-normed embedding for any word or phrase."""
    return MODEL.encode([text.strip()], normalize_embeddings=True)[0].astype(np.float32)



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are unit-normed, so cosine sim == dot product
    return float(np.dot(a, b))


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project N x D array to N x 2 via power-iteration PCA."""
    mean = vectors.mean(axis=0)
    C = vectors - mean

    def power_iter(M, iters=120):
        u = M.sum(axis=0)
        nm = np.linalg.norm(u)
        u = u / nm if nm > 1e-12 else np.random.default_rng(42).standard_normal(M.shape[1])
        u /= np.linalg.norm(u)
        for _ in range(iters):
            nu = M.T @ (M @ u)
            nm = np.linalg.norm(nu)
            if nm < 1e-12:
                break
            u = nu / nm
        return u

    pc1 = power_iter(C)
    resid = C - np.outer(C @ pc1, pc1)
    pc2 = power_iter(resid)
    return np.column_stack([C @ pc1, C @ pc2])



def interpret(sim: float) -> str:
    if sim > 0.95: return "nearly identical"
    if sim > 0.85: return "very similar"
    if sim > 0.70: return "quite similar"
    if sim > 0.50: return "somewhat similar / related"
    if sim > 0.30: return "loosely related"
    if sim > 0.10: return "little relation"
    if sim > -0.10: return "essentially unrelated"
    return "dissimilar / contrasting"


def sim_color(sim: float) -> str:
    if sim > 0.70: return "#1D9E75"
    if sim > 0.30: return "#EF9F27"
    return "#D85A30"


CONTEXT_WORDS = [
    "king", "queen", "man", "woman", "cat", "dog",
    "paris", "france", "london", "happy", "sad",
    "fast", "slow", "computer", "phone", "water",
    "fire", "good", "bad", "love", "hate", "science",
    "music", "run", "sleep", "big", "small",
]

_context_cache: dict = {}

def get_context_vecs():
    if not _context_cache:
        print("  (embedding context words for 2D plot...)")
        for w in CONTEXT_WORDS:
            _context_cache[w] = embed(w)
    return _context_cache


def plot(word1: str, word2: str, v1: np.ndarray, v2: np.ndarray, sim: float):
    ctx = get_context_vecs()

    all_words = list(ctx.keys())
    for w in [word1, word2]:
        if w not in ctx:
            all_words.append(w)

    all_vecs = np.array([ctx.get(w, embed(w)) for w in all_words])
    proj = pca_2d(all_vecs)

    i1 = all_words.index(word1)
    i2 = all_words.index(word2)

    COLORS = ["#1D9E75", "#D85A30"]
    BG = "#F8F7F4"

    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    fig.suptitle(
        "Word Vector Embedding Explorer  |  all-MiniLM-L6-v2  (384 dim)",
        fontsize=14, fontweight="bold", color="#2C2C2A", y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35,
                  left=0.06, right=0.97, top=0.91, bottom=0.07)


    for idx, (word, vec, col) in enumerate([(word1, v1, COLORS[0]),
                                             (word2, v2, COLORS[1])]):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor("#F0EEE8")
        top_idx = np.argsort(np.abs(vec))[::-1][:20]
        vals = vec[top_idx]
        bar_cols = [col if v >= 0 else "#888780" for v in vals]
        ax.barh(range(20), vals[::-1], color=bar_cols[::-1], height=0.72, edgecolor="none")
        ax.set_yticks(range(20))
        ax.set_yticklabels([f"d{i:03d}" for i in top_idx[::-1]], fontsize=7, fontfamily="monospace")
        ax.axvline(0, color="#B4B2A9", linewidth=0.8)
        ax.set_title(f'"{word}"\n384-dim embedding (top 20 by magnitude)', fontsize=9, color="#444441", pad=5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#D3D1C7")


    ax_sim = fig.add_subplot(gs[0, 2])
    ax_sim.set_facecolor("#F0EEE8")
    ax_sim.set_xlim(-1.15, 1.15)
    ax_sim.set_ylim(-0.35, 1.45)
    ax_sim.axis("off")

    ax_sim.add_patch(mpatches.FancyBboxPatch(
        (-1, 0.52), 2, 0.20, boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#D3D1C7", zorder=1))
    ax_sim.add_patch(mpatches.FancyBboxPatch(
        (-1, 0.52), sim + 1, 0.20, boxstyle="round,pad=0.02",
        linewidth=0, facecolor=sim_color(sim), zorder=2))

    ax_sim.text(0, 1.22, f"{sim:.4f}", ha="center", va="center",
                fontsize=30, fontweight="bold", color="#2C2C2A")
    ax_sim.text(0, 1.00, interpret(sim), ha="center", va="center",
                fontsize=10, color="#5F5E5A")
    ax_sim.text(0, 0.40, "cosine similarity", ha="center",
                fontsize=8, color="#888780", style="italic")
    for x, lbl in [(-1, "-1\nopposite"), (0, "0\northog."), (1, "+1\nidentical")]:
        ax_sim.text(x, 0.32, lbl, ha="center", fontsize=7, color="#888780")

    angle_deg = float(np.degrees(np.arccos(np.clip(sim, -1, 1))))
    ax_sim.text(0, 0.08, f"angle: {angle_deg:.2f} degrees", ha="center",
                fontsize=9, color="#444441", fontfamily="monospace")
    ax_sim.text(0, -0.08, "model: all-MiniLM-L6-v2", ha="center",
                fontsize=7.5, color="#888780", fontfamily="monospace")
    ax_sim.text(0, -0.22, "dims: 384  |  unit-normed", ha="center",
                fontsize=7.5, color="#888780", fontfamily="monospace")
    ax_sim.set_title("Cosine Similarity", fontsize=9, color="#444441", pad=5)

    
    ax2d = fig.add_subplot(gs[1, :2])
    ax2d.set_facecolor("#F0EEE8")

    dy = (proj[:, 1].max() - proj[:, 1].min()) * 0.015

    for i, (w, (px, py)) in enumerate(zip(all_words, proj)):
        if w in (word1, word2):
            continue
        ax2d.scatter(px, py, s=18, color="#B4B2A9", zorder=2, alpha=0.7)
        ax2d.text(px, py + dy, w, fontsize=6.5, color="#888780",
                  ha="center", va="bottom", fontfamily="monospace")

    ax2d.plot([proj[i1, 0], proj[i2, 0]], [proj[i1, 1], proj[i2, 1]],
              linestyle="--", linewidth=1.3, color="#534AB7", alpha=0.5, zorder=3)

    for i, (word, col) in enumerate([(word1, COLORS[0]), (word2, COLORS[1])]):
        px, py = proj[[i1, i2][i]]
        ax2d.scatter(px, py, s=110, color=col, zorder=5, edgecolors="white", linewidths=1.5)
        ax2d.text(px, py + dy * 1.5, word, fontsize=9.5, color=col,
                  fontweight="bold", ha="center", va="bottom", fontfamily="monospace")

    ax2d.set_title(
        f'2D PCA projection  |  {len(all_words)} reference words  |  "{word1}" vs "{word2}"',
        fontsize=9, color="#444441", pad=5)
    ax2d.set_xlabel("PC 1", fontsize=8, color="#888780")
    ax2d.set_ylabel("PC 2", fontsize=8, color="#888780")
    ax2d.tick_params(labelsize=7.5, colors="#888780")
    ax2d.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax2d.spines[s].set_color("#D3D1C7")


    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    ax_tbl.set_facecolor("#F0EEE8")

    top_n = 12
    diff_idx = np.argsort(np.abs(v1 - v2))[::-1][:top_n]
    rows = [[f"d{d:03d}", f"{v1[d]:+.3f}", f"{v2[d]:+.3f}", f"{v1[d]-v2[d]:+.3f}"]
            for d in diff_idx]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["dim", word1[:8], word2[:8], "delta"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D3D1C7")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#D3D1C7")
            cell.set_text_props(fontweight="bold", color="#2C2C2A")
        else:
            cell.set_facecolor("#F0EEE8" if r % 2 == 0 else "#F8F7F4")
            cell.set_text_props(color="#444441", fontfamily="monospace")
    ax_tbl.set_title(f"Top {top_n} dims by |delta|", fontsize=9, color="#444441", pad=5)

    fname = f"embedding_{word1}_vs_{word2}.png".replace(" ", "_")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  -> saved: {fname}")
    plt.show()



def main():
    print("=" * 58)
    print("  Word Vector Embedding Explorer")
    print("  Model : sentence-transformers/all-MiniLM-L6-v2")
    print("  Dims  : 384   |   Any English word or phrase works!")
    print("=" * 58)
    print()

    while True:
        w1 = input("Enter word 1 (or 'q' to quit): ").strip()
        if w1.lower() == "q":
            break
        if not w1:
            continue
        w2 = input("Enter word 2: ").strip()
        if not w2:
            continue

        print(f"\n  Embedding '{w1}' and '{w2}'...")
        v1 = embed(w1)
        v2 = embed(w2)
        sim = cosine_similarity(v1, v2)
        angle = float(np.degrees(np.arccos(np.clip(sim, -1, 1))))

        print()
        print(f"  +-- Results for '{w1}' vs '{w2}' " + "-" * 20)
        print(f"  |  Cosine similarity : {sim:.4f}")
        print(f"  |  Interpretation    : {interpret(sim)}")
        print(f"  |  Angle (degrees)   : {angle:.2f}")
        print(f"  |  ||{w1}|| norm      : {np.linalg.norm(v1):.4f}")
        print(f"  |  ||{w2}|| norm      : {np.linalg.norm(v2):.4f}")
        print(f"  +" + "-" * 45)
        print()
        print("  Generating plot...")
        plot(w1, w2, v1, v2, sim)
        print()


if __name__ == "__main__":
    main()