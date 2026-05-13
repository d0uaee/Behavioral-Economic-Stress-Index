"""
Visualiseur rapide — Affiche tous les graphiques générés
Permet de vérifier la qualité des visualisations avant la présentation
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Chemins
ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "outputs" / "figures"

# Liste des graphiques
GRAPHS = [
    "01_besi_vs_ipc.png",
    "02_besi_components.png",
    "03_structural_break_2022.png",
    "04_correlation_lags.png",
    "05_stress_heatmap.png",
    "06_distribution_stats.png",
    "07_summary_statistics.png",
    "08_boxplots_comparison.png",
]

def display_all_graphs():
    """Affiche tous les graphiques dans des fenêtres séparées"""
    print("\n" + "="*80)
    print("🖼️  AFFICHAGE DE TOUS LES GRAPHIQUES — PROJET BESI MAROC")
    print("="*80 + "\n")
    
    for i, graph in enumerate(GRAPHS, 1):
        graph_path = FIGURES / graph
        
        if graph_path.exists():
            print(f"[{i}/8] Affichage : {graph}")
            
            fig, ax = plt.subplots(figsize=(14, 8))
            img = mpimg.imread(graph_path)
            ax.imshow(img)
            ax.axis('off')
            fig.suptitle(f"Graphique {i}/8 : {graph.replace('.png', '').replace('_', ' ').title()}", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠ Fichier non trouvé : {graph_path}")
    
    print("\n" + "="*80)
    print("✅ Tous les graphiques ont été affichés")
    print("="*80 + "\n")


def list_graphs_info():
    """Liste les graphiques avec informations"""
    print("\n" + "="*80)
    print("📋 LISTE DES GRAPHIQUES GÉNÉRÉS")
    print("="*80 + "\n")
    
    descriptions = {
        "01_besi_vs_ipc.png": "Évolution temporelle du BESI vs IPC (1970-2024)",
        "02_besi_components.png": "Décomposition des 4 composants du BESI (Trends, Reddit, YouTube, IPC)",
        "03_structural_break_2022.png": "Analyse de la rupture structurelle 2022 (avant/après)",
        "04_correlation_lags.png": "Corrélation croisée et lead time (capacité d'alerte précoce)",
        "05_stress_heatmap.png": "Calendrier du stress par année et mois (heatmap)",
        "06_distribution_stats.png": "Distributions statistiques du BESI et IPC (histogrammes)",
        "07_summary_statistics.png": "Tableau récapitulatif de toutes les statistiques descriptives",
        "08_boxplots_comparison.png": "Box plots comparatifs avant/après 2022",
    }
    
    for i, graph in enumerate(GRAPHS, 1):
        graph_path = FIGURES / graph
        exists = "✓" if graph_path.exists() else "✗"
        desc = descriptions.get(graph, "Description indisponible")
        
        print(f"{i}. {exists} {graph}")
        print(f"   └─ {desc}\n")
    
    print("="*80)
    print(f"📂 Emplacement : {FIGURES}\n")


if __name__ == "__main__":
    # Afficher la liste
    list_graphs_info()
    
    # Option pour afficher les graphiques
    choice = input("\n🎨 Voulez-vous afficher tous les graphiques ? (o/n) : ").strip().lower()
    if choice == 'o':
        display_all_graphs()
    else:
        print("Affichage annulé. Les fichiers PNG sont prêts pour la présentation.")
