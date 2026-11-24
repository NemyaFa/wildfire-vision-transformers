# Détection par télédétection satellitaire de zones forestières à risque d'incendie (ViT vs Swin Transformer)

Projet d'application des architectures **Vision Transformers** à l'imagerie satellitaire pour la classification binaire de zones forestières affectées ou vulnérables aux incendies (`wildfire` vs `nowildfire`).

## Objectifs
- Fine-tuner et comparer deux paradigmes de Vision Transformers : **ViT-Base** (`google/vit-base-patch16-224`, attention globale) et **Swin-Tiny** (`microsoft/swin-tiny-patch4-window7-224`, attention hiérarchique par fenêtres décalées).
- Évaluer leurs performances sur le *Wildfire Prediction Dataset* (Accuracy, Précision, Rappel, F1-Score, ROC-AUC et latence d'inférence).
- Interpréter les décisions via des cartes d'attention (*Attention Rollout*) et une analyse qualitative des erreurs.

## Résultats sur le jeu de test

| Modèle | Paramètres (M) | Accuracy (%) | Précision (%) | Rappel (%) | F1-Score (%) | ROC-AUC (%) | Latence (ms/img) | Temps Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT-Base** | 85.8 | 99.30 | **99.71** | 99.02 | 99.37 | **99.98** | 1.49 | 447.0 |
| **Swin-Tiny** | **27.5** | **99.57** | 99.60 | **99.63** | **99.61** | **99.98** | **1.16** | **322.4** |

**Enseignement principal :** `Swin-Tiny` surpasse `ViT-Base` sur le F1-Score et le Rappel (critère prioritaire pour limiter les faux négatifs en prévention d'incendies), tout en étant **3,1× plus compact** et **22 % plus rapide** en inférence.

## Structure du dépôt

    wildfire-vision-transformers/
    ├── notebooks/
    │   └── wildfire_vit_vs_swin.ipynb   # Notebook Colab complet
    ├── results/
    │   ├── dataset_samples.png          # Échantillon du dataset
    │   ├── training_curves.png          # Courbes Loss / Val F1
    │   ├── confusion_matrices.png       # Matrices de confusion (ViT vs Swin)
    │   ├── vit_attention_maps.png       # Cartes d'attention (Attention Rollout)
    │   ├── error_analysis.png           # Faux positifs et faux négatifs
    │   └── metrics_summary.csv          # Métriques exportées
    ├── requirements.txt
    └── README.md

## Reproductibilité (Google Colab)
1. Ouvrir `notebooks/wildfire_vit_vs_swin.ipynb` dans Google Colab avec un runtime GPU.
2. Exécuter les cellules séquentiellement (`SEED = 42`) : le dataset est téléchargé automatiquement via `kagglehub`.
