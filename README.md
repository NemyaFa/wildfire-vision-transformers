# Détection par télédétection satellitaire de zones forestières à risque d'incendie (ViT vs Swin Transformer)

Projet d'application des architectures **Vision Transformers** à l'imagerie satellitaire pour la classification de zones forestières affectées ou vulnérables aux incendies (`wildfire` vs `nowildfire`).

## Objectifs
- Fine-tuner et comparer deux architectures Transformers de vision : **ViT-Base** (`google/vit-base-patch16-224`) et **Swin-Tiny** (`microsoft/swin-tiny-patch4-window7-224`).
- Évaluer leurs performances sur le *Wildfire Prediction Dataset* (Accuracy, Précision, Rappel, F1-Score, ROC-AUC et latence d'inférence).
- Interpréter les décisions du modèle via des cartes d'attention (*Attention Maps*) et une analyse d'erreurs.

## Structure du dépôt

    wildfire-vision-transformers/
    ├── notebooks/      # Notebook Colab d'entraînement et d'évaluation
    ├── results/        # Figures générées (courbes, matrices de confusion, cartes d'attention)
    ├── requirements.txt
    └── README.md

## Exécution rapide (Google Colab)
1. Ouvrir le notebook dans `notebooks/` sur Google Colab avec un runtime GPU (A100 ou L4 recommandé).
2. Installer les dépendances (`pip install -r requirements.txt`).
3. Exécuter les cellules séquentiellement : le dataset est téléchargé automatiquement via `kagglehub`.

## Résultats
*(Tableau comparatif ViT-Base vs Swin-Tiny à compléter après l'exécution sur Colab)*
