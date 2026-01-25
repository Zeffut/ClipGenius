#!/bin/bash
# Script de build rapide pour l'application Mac ClipGenius
# Utilise py2app pour créer un bundle .app

set -e

echo ""
echo "======================================================"
echo "  ClipGenius - Build Application Mac"
echo "======================================================"
echo ""

# Vérifier que Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

echo "✓ Python 3 installé: $(python3 --version)"

# Vérifier l'environnement virtuel
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Environnement virtuel non activé"
    echo "   Activation recommandée: source .venv/bin/activate"
else
    echo "✓ Environnement virtuel actif: $VIRTUAL_ENV"
fi

# Installer py2app si nécessaire
if ! python3 -c "import py2app" 2>/dev/null; then
    echo "📦 Installation de py2app..."
    pip install py2app
fi

echo "✓ py2app installé"

# Nettoyer les builds précédents
echo ""
echo "🗑️  Nettoyage des builds précédents..."
rm -rf build dist *.egg-info

# Lancer le script Python de build
echo ""
echo "🔧 Lancement du build..."
python3 build_mac.py

# Si succès, proposer d'ouvrir l'app
if [ -d "dist/ClipGenius.app" ]; then
    echo ""
    read -p "Voulez-vous lancer l'application? (o/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        open dist/ClipGenius.app
    fi
fi

echo ""
echo "✅ Build terminé!"
echo ""
