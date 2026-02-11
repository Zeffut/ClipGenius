#!/usr/bin/env python3
"""
Script de test pour vérifier que la page test-video fonctionne
Usage: python test_server.py
Puis ouvre http://localhost:5555/test-video dans ton navigateur
"""

import sys
from pathlib import Path

# Constantes de test
TEST_SERVER_PORT = 5555

# Ajouter le dossier App au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_app import app, socketio

if __name__ == '__main__':
    print("=" * 60)
    print("🧪 SERVEUR DE TEST CLIPGENIUS")
    print("=" * 60)
    print()
    print("📍 Ouvre ton navigateur sur:")
    print("   http://localhost:5555/test-video")
    print()
    print("Pages disponibles:")
    print("   http://localhost:5555/          - Interface principale")
    print("   http://localhost:5555/test-video - Page de test vidéo")
    print()
    print("Pour arrêter: Ctrl+C")
    print("=" * 60)
    print()
    
    socketio.run(
        app,
        host='127.0.0.1',
        port=TEST_SERVER_PORT,
        debug=True,
        use_reloader=False
    )
