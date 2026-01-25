import sys
from pathlib import Path

# Ajouter le répertoire racine au path (parent de tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.stdout.reconfigure(line_buffering=True)
print('Starting...')

import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Importing viral_detector...')
from src.viral_detector import ViralMoment
print('OK viral_detector')

print('Importing clip_generator...')
from src.clip_generator import ClipGenerator, ClipConfig
print('OK clip_generator')

print('All imports successful!')
