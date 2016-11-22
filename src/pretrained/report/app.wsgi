import sys
# Add path for deployment on Apache
sys.path.insert(0, '/Library/WebServer/Documents/fireballs/report/')
from app import app as application