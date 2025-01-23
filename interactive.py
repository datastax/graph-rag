import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'packages/metadata-extractors/src')))

# Set the current working directory to the src directory
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'packages/metadata-extractors/src')))

# Import the necessary classes
from keybert_metadata_extractor import KeyBERTMetadataExtractor

print("Environment setup complete. You can now use KeyBERTMetadataExtractor.")