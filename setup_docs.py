#!/usr/bin/env python3
"""
Setup script for MkDocs documentation system.
This script helps set up the documentation environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up MkDocs Documentation System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("mkdocs.yml").exists():
        print("❌ Error: mkdocs.yml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install documentation dependencies
    if not run_command("pip install -r requirements-docs.txt", "Installing MkDocs dependencies"):
        print("⚠️  Warning: Some dependencies may not have installed correctly")
    
    # Install project dependencies
    if not run_command("pip install gensim scikit-learn pandas numpy matplotlib seaborn wordcloud pyLDAvis", "Installing project dependencies"):
        print("⚠️  Warning: Some project dependencies may not have installed correctly")
    
    # Test imports
    print("🔄 Testing Python imports...")
    try:
        import sys
        sys.path.append('.')
        from src.topics.lda_model import LDAModeler
        print("✅ Python imports working correctly")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        print("Please check that all dependencies are installed correctly")
        return False
    
    # Test MkDocs build
    if not run_command("mkdocs build", "Testing MkDocs build"):
        print("❌ MkDocs build failed. Please check the configuration.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Run 'mkdocs serve' to start the development server")
    print("2. Open http://127.0.0.1:8000 in your browser")
    print("3. Edit your Python docstrings to update the API documentation")
    print("4. Run 'mkdocs build' to create production-ready documentation")
    print("5. Run 'mkdocs gh-deploy' to deploy to GitHub Pages")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
