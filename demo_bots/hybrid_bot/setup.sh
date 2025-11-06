#!/bin/bash
# Setup script for Hybrid Chess Bot

echo "=========================================="
echo "Hybrid Chess Bot - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Install pip if needed
echo ""
echo "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found. Installing..."
    sudo apt update
    sudo apt install -y python3-pip
fi

# Install required packages
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

pip3 install --user chess==1.11.2
pip3 install --user pygame==2.6.1
pip3 install --user torch>=2.0.0
pip3 install --user numpy>=1.24.0
pip3 install --user pandas>=2.0.0

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To train the bot:"
echo "  python3 -m demo_bots.hybrid_bot train"
echo ""
echo "To play as white:"
echo "  python3 -m demo_bots.hybrid_bot play w"
echo ""
echo "To play as black:"
echo "  python3 -m demo_bots.hybrid_bot play b"
echo ""
