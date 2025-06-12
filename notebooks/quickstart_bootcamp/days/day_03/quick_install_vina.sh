#!/bin/bash

# Quick AutoDock Vina Installation Script
# Downloads precompiled binary for faster installation

echo "⚡ Quick AutoDock Vina Installation"
echo "=================================="

# Check if vina is already installed
if command -v vina &> /dev/null; then
    echo "✅ AutoDock Vina is already installed!"
    vina --version 2>&1 | head -1
    exit 0
fi

echo "📥 Downloading AutoDock Vina precompiled binary..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download the latest precompiled binary for macOS
VINA_URL="https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_mac_64bit"

if curl -L -o vina "$VINA_URL"; then
    echo "📦 Installing AutoDock Vina to /usr/local/bin..."

    # Make executable
    chmod +x vina

    # Install to system path (requires sudo)
    if sudo cp vina /usr/local/bin/vina; then
        echo "✅ AutoDock Vina installed successfully!"

        # Verify installation
        echo "🧪 Testing installation..."
        if /usr/local/bin/vina --version >/dev/null 2>&1; then
            echo "✅ AutoDock Vina is working correctly!"
            /usr/local/bin/vina --version 2>&1 | head -1
        else
            echo "⚠️  Installation completed but Vina may need additional setup"
        fi
    else
        echo "❌ Failed to install Vina (permission denied)"
        echo "💡 Try running: sudo cp $TEMP_DIR/vina /usr/local/bin/vina"
    fi
else
    echo "❌ Failed to download AutoDock Vina"
    echo "🔄 Alternatives:"
    echo "1. Run the full installation script: ./install_docking_tools.sh"
    echo "2. Install via conda: conda install -c conda-forge autodock-vina"
    echo "3. Build from source: https://github.com/ccsb-scripps/AutoDock-Vina"
fi

# Clean up
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "🎯 Ready to use AutoDock Vina for molecular docking!"
echo "🔄 Don't forget to restart your Jupyter kernel."
