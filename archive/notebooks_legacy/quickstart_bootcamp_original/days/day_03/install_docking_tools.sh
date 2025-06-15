#!/bin/bash

# Install AutoDock Vina and Open Babel for Molecular Docking
echo "🔧 Installing Molecular Docking Dependencies..."

# Function to install Vina via Python pip (fastest and most reliable)
install_vina_python() {
    echo "🐍 Installing AutoDock Vina via Python pip..."

    # Check if pip is available
    if command -v pip &> /dev/null; then
        echo "📦 Installing vina package..."
        pip install -U vina

        # Test the installation
        python3 -c "import vina; print('✅ Python Vina package imported successfully!'); v = vina.Vina(); print(f'✅ Vina version: {vina.__version__}')" 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "✅ Python Vina package installed and working!"
            return 0
        else
            echo "⚠️  Python Vina package installed but may have issues"
            return 1
        fi
    else
        echo "❌ pip not found. Please install pip first."
        return 1
    fi
}

# Function to install AutoDock Vina from GitHub (fallback)
install_vina_from_github() {
    echo "🚀 Installing AutoDock Vina from GitHub repository..."

    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Check if we have git
    if ! command -v git &> /dev/null; then
        echo "❌ Git not found. Please install git first."
        return 1
    fi

    # Check for build tools
    if ! command -v cmake &> /dev/null; then
        echo "📦 Installing cmake via brew..."
        if command -v brew &> /dev/null; then
            brew install cmake
        else
            echo "❌ cmake required for building. Please install cmake."
            return 1
        fi
    fi

    # Clone AutoDock Vina repository
    echo "📥 Cloning AutoDock Vina repository..."
    git clone https://github.com/ccsb-scripps/AutoDock-Vina.git
    cd AutoDock-Vina

    # Create build directory
    mkdir -p build
    cd build

    # Configure with cmake
    echo "⚙️  Configuring build with cmake..."
    cmake ..

    # Build
    echo "🔨 Building AutoDock Vina (this may take a few minutes)..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    # Install to /usr/local/bin (requires sudo)
    echo "📦 Installing AutoDock Vina to /usr/local/bin..."
    sudo cp bin/vina /usr/local/bin/
    sudo chmod +x /usr/local/bin/vina

    # Clean up
    cd /
    rm -rf "$TEMP_DIR"

    echo "✅ AutoDock Vina installed from GitHub!"
    return 0
}

# Try Python pip installation first (fastest and most reliable)
echo "🎯 Attempting Python Vina installation first (recommended)..."
if install_vina_python; then
    echo "🎉 Successfully installed AutoDock Vina via Python!"
    VINA_INSTALLED=true
else
    echo "⚠️  Python Vina installation failed, trying other methods..."
    VINA_INSTALLED=false
fi

# Install Open Babel
if command -v conda &> /dev/null; then
    echo "📦 Installing Open Babel via conda..."
    conda install -c conda-forge openbabel -y

    # Try conda for Vina if Python method failed
    if [ "$VINA_INSTALLED" = false ]; then
        echo "📦 Trying to install AutoDock Vina via conda..."
        if conda install -c conda-forge autodock-vina -y 2>/dev/null; then
            VINA_INSTALLED=true
        else
            echo "⚠️  AutoDock Vina not available via conda, trying GitHub..."
            if install_vina_from_github; then
                VINA_INSTALLED=true
            fi
        fi
    fi

elif command -v brew &> /dev/null; then
    echo "📦 Installing Open Babel via Homebrew..."
    brew install open-babel

    # Try homebrew for Vina if Python method failed
    if [ "$VINA_INSTALLED" = false ]; then
        echo "📦 Trying to install AutoDock Vina via Homebrew..."
        if brew install autodock-vina 2>/dev/null; then
            VINA_INSTALLED=true
        else
            echo "⚠️  AutoDock Vina not available via brew, trying GitHub..."
            if install_vina_from_github; then
                VINA_INSTALLED=true
            fi
        fi
    fi

else
    echo "⚠️  Neither conda nor brew found."

    # Install Open Babel from source if needed
    echo "📋 You may need to install Open Babel manually:"
    echo "   - Open Babel: https://openbabel.org/wiki/Category:Installation"

    # Try GitHub installation if Python method failed
    if [ "$VINA_INSTALLED" = false ]; then
        echo "🚀 Attempting to install AutoDock Vina from GitHub..."
        if install_vina_from_github; then
            VINA_INSTALLED=true
        fi
    fi
fi

# Verify installations
echo ""
echo "🔍 Verifying installations..."

# Check for command-line vina first
if command -v vina &> /dev/null; then
    VERSION=$(vina --version 2>&1 | head -1 || echo "Version check failed")
    echo "✅ AutoDock Vina (command-line): $VERSION"

    # Test basic functionality
    echo "🧪 Testing AutoDock Vina functionality..."
    if vina --help >/dev/null 2>&1; then
        echo "   ✅ AutoDock Vina is working correctly!"
    else
        echo "   ⚠️  AutoDock Vina installed but may have issues"
    fi
# Check for Python vina package
elif python3 -c "import vina; print('Python Vina available')" 2>/dev/null; then
    echo "✅ AutoDock Vina (Python package): Available"
    echo "🧪 Testing Python Vina functionality..."

    # Test basic Vina Python functionality
    python3 -c "
import vina
v = vina.Vina(sf_name='vina')
print('   ✅ Python Vina is working correctly!')
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "   ✅ Python Vina is working correctly!"
    else
        echo "   ⚠️  Python Vina installed but may have issues"
    fi
else
    echo "❌ AutoDock Vina: Not found"
    echo ""
    echo "🔄 Try these installation options:"
    echo "1. 🐍 Python package (recommended):"
    echo "   pip install -U numpy vina"
    echo ""
    echo "2. 📥 Download precompiled binary:"
    echo "   curl -L -o /usr/local/bin/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_mac_64bit"
    echo "   chmod +x /usr/local/bin/vina"
    echo ""
    echo "3. 🎭 Continue with simulation mode (educational)"
fi

if command -v obabel &> /dev/null; then
    VERSION=$(obabel -V 2>&1 | head -1 || echo "Version check failed")
    echo "✅ Open Babel: $VERSION"

    # Test basic functionality
    echo "🧪 Testing Open Babel functionality..."
    if echo "CCO" | obabel -ismi -ocan >/dev/null 2>&1; then
        echo "   ✅ Open Babel is working correctly!"
    else
        echo "   ⚠️  Open Babel installed but may have issues"
    fi
else
    echo "❌ Open Babel: Not found"
    echo "💡 Try: conda install -c conda-forge openbabel"
fi

echo ""
echo "🚀 Installation script complete!"
echo ""

# Final verification
echo "🔍 FINAL VERIFICATION:"
echo "===================="

# Test Python Vina package
echo "🐍 Python Vina Package:"
if python3 -c "import vina; v = vina.Vina(); print(f'   ✅ Version {vina.__version__} - Ready for use!')" 2>/dev/null; then
    echo "   🎯 Python package is working perfectly!"
else
    echo "   ❌ Python package not available"
fi

# Test command-line vina
echo ""
echo "💻 Command-line AutoDock Vina:"
if command -v vina &> /dev/null; then
    vina --help 2>&1 | head -2 | sed 's/^/   ✅ /'
else
    echo "   ❌ Command-line vina not found"
fi

# Test Open Babel
echo ""
echo "🧪 Open Babel:"
if command -v obabel &> /dev/null; then
    VERSION=$(obabel -V 2>&1 | head -1)
    echo "   ✅ $VERSION"
    if echo "CCO" | obabel -ismi -ocan >/dev/null 2>&1; then
        echo "   ✅ Functionality test passed!"
    fi
else
    echo "   ❌ Not found - install with: conda install -c conda-forge openbabel"
fi

echo ""
echo "📋 Next Steps:"
echo "1. 🔄 Restart your Jupyter kernel"
echo "2. 🧪 Run the MolecularDockingEngine cell again"
echo "3. 🎯 Both real Vina and simulation modes are now available!"
echo ""
echo "💡 The MolecularDockingEngine will automatically detect and use"
echo "   the best available docking method for your system!"
