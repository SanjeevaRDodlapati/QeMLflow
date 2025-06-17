#!/bin/bash

# QeMLflow Renaming Script - PRODUCTION VERSION
# This script safely renames QeMLflow to QeMLflow throughout the entire codebase

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(pwd)"
BACKUP_DIR="${PROJECT_ROOT}/pre_qemlflow_rename_backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/qemlflow_rename_log_$(date +%Y%m%d_%H%M%S).txt"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}QeMLflow Renaming Script - PRODUCTION${NC}"
echo -e "${BLUE}=========================================${NC}"

# Function to log messages
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to create comprehensive backup
create_backup() {
    echo -e "${YELLOW}Creating comprehensive backup...${NC}"
    log_message "Creating backup at: $BACKUP_DIR"
    
    mkdir -p "$BACKUP_DIR"
    
    # Copy the entire project structure except heavy directories
    rsync -av --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
          --exclude='node_modules/' --exclude='.pytest_cache/' \
          --exclude='qemlflow_env/' --exclude='site/' \
          "$PROJECT_ROOT/" "$BACKUP_DIR/"
    
    log_message "Backup created successfully at: $BACKUP_DIR"
    echo -e "${GREEN}✓ Backup created successfully${NC}"
}

# Function to verify current state
verify_current_state() {
    echo -e "${YELLOW}Verifying current state...${NC}"
    
    local qemlflow_count=$(find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) \
                        -not -path "./.git/*" -not -path "./pre_*" -not -path "./test_*" \
                        -exec grep -l "qemlflow\|QeMLflow" {} \; | wc -l)
    
    log_message "Found $qemlflow_count files containing QeMLflow references"
    echo -e "${BLUE}Found $qemlflow_count files containing QeMLflow references${NC}"
    
    if [ "$qemlflow_count" -eq 0 ]; then
        echo -e "${RED}Warning: No QeMLflow references found. Already renamed?${NC}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to perform text replacements
perform_text_replacements() {
    echo -e "${YELLOW}Performing text replacements...${NC}"
    
    # Find all relevant files (excluding git, backups, and cache directories)
    local files_to_process=$(find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" -o -name "*.cfg" -o -name "*.ini" \) \
                           -not -path "./.git/*" -not -path "./pre_*" -not -path "./test_*" -not -path "./__pycache__/*" -not -path "./qemlflow_env/*" -not -path "./site/*")
    
    local file_count=$(echo "$files_to_process" | wc -l)
    log_message "Processing $file_count files for text replacement"
    
    echo "$files_to_process" | while read -r file; do
        if [ -f "$file" ]; then
            # Create individual backup
            cp "$file" "${file}.pre_qemlflow_bak"
            
            # Perform replacements
            sed -i.tmp 's/qemlflow/qemlflow/g' "$file"
            sed -i.tmp 's/QeMLflow/QeMLflow/g' "$file"
            sed -i.tmp 's/QEMLFLOW/QEMLFLOW/g' "$file"
            
            # Remove temporary files
            rm -f "${file}.tmp"
            
            echo "Processed: $file"
        fi
    done
    
    log_message "Text replacements completed"
    echo -e "${GREEN}✓ Text replacements completed${NC}"
}

# Function to rename directories
rename_directories() {
    echo -e "${YELLOW}Renaming directories...${NC}"
    
    # Find and rename qemlflow directories
    if [ -d "src/qemlflow" ]; then
        log_message "Renaming src/qemlflow to src/qemlflow"
        mv "src/qemlflow" "src/qemlflow"
        echo -e "${GREEN}✓ Renamed src/qemlflow to src/qemlflow${NC}"
    fi
    
    # Check for any other qemlflow directories
    local other_dirs=$(find . -type d -name "*qemlflow*" -not -path "./.git/*" -not -path "./pre_*" -not -path "./test_*")
    if [ -n "$other_dirs" ]; then
        echo -e "${YELLOW}Found other directories with 'qemlflow' in name:${NC}"
        echo "$other_dirs"
        echo -e "${YELLOW}Please rename these manually if needed.${NC}"
        log_message "Additional qemlflow directories found: $other_dirs"
    fi
    
    log_message "Directory renaming completed"
}

# Function to update package configuration
update_package_config() {
    echo -e "${YELLOW}Updating package configuration...${NC}"
    
    # Update setup.py if it exists
    if [ -f "setup.py" ]; then
        log_message "Updating setup.py"
        # Additional specific replacements for setup.py
        sed -i.qemlflow_bak 's/find_packages(where="src")/find_packages(where="src")/g' setup.py
        echo -e "${GREEN}✓ Updated setup.py${NC}"
    fi
    
    # Update pyproject.toml if it exists
    if [ -f "pyproject.toml" ]; then
        log_message "Updating pyproject.toml"
        echo -e "${GREEN}✓ Updated pyproject.toml${NC}"
    fi
    
    log_message "Package configuration updated"
}

# Function to verify renaming results
verify_renaming_results() {
    echo -e "${YELLOW}Verifying renaming results...${NC}"
    
    # Check for remaining QeMLflow references
    local remaining_qemlflow=$(find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) \
                           -not -path "./.git/*" -not -path "./pre_*" -not -path "./test_*" -not -name "*.bak" \
                           -exec grep -l "qemlflow\|QeMLflow" {} \; 2>/dev/null | wc -l)
    
    local new_qemlflow=$(find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) \
                        -not -path "./.git/*" -not -path "./pre_*" -not -path "./test_*" -not -name "*.bak" \
                        -exec grep -l "qemlflow\|QeMLflow" {} \; 2>/dev/null | wc -l)
    
    log_message "Verification results: $remaining_qemlflow QeMLflow files remaining, $new_qemlflow QeMLflow files created"
    
    if [ "$remaining_qemlflow" -gt 0 ]; then
        echo -e "${YELLOW}Warning: $remaining_qemlflow files still contain QeMLflow references${NC}"
        echo -e "${YELLOW}These may need manual review:${NC}"
        find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) \
             -not -path "./.git/*" -not -path "./pre_*" -not -path "./test_*" -not -name "*.bak" \
             -exec grep -l "qemlflow\|QeMLflow" {} \; 2>/dev/null | head -10
    else
        echo -e "${GREEN}✓ No remaining QeMLflow references found${NC}"
    fi
    
    if [ "$new_qemlflow" -gt 0 ]; then
        echo -e "${GREEN}✓ Successfully created $new_qemlflow files with QeMLflow references${NC}"
    else
        echo -e "${RED}✗ No QeMLflow references found - something may be wrong${NC}"
    fi
    
    # Check if src/qemlflow exists
    if [ -d "src/qemlflow" ]; then
        echo -e "${GREEN}✓ src/qemlflow directory exists${NC}"
    else
        echo -e "${RED}✗ src/qemlflow directory not found${NC}"
    fi
}

# Function to test basic import functionality
test_basic_functionality() {
    echo -e "${YELLOW}Testing basic Python syntax...${NC}"
    
    local syntax_errors=0
    for py_file in $(find src/ -name "*.py" 2>/dev/null); do
        if ! python -m py_compile "$py_file" 2>/dev/null; then
            echo -e "${RED}Syntax error in: $py_file${NC}"
            syntax_errors=$((syntax_errors + 1))
        fi
    done
    
    if [ "$syntax_errors" -eq 0 ]; then
        echo -e "${GREEN}✓ All Python files have valid syntax${NC}"
        log_message "Python syntax validation passed"
    else
        echo -e "${YELLOW}Warning: $syntax_errors Python files have syntax errors${NC}"
        log_message "Python syntax validation found $syntax_errors errors"
    fi
}

# Function to provide rollback instructions
provide_rollback_instructions() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}ROLLBACK INSTRUCTIONS${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${YELLOW}If you need to rollback the changes:${NC}"
    echo "1. Full rollback from backup:"
    echo "   rm -rf src docs reports tools scripts tests examples"
    echo "   cp -r \"$BACKUP_DIR\"/* ."
    echo ""
    echo "2. Individual file rollback:"
    echo "   find . -name '*.pre_qemlflow_bak' -exec bash -c 'mv \"\$1\" \"\${1%.pre_qemlflow_bak}\"' _ {} \\;"
    echo ""
    echo -e "${YELLOW}Backup location: $BACKUP_DIR${NC}"
    echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
}

# Main execution function
main() {
    log_message "Starting QeMLflow renaming process"
    echo -e "${BLUE}Starting QeMLflow to QeMLflow renaming process...${NC}"
    
    # Confirm with user
    echo -e "${YELLOW}This will rename all instances of QeMLflow to QeMLflow in your codebase.${NC}"
    echo -e "${YELLOW}A backup will be created automatically.${NC}"
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
    
    # Execute renaming steps
    verify_current_state
    create_backup
    perform_text_replacements
    rename_directories
    update_package_config
    verify_renaming_results
    test_basic_functionality
    provide_rollback_instructions
    
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}QeMLflow renaming completed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Review the changes carefully"
    echo "2. Test your code functionality"
    echo "3. Update GitHub repository name"
    echo "4. Commit the changes if everything looks good"
    
    log_message "QeMLflow renaming process completed successfully"
}

# Cleanup function for emergency exit
cleanup_on_exit() {
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    find . -name "*.tmp" -delete 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup_on_exit EXIT

# Run the main function
main "$@"
