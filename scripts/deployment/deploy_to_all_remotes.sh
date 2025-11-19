#!/bin/bash

# QeMLflow Multi-Remote Deployment Script
# Automatically switches GitHub accounts and pushes to all configured remotes
# Usage: ./deploy_to_all_remotes.sh [commit_message]

set -e  # Exit on any error

# Configuration
REMOTES=(
    "origin:SanjeevaRDodlapati"
    "sdodlapa:sdodlapa" 
    "sdodlapati3:sdodlapati3"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if gh CLI is authenticated for a user
check_gh_auth() {
    local user=$1
    if gh auth status 2>/dev/null | grep -q "account $user"; then
        return 0
    else
        log_error "GitHub CLI not authenticated for user: $user"
        log_info "Please run: gh auth login --user $user"
        return 1
    fi
}

# Function to switch GitHub account
switch_github_account() {
    local user=$1
    log_info "Switching to GitHub account: $user"
    
    if ! gh auth switch --user "$user" >/dev/null 2>&1; then
        log_error "Failed to switch to GitHub account: $user"
        return 1
    fi
    
    log_success "Successfully switched to: $user"
    return 0
}

# Function to push to a remote
push_to_remote() {
    local remote=$1
    local branch=${2:-main}
    
    log_info "Pushing to remote: $remote (branch: $branch)"
    
    # Check if remote exists
    if ! git remote get-url "$remote" >/dev/null 2>&1; then
        log_error "Remote '$remote' not found in git configuration"
        return 1
    fi
    
    # Push to remote
    if git push "$remote" "$branch" 2>/dev/null; then
        log_success "Successfully pushed to: $remote"
        return 0
    else
        log_error "Failed to push to: $remote"
        return 1
    fi
}

# Function to get current git status
get_git_status() {
    local status=$(git status --porcelain)
    if [ -n "$status" ]; then
        echo "uncommitted"
    else
        echo "clean"
    fi
}

# Function to commit changes if provided
commit_changes() {
    local message="$1"
    
    if [ -n "$message" ]; then
        log_info "Committing changes with message: $message"
        
        # Add all changes
        git add -A
        
        # Check if there are changes to commit
        if git diff --staged --quiet; then
            log_warning "No changes to commit"
            return 0
        fi
        
        # Commit changes
        if git commit -m "$message"; then
            log_success "Changes committed successfully"
            return 0
        else
            log_error "Failed to commit changes"
            return 1
        fi
    fi
}

# Function to create deployment summary
create_deployment_summary() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local commit_hash=$(git rev-parse HEAD)
    local commit_message=$(git log -1 --pretty=%B)
    
    cat > "docs/deployment/deployment_$(date +%Y%m%d_%H%M%S).md" << EOF
# Multi-Remote Deployment Summary

**Date:** $timestamp  
**Commit:** $commit_hash  
**Branch:** $(git branch --show-current)  

## Commit Message
\`\`\`
$commit_message
\`\`\`

## Deployment Status
EOF

    echo "| Remote | Account | Status | Notes |" >> "docs/deployment/deployment_$(date +%Y%m%d_%H%M%S).md"
    echo "|--------|---------|--------|-------|" >> "docs/deployment/deployment_$(date +%Y%m%d_%H%M%S).md"
}

# Main deployment function
main() {
    local commit_message="$1"
    local branch=${2:-main}
    local failed_deployments=()
    local successful_deployments=()
    
    log_info "Starting QeMLflow multi-remote deployment"
    log_info "Current directory: $(pwd)"
    log_info "Git status: $(get_git_status)"
    
    # Ensure we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Commit changes if message provided
    if [ -n "$commit_message" ]; then
        commit_changes "$commit_message"
    fi
    
    # Check git status after potential commit
    if [ "$(get_git_status)" != "clean" ]; then
        log_warning "Working directory has uncommitted changes"
        log_info "These changes will not be pushed to remotes"
    fi
    
    # Create deployment summary
    create_deployment_summary
    local summary_file="docs/deployment/deployment_$(date +%Y%m%d_%H%M%S).md"
    
    log_info "Deploying to ${#REMOTES[@]} remotes..."
    
    # Process each remote
    for remote_config in "${REMOTES[@]}"; do
        IFS=':' read -r remote_name github_user <<< "$remote_config"
        
        log_info "Processing remote: $remote_name (user: $github_user)"
        
        # Check authentication
        if ! check_gh_auth "$github_user"; then
            log_error "Skipping $remote_name due to authentication issues"
            failed_deployments+=("$remote_name (auth failed)")
            echo "| $remote_name | $github_user | ❌ Failed | Authentication failed |" >> "$summary_file"
            continue
        fi
        
        # Switch GitHub account
        if ! switch_github_account "$github_user"; then
            log_error "Skipping $remote_name due to account switch failure"
            failed_deployments+=("$remote_name (switch failed)")
            echo "| $remote_name | $github_user | ❌ Failed | Account switch failed |" >> "$summary_file"
            continue
        fi
        
        # Push to remote
        if push_to_remote "$remote_name" "$branch"; then
            successful_deployments+=("$remote_name")
            echo "| $remote_name | $github_user | ✅ Success | Deployed successfully |" >> "$summary_file"
        else
            failed_deployments+=("$remote_name (push failed)")
            echo "| $remote_name | $github_user | ❌ Failed | Push failed |" >> "$summary_file"
        fi
        
        # Small delay between operations
        sleep 1
    done
    
    # Finalize deployment summary
    echo "" >> "$summary_file"
    echo "## Summary" >> "$summary_file"
    echo "- **Successful:** ${#successful_deployments[@]} remotes" >> "$summary_file"
    echo "- **Failed:** ${#failed_deployments[@]} remotes" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "*Generated by deploy_to_all_remotes.sh*" >> "$summary_file"
    
    # Final report
    echo ""
    log_info "Deployment Summary:"
    log_success "Successfully deployed to: ${successful_deployments[*]:-none}"
    
    if [ ${#failed_deployments[@]} -gt 0 ]; then
        log_error "Failed deployments: ${failed_deployments[*]}"
        log_info "Deployment summary saved to: $summary_file"
        exit 1
    else
        log_success "All deployments completed successfully!"
        log_info "Deployment summary saved to: $summary_file"
        exit 0
    fi
}

# Help function
show_help() {
    cat << EOF
QeMLflow Multi-Remote Deployment Script

USAGE:
    $0 [OPTIONS] [COMMIT_MESSAGE]

OPTIONS:
    -h, --help          Show this help message
    -b, --branch BRANCH Specify branch to push (default: main)
    --status            Show deployment configuration and status
    --test              Test authentication for all accounts

EXAMPLES:
    $0 "feat: add new feature"
    $0 --branch develop "fix: critical bug fix"
    $0 --test
    $0 --status

CONFIGURATION:
    Remotes are configured in the REMOTES array at the top of this script.
    Current configuration:
$(printf '    %s\n' "${REMOTES[@]}")

REQUIREMENTS:
    - GitHub CLI (gh) must be installed and authenticated for each account
    - Git repository with configured remotes
    - Proper permissions for each remote repository

EOF
}

# Test authentication function
test_authentication() {
    log_info "Testing authentication for all configured accounts..."
    
    local auth_status=0
    for remote_config in "${REMOTES[@]}"; do
        IFS=':' read -r remote_name github_user <<< "$remote_config"
        
        if check_gh_auth "$github_user"; then
            log_success "$github_user: ✅ Authenticated"
        else
            log_error "$github_user: ❌ Not authenticated"
            auth_status=1
        fi
    done
    
    if [ $auth_status -eq 0 ]; then
        log_success "All accounts are properly authenticated"
    else
        log_error "Some accounts need authentication. Run: gh auth login --user <username>"
    fi
    
    exit $auth_status
}

# Show status function
show_status() {
    log_info "QeMLflow Multi-Remote Deployment Configuration"
    echo ""
    echo "Current directory: $(pwd)"
    echo "Git repository: $(git remote get-url origin 2>/dev/null || echo 'Not configured')"
    echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'Unknown')"
    echo "Git status: $(get_git_status)"
    echo ""
    echo "Configured remotes:"
    
    for remote_config in "${REMOTES[@]}"; do
        IFS=':' read -r remote_name github_user <<< "$remote_config"
        local remote_url=$(git remote get-url "$remote_name" 2>/dev/null || echo "Not configured")
        echo "  $remote_name ($github_user): $remote_url"
    done
    
    echo ""
    test_authentication
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --test)
        test_authentication
        ;;
    --status)
        show_status
        ;;
    --branch|-b)
        if [ -z "${2:-}" ]; then
            log_error "Branch name required after --branch"
            exit 1
        fi
        main "${3:-}" "$2"
        ;;
    *)
        main "$1"
        ;;
esac
