#!/bin/bash
# Setup script for Docker operations scripts
# Location: ./setup-docker-ops.sh (project root)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Docker Operations Scripts${NC}"
echo "===================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p ./scripts/ops/docker
mkdir -p ./backups

# Create script list
SCRIPTS=(
    "startup.sh"
    "shutdown.sh"
    "health-check.sh"
    "backup.sh"
    "restore.sh"
    "cleanup.sh"
    "reset.sh"
    "metrics.sh"
    "logs.sh"
)

# Make scripts executable
echo "Setting executable permissions..."
for script in "${SCRIPTS[@]}"; do
    if [ -f "./scripts/ops/docker/$script" ]; then
        chmod +x "./scripts/ops/docker/$script"
        echo "  ✓ $script"
    else
        echo "  ⚠ $script not found - please create it"
    fi
done

# Create convenience symlinks in project root (optional)
echo ""
read -p "Create convenience symlinks in project root? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ln -sf ./scripts/ops/docker/startup.sh ./docker-start.sh
    ln -sf ./scripts/ops/docker/shutdown.sh ./docker-stop.sh
    ln -sf ./scripts/ops/docker/health-check.sh ./docker-health.sh
    ln -sf ./scripts/ops/docker/cleanup.sh ./docker-cleanup.sh
    chmod +x ./docker-*.sh
    echo "  ✓ Symlinks created"
fi

# Create .gitignore entries
echo ""
read -p "Add backup directory to .gitignore? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! grep -q "^backups/" .gitignore 2>/dev/null; then
        echo "backups/" >> .gitignore
        echo "  ✓ Added backups/ to .gitignore"
    else
        echo "  ✓ backups/ already in .gitignore"
    fi
fi

# Verify docker-compose.yml exists
if [ -f "docker-compose.yml" ]; then
    echo -e "\n${GREEN}✓ docker-compose.yml found${NC}"
else
    echo -e "\n${YELLOW}⚠ docker-compose.yml not found in project root${NC}"
fi

# Display summary
echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo "Directory structure:"
echo "  ./scripts/ops/docker/     - Docker operation scripts"
echo "  ./backups/                - Database backups (gitignored)"
echo ""
echo "Available scripts:"
for script in "${SCRIPTS[@]}"; do
    echo "  ./scripts/ops/docker/$script"
done
echo ""
echo "Quick start:"
echo "  1. Start services:  ./scripts/ops/docker/startup.sh"
echo "  2. Check health:    ./scripts/ops/docker/health-check.sh"
echo "  3. View logs:       ./scripts/ops/docker/logs.sh"
echo "  4. Stop services:   ./scripts/ops/docker/shutdown.sh"
echo ""
echo "For detailed documentation, see: ./scripts/ops/docker/README.md"