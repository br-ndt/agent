#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# bootstrap.sh — Configure firewall, install deps, set up agent
#
# Works on any Ubuntu/Debian VPS. Detects Oracle Cloud and applies
# platform-specific firewall rules when needed.
#
# Usage:
#   chmod +x bootstrap.sh && ./bootstrap.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[✗]${NC} $*"; }
ask()   { echo -e "${CYAN}[?]${NC} $*"; }

# ─────────────────────────────────────────────────────────────
# Detect platform
# ─────────────────────────────────────────────────────────────
detect_platform() {
    if [ -f /etc/oracle-cloud-agent/agent.yml ] || \
       grep -qi "oracle" /sys/class/dmi/id/board_vendor 2>/dev/null || \
       grep -qi "OracleCloud" /etc/cloud/cloud.cfg 2>/dev/null; then
        echo "oracle"; return
    fi

    if [ -f /sys/class/dmi/id/board_vendor ]; then
        local vendor
        vendor=$(cat /sys/class/dmi/id/board_vendor 2>/dev/null || echo "")
        case "$vendor" in
            *Amazon*|*EC2*)  echo "aws"; return ;;
            *Google*)        echo "gcp"; return ;;
            *Microsoft*)     echo "azure"; return ;;
            *Hetzner*)       echo "hetzner"; return ;;
            *DigitalOcean*)  echo "digitalocean"; return ;;
        esac
    fi

    echo "unknown"
}

PLATFORM=$(detect_platform)

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Agent bootstrap"
echo "═══════════════════════════════════════════════════════"

if [ "$PLATFORM" != "unknown" ]; then
    info "Detected platform: $PLATFORM"
else
    ask "Couldn't auto-detect your platform. Which are you running on?"
    echo "  1) Oracle Cloud (OCI)"
    echo "  2) AWS / EC2"
    echo "  3) Hetzner"
    echo "  4) DigitalOcean"
    echo "  5) Other / bare metal"
    read -rp "  Choice [1-5]: " choice
    case "$choice" in
        1) PLATFORM="oracle" ;;
        2) PLATFORM="aws" ;;
        3) PLATFORM="hetzner" ;;
        4) PLATFORM="digitalocean" ;;
        *) PLATFORM="other" ;;
    esac
fi

# ─────────────────────────────────────────────────────────────
# PHASE 1: Firewall
# ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Phase 1: Firewall configuration"
echo "═══════════════════════════════════════════════════════"

configure_firewall_oracle() {
    warn "Oracle Cloud detected — configuring iptables (NOT ufw)"
    warn ""
    warn "You also need Ingress Rules in the OCI Console:"
    warn "  Networking > VCN > Subnet > Security List > Add Ingress Rules"
    warn "  - 0.0.0.0/0  TCP  443   (HTTPS)"
    warn "  - 0.0.0.0/0  TCP  8765  (status page, optional)"
    echo ""

    if ! sudo iptables -C INPUT -p tcp -m state --state NEW -m tcp --dport 443 -j ACCEPT 2>/dev/null; then
        sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
        info "Added iptables rule: allow TCP 443"
    else
        info "iptables rule for TCP 443 already exists"
    fi

    if ! sudo iptables -C INPUT -p tcp -m state --state NEW -m tcp --dport 8765 -j ACCEPT 2>/dev/null; then
        sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8765 -j ACCEPT
        info "Added iptables rule: allow TCP 8765"
    else
        info "iptables rule for TCP 8765 already exists"
    fi

    if command -v netfilter-persistent &>/dev/null; then
        sudo netfilter-persistent save
    else
        sudo apt-get install -y iptables-persistent 2>/dev/null || true
        sudo netfilter-persistent save 2>/dev/null || true
    fi
    info "iptables rules persisted"
}

configure_firewall_ufw() {
    if ! command -v ufw &>/dev/null; then
        sudo apt-get install -y ufw 2>/dev/null
    fi

    sudo ufw allow 22/tcp
    sudo ufw allow 443/tcp
    sudo ufw allow 8765/tcp

    if ! sudo ufw status | grep -q "Status: active"; then
        ask "Enable ufw firewall? (y/n)"
        read -rp "  " enable_ufw
        if [ "$enable_ufw" = "y" ]; then
            sudo ufw --force enable
            info "ufw enabled"
        else
            warn "ufw not enabled — configure your firewall manually"
        fi
    else
        sudo ufw reload
        info "ufw rules updated"
    fi
}

configure_firewall_skip() {
    warn "Skipping firewall configuration"
    warn "Make sure ports 22 (SSH) and 443 (HTTPS) are open"
    warn "Port 8765 (status page) is optional"
}

case "$PLATFORM" in
    oracle)
        configure_firewall_oracle
        ;;
    aws)
        warn "AWS uses Security Groups (configured in the EC2 console)"
        warn "Make sure your Security Group allows inbound TCP 443 and 8765"
        configure_firewall_ufw
        ;;
    hetzner|digitalocean)
        configure_firewall_ufw
        ;;
    *)
        ask "Configure firewall? (y/n)"
        read -rp "  " do_firewall
        if [ "$do_firewall" = "y" ]; then
            configure_firewall_ufw
        else
            configure_firewall_skip
        fi
        ;;
esac

# ─────────────────────────────────────────────────────────────
# PHASE 2: System dependencies
# ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Phase 2: System dependencies"
echo "═══════════════════════════════════════════════════════"

sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl build-essential libssl-dev pkg-config \
    python3.12 python3.12-venv python3.12-dev \
    jq 2>/dev/null || {
    warn "python3.12 not available, trying default python3"
    sudo apt-get install -y -qq \
        git curl build-essential libssl-dev pkg-config \
        python3 python3-venv python3-dev \
        jq 2>/dev/null
}

info "System packages installed"

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    info "uv installed"
else
    info "uv already installed ($(uv --version))"
fi

if ! grep -q 'local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# ─────────────────────────────────────────────────────────────
# PHASE 3: Claude Code CLI
# ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Phase 3: Claude Code CLI (optional)"
echo "═══════════════════════════════════════════════════════"

if command -v claude &>/dev/null; then
    info "Claude Code CLI already installed"
elif command -v npm &>/dev/null; then
    ask "Install Claude Code CLI? Uses your Pro/Max subscription for \$0 API cost (y/n)"
    read -rp "  " install_claude
    if [ "$install_claude" = "y" ]; then
        npm install -g @anthropic-ai/claude-code
        info "Claude Code CLI installed — run 'claude login' to authenticate"
    else
        info "Skipped — install later: npm install -g @anthropic-ai/claude-code"
    fi
else
    warn "npm not found — install Node.js for Claude CLI support:"
    warn "  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -"
    warn "  sudo apt install -y nodejs"
    warn "  npm install -g @anthropic-ai/claude-code"
fi

# ─────────────────────────────────────────────────────────────
# PHASE 4: Project setup
# ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Phase 4: Project setup"
echo "═══════════════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    cd "$SCRIPT_DIR"
    info "Project directory: $SCRIPT_DIR"

    ~/.local/bin/uv sync 2>/dev/null || uv sync
    info "Python dependencies installed"

    if [ ! -f .env ] && [ -f .env.example ]; then
        cp .env.example .env
        warn "Created .env from template — edit it with your API keys"
    fi

    mkdir -p state
else
    warn "pyproject.toml not found — run this script from the project root"
fi

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Bootstrap complete!  (platform: $PLATFORM)"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " Next steps:"
echo "   nano .env                          # Set API keys / tokens"
if command -v claude &>/dev/null; then
echo "   claude login                       # Auth with Claude subscription"
fi
echo "   uv run python test_connection.py   # Verify providers"
echo "   uv run python -m agent.main        # Start (CLI + Discord)"
echo ""
echo " To run as a daemon:"
echo "   sudo cp systemd/agent.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable --now agent"
echo ""