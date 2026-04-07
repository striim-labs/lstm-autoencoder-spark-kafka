#!/bin/bash
# =============================================================
# NYCAD Scorer - Build Script (WAEvent pass-through, no types JAR)
# =============================================================
# Run this from the nycad-scorer/ directory.
#
# Prerequisites:
#   - Maven installed (brew install maven)
#   - Striim installed at /opt/Striim
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STRIIM_HOME="${STRIIM_HOME:-/opt/Striim}"

echo "=== Step 1: Install Striim SDK into local Maven repo ==="

SDK_JAR="$STRIIM_HOME/StriimSDK/StriimOpenProcessor-SDK.jar"

if [ ! -f "$SDK_JAR" ]; then
    echo "ERROR: SDK jar not found at $SDK_JAR"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=OpenProcessorSDK \
    -Dversion=1.0.0-SNAPSHOT \
    -Dpackaging=jar \
    -Dfile="$SDK_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 1b: Install Striim Common (runtime WAEvent class) ==="

COMMON_JAR="$STRIIM_HOME/lib/Common-5.2.0.4.jar"

if [ ! -f "$COMMON_JAR" ]; then
    echo "ERROR: Common jar not found at $COMMON_JAR"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=Common \
    -Dversion=5.2.0.4 \
    -Dpackaging=jar \
    -Dfile="$COMMON_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 2: Build the Open Processor ==="

mvn clean package

echo ""
echo "=== Step 3: Copy .scm to Striim modules ==="

cp target/NYCADScorer.jar target/NYCADScorer.scm
cp target/NYCADScorer.scm "$STRIIM_HOME/modules/NYCADScorer.scm"
echo "Copied to $STRIIM_HOME/modules/NYCADScorer.scm"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Start Striim if not running:  \$STRIIM_HOME/bin/server.sh"
echo "  2. In the Striim console:"
echo '     LOAD OPEN PROCESSOR "/opt/Striim/modules/NYCADScorer.scm";'
echo "  3. Paste the TQL from striim-op/NYCAD.tql"
echo "  4. Deploy, start, then copy data"
