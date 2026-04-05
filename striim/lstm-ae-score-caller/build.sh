#!/bin/bash
# =============================================================
# LSTM-AE Score Caller - Build & Install Script
# =============================================================
# Run this from the lstm-ae-score-caller/ directory.
#
# Prerequisites:
#   - Maven installed (brew install maven)
#   - Striim installed at /opt/Striim
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1a: Install Striim SDK into local Maven repo ==="
echo ""

SDK_JAR="/opt/Striim/StriimSDK/StriimOpenProcessor-SDK.jar"

if [ ! -f "$SDK_JAR" ]; then
    echo "ERROR: SDK jar not found at $SDK_JAR"
    echo ""
    echo "Look for it with:  find /opt/Striim -name '*OpenProcessor*' -o -name '*SDK*' 2>/dev/null"
    echo "Then update the SDK_JAR variable in this script."
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
echo "=== Step 1b: Install Striim Common and Kryo into local Maven repo ==="
echo ""
echo "The hand-built type classes need UUID, SimpleEvent, Kryo, etc."
echo ""

COMMON_JAR="/opt/Striim/lib/Common-5.2.0.4.jar"
KRYO_JAR="/opt/Striim/lib/kryo-2.20.jar"

if [ ! -f "$COMMON_JAR" ]; then
    echo "WARNING: Common JAR not found at $COMMON_JAR"
    echo "Look for it with:  find /opt/Striim/lib -name 'Common-*' 2>/dev/null"
    exit 1
fi

if [ ! -f "$KRYO_JAR" ]; then
    echo "WARNING: Kryo JAR not found at $KRYO_JAR"
    echo "Look for it with:  find /opt/Striim/lib -name 'kryo-*' 2>/dev/null"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=Common \
    -Dversion=5.2.0.4 \
    -Dpackaging=jar \
    -Dfile="$COMMON_JAR" \
    -DgeneratePom=true

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=kryo \
    -Dversion=2.20 \
    -Dpackaging=jar \
    -Dfile="$KRYO_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 2: Build the Open Processor ==="
echo ""
echo "Type classes are compiled from source but EXCLUDED from the .scm"
echo "(to avoid classloader conflicts). They go in a separate types JAR."
echo ""

mvn clean package

echo ""
echo "=== Step 3: Build the types JAR ==="
echo ""

# Package the hand-built type classes into a separate JAR for $STRIIM_HOME/lib/
cd target/classes
jar cf ../lstmae_types.jar wa/lstmae/*.class
cd ../..

echo "Built: target/lstmae_types.jar"

echo ""
echo "=== Step 4: Rename .jar to .scm ==="
echo ""

cp target/LSTMAEScoreCaller.jar target/LSTMAEScoreCaller.scm

echo "Built: target/LSTMAEScoreCaller.scm"

# Also copy types JAR to striim/lib/ for repo reference
mkdir -p ../lib
cp target/lstmae_types.jar ../lib/lstmae_types.jar
echo "Copied: ../lib/lstmae_types.jar"

echo ""
echo "=== Step 5: Install types JAR into Striim ==="
echo ""

STRIIM_HOME="${STRIIM_HOME:-/opt/Striim}"
cp target/lstmae_types.jar "$STRIIM_HOME/lib/lstmae_types.jar"
echo "Copied to $STRIIM_HOME/lib/lstmae_types.jar"

echo ""
echo "=== Step 6: Copy .scm to Striim modules directory ==="
echo ""

cp target/LSTMAEScoreCaller.scm "$STRIIM_HOME/modules/LSTMAEScoreCaller.scm"
echo "Copied to $STRIIM_HOME/modules/LSTMAEScoreCaller.scm"

echo ""
echo "IMPORTANT: Striim must be restarted after placing the types JAR in lib/."
echo ""
echo "  $STRIIM_HOME/bin/server.sh stop"
echo "  $STRIIM_HOME/bin/server.sh start"
echo ""
echo "Then in the Striim Console:"
echo ""
echo '  LOAD OPEN PROCESSOR "/opt/Striim/modules/LSTMAEScoreCaller.scm";'
echo ""
echo "=== Done ==="
