#! /bin/bash -eux

# Script to build, deploy, and run OpenCL Rene model on Android
# Supports both text input (new) and binary token file input (backward compatible)
#
# Usage:
#   ./copy_and_run.sh "Hello world" weights_dir output.bin 10  # Text input (max_tokens optional, default: 10)
#   ./copy_and_run.sh tokens.bin weights_dir output.bin 10      # Binary token file (max_tokens optional, default: 10)

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find Android NDK
ANDROID_NDK=$(find "$HOME/Library/Android/sdk/ndk" -maxdepth 1 -type d | sort -V | tail -1)

# Find adb in Android SDK platform-tools
ADB_PATH="$HOME/Library/Android/sdk/platform-tools/adb"
if [ ! -f "$ADB_PATH" ]; then
    echo "Error: adb not found at $ADB_PATH"
    echo "Please ensure Android SDK platform-tools is installed"
    exit 1
fi

# Use relative paths
BINARY_PATH="$SCRIPT_DIR/build_android_standalone/cartesia_opencl_test"
# Use MLX-generated prompt tokens from example_debug.py (fallback)
MLX_OUTPUT_DIR="$SCRIPT_DIR/../cartesia-mlx/mlx_debug_outputs"
PROMPT_TOKENS_PATH="$MLX_OUTPUT_DIR/prompt_tokens.bin"
# Remove trailing slash from weights_dir if present
WEIGHTS_DIR_RAW="${2:-$SCRIPT_DIR/mamba2_130m_weights/}"
WEIGHTS_DIR="${WEIGHTS_DIR_RAW%/}"
OUTPUT_FILE="${3:-/data/local/tmp/output_opencl_mamba2.bin}"
MAX_TOKENS="${4:-10}"
MLX_GENERATED_TOKENS="$MLX_OUTPUT_DIR/generated_tokens.bin"

# Parse first argument: detect if it's a file or text
# Default to "Rene Descartes was" for consistency with MLX runs
INPUT_ARG="${1:-}"
IS_TEXT_INPUT=false
TOKEN_FILE_PATH=""

if [ -z "$INPUT_ARG" ]; then
    # No argument provided - use default text prompt
    IS_TEXT_INPUT=true
    TEXT_INPUT="Rene Descartes was"
elif [ -f "$INPUT_ARG" ]; then
    # File exists - treat as binary token file
    TOKEN_FILE_PATH="$INPUT_ARG"
    IS_TEXT_INPUT=false
else
    # Not a file - treat as text input
    IS_TEXT_INPUT=true
    TEXT_INPUT="$INPUT_ARG"
fi

if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found at $BINARY_PATH"
    echo "Please build first: cd $SCRIPT_DIR && ./build_android.sh"
    exit 1
fi

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Error: Weights directory not found at $WEIGHTS_DIR"
    echo "Please run export_mlx_weights.py first to generate weights"
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/embedding_weight.bin" ]; then
    echo "Error: embedding_weight.bin not found in $WEIGHTS_DIR"
    exit 1
fi

# Handle text input: check for tokenizer files
if [ "$IS_TEXT_INPUT" = true ]; then
    echo "Detected text input: \"$TEXT_INPUT\""
    
    # Check for tokenizer files
    TOKENIZER_DIR="$WEIGHTS_DIR/tokenizer"
    VOCAB_PATH="$TOKENIZER_DIR/vocab.json"
    MERGES_PATH="$TOKENIZER_DIR/merges.txt"
    
    # Try tokenizer subdirectory first, then weights_dir directly
    if [ ! -f "$VOCAB_PATH" ] || [ ! -f "$MERGES_PATH" ]; then
        VOCAB_PATH="$WEIGHTS_DIR/vocab.json"
        MERGES_PATH="$WEIGHTS_DIR/merges.txt"
    fi
    
    if [ ! -f "$VOCAB_PATH" ] || [ ! -f "$MERGES_PATH" ]; then
        echo "Error: Tokenizer files not found for text input"
        echo "Expected:"
        echo "  $WEIGHTS_DIR/tokenizer/vocab.json and merges.txt"
        echo "  Or: $WEIGHTS_DIR/vocab.json and merges.txt"
        echo ""
        echo "To get tokenizer files, run:"
        echo "  python3 -c \"from transformers import AutoTokenizer; import json; t = AutoTokenizer.from_pretrained('allenai/OLMo-1B-hf'); import os; os.makedirs('$TOKENIZER_DIR', exist_ok=True); json.dump(t.backend_tokenizer.model.vocab, open('$VOCAB_PATH', 'w')); open('$MERGES_PATH', 'w').write('\\n'.join(t.backend_tokenizer.model.merges))\""
        echo ""
        echo "Or use binary token file input instead."
        exit 1
    fi
    
    echo "Found tokenizer files:"
    echo "  Vocab: $VOCAB_PATH"
    echo "  Merges: $MERGES_PATH"
else
    # Binary token file input
    if [ ! -f "$TOKEN_FILE_PATH" ]; then
        echo "Error: Token file not found at $TOKEN_FILE_PATH"
        exit 1
    fi
    echo "Using binary token file: $TOKEN_FILE_PATH"
fi

if [ ! -f "$MLX_GENERATED_TOKENS" ]; then
    echo "Warning: MLX generated_tokens.bin not found at $MLX_GENERATED_TOKENS"
    echo "This is needed for comparison. Run example_debug.py first."
fi

echo ""
echo "Pushing files to Android device..."

# Push binary
"$ADB_PATH" push "$BINARY_PATH" /data/local/tmp/
"$ADB_PATH" shell chmod +x /data/local/tmp/cartesia_opencl_test

# Push CLBlast library if available
CLBLAST_LIB_HOST="${CLBLAST_LIB:-$SCRIPT_DIR/third_party/CLBlast/build-android-arm64-v8a/install/lib/libclblast.so}"
if [ ! -f "$CLBLAST_LIB_HOST" ] && [ -f "$SCRIPT_DIR/third_party/CLBlast/build-android-arm64-v8a/install/lib64/libclblast.so" ]; then
    CLBLAST_LIB_HOST="$SCRIPT_DIR/third_party/CLBlast/build-android-arm64-v8a/install/lib64/libclblast.so"
fi

if [ -f "$CLBLAST_LIB_HOST" ]; then
    echo "Pushing CLBlast library: $CLBLAST_LIB_HOST"
    "$ADB_PATH" push "$CLBLAST_LIB_HOST" /data/local/tmp/libclblast.so >/dev/null
else
    echo "Warning: CLBlast library not found locally; runtime will fall back to OpenCL kernels."
fi

# Push OpenCL library if bundled
HOST_OPENCL_LIB="${HOST_OPENCL_LIB:-$SCRIPT_DIR/libOpenCL.so}"
if [ -f "$HOST_OPENCL_LIB" ]; then
    echo "Pushing OpenCL library: $HOST_OPENCL_LIB"
    "$ADB_PATH" push "$HOST_OPENCL_LIB" /data/local/tmp/libOpenCL.so >/dev/null
fi

# Push weights directory structure (if not already pushed)
WEIGHTS_DEVICE_DIR="/data/local/tmp/$(basename "$WEIGHTS_DIR")"
"$ADB_PATH" shell "mkdir -p $WEIGHTS_DEVICE_DIR" 2>/dev/null || true

# Push weights directory
echo "NOT Pushing weights from $WEIGHTS_DIR to device..."
#"$ADB_PATH" push "$WEIGHTS_DIR" /data/local/tmp/
#echo "✓ Weights pushed"

# Push tokenizer files if text input
if [ "$IS_TEXT_INPUT" = true ]; then
    echo "Pushing tokenizer files..."
    TOKENIZER_DEVICE_DIR="$WEIGHTS_DEVICE_DIR/tokenizer"
    "$ADB_PATH" shell "mkdir -p $TOKENIZER_DEVICE_DIR" 2>/dev/null || true
    
    # Determine which tokenizer directory we found
    if [ -f "$WEIGHTS_DIR/tokenizer/vocab.json" ]; then
        TOKENIZER_SRC_DIR="$WEIGHTS_DIR/tokenizer"
    else
        TOKENIZER_SRC_DIR="$WEIGHTS_DIR"
        TOKENIZER_DEVICE_DIR="$WEIGHTS_DEVICE_DIR"
    fi
    
    "$ADB_PATH" push "$TOKENIZER_SRC_DIR/vocab.json" "$TOKENIZER_DEVICE_DIR/"
    "$ADB_PATH" push "$TOKENIZER_SRC_DIR/merges.txt" "$TOKENIZER_DEVICE_DIR/"
    echo "✓ Tokenizer files pushed"
fi

# Push token file if binary input
if [ "$IS_TEXT_INPUT" = false ]; then
    "$ADB_PATH" push "$TOKEN_FILE_PATH" /data/local/tmp/prompt_tokens_mamba2.bin
fi

echo ""
echo "Verifying weights were pushed..."
"$ADB_PATH" shell "ls -lh $WEIGHTS_DEVICE_DIR/embedding_weight.bin" || echo "WARNING: embedding_weight.bin not found!"

echo ""
echo "Running OpenCL binary with mamba2-130m..."

REMOTE_ENV="LD_LIBRARY_PATH=/data/local/tmp"
if [ -n "${CARTESIA_PROFILE:-}" ]; then
    REMOTE_ENV="$REMOTE_ENV CARTESIA_PROFILE=${CARTESIA_PROFILE}"
fi

if [ "$IS_TEXT_INPUT" = true ]; then
    # Text input: pass text directly (properly quoted)
    # Escape quotes in text for adb shell
    ESCAPED_TEXT=$(echo "$TEXT_INPUT" | sed "s/'/'\\''/g")
    REMOTE_CMD="/data/local/tmp/cartesia_opencl_test '$ESCAPED_TEXT' $WEIGHTS_DEVICE_DIR $OUTPUT_FILE $MAX_TOKENS"
else
    # Binary token file input
    REMOTE_CMD="/data/local/tmp/cartesia_opencl_test /data/local/tmp/prompt_tokens_mamba2.bin $WEIGHTS_DEVICE_DIR $OUTPUT_FILE $MAX_TOKENS"
fi

if [ -n "$REMOTE_ENV" ]; then
    REMOTE_CMD="$REMOTE_ENV $REMOTE_CMD"
fi

"$ADB_PATH" shell "$REMOTE_CMD"

exit

echo ""
echo "Pulling OpenCL output..."
OUTPUT_LOCAL="$SCRIPT_DIR/output_opencl_mamba2.bin"
"$ADB_PATH" pull "$OUTPUT_FILE" "$OUTPUT_LOCAL" || echo "Warning: Failed to pull output file"

echo ""
if [ -f "$OUTPUT_LOCAL" ]; then
    echo "✓ Output file pulled to: $OUTPUT_LOCAL"
    
    # Show raw token IDs first
    echo ""
    echo "Raw generated token IDs:"
    echo "=================================================================================="
    if command -v python3 &> /dev/null; then
        python3 << EOF
import numpy as np
import sys
try:
    tokens = np.fromfile('$OUTPUT_LOCAL', dtype=np.int32)
    print(f'Number of tokens: {len(tokens)}')
    print(f'Token IDs: {list(tokens)}')
except Exception as e:
    print(f'Error reading tokens: {e}')
EOF
    else
        echo "Python3 not available, skipping token dump"
    fi
    echo "=================================================================================="
    
    # Decode tokens to English text
    echo ""
    echo "Decoding generated tokens to English text..."
    echo "=================================================================================="
    if [ -f "$SCRIPT_DIR/tools/decode_tokens.py" ]; then
        # Use the venv if available (try multiple possible locations)
        VENV_ACTIVATE=""
        if [ -f "$SCRIPT_DIR/../../venv_mlx_arm/bin/activate" ]; then
            VENV_ACTIVATE="$SCRIPT_DIR/../../venv_mlx_arm/bin/activate"
        elif [ -f "$SCRIPT_DIR/../../../venv_mlx_arm/bin/activate" ]; then
            VENV_ACTIVATE="$SCRIPT_DIR/../../../venv_mlx_arm/bin/activate"
        fi
        
        if [ -n "$VENV_ACTIVATE" ]; then
            source "$VENV_ACTIVATE" 2>/dev/null || true
        fi
        
        python3 "$SCRIPT_DIR/tools/decode_tokens.py" "$OUTPUT_LOCAL" \
            --tokenizer "allenai/OLMo-1B-hf" \
            --format "simple" \
            --verbose || echo "Warning: Failed to decode tokens"
        echo "=================================================================================="
    else
        echo "Warning: decode_tokens.py not found at $SCRIPT_DIR/tools/decode_tokens.py"
    fi
    
    # Compare with MLX output if available
    if [ -f "$MLX_GENERATED_TOKENS" ]; then
        echo ""
        echo "Comparing outputs..."
        echo "MLX output: $MLX_GENERATED_TOKENS"
        echo "OpenCL output: $OUTPUT_LOCAL"
        echo ""
        cd "$SCRIPT_DIR/../test_validation"
        if [ -f "compare_decoded_outputs.py" ]; then
            python3 compare_decoded_outputs.py \
                "$MLX_GENERATED_TOKENS" \
                "$OUTPUT_LOCAL" || true
        fi
        
        if [ -f "find_first_mismatch.py" ]; then
            echo ""
            echo "Finding first mismatch..."
            python3 find_first_mismatch.py \
                "$MLX_OUTPUT_DIR" \
                "$OUTPUT_LOCAL" || true
        fi
    else
        echo ""
        echo "Note: MLX output not available for comparison"
        echo "  To generate: cd ../cartesia-mlx && python example_debug.py --model cartesia-ai/mamba2-130m-mlx --prompt 'Rene Descartes was' --max-tokens $MAX_TOKENS --dump-outputs"
    fi
else
    echo "Warning: Output file not found locally"
    echo "  Device path: $OUTPUT_FILE"
    echo "  Local path: $OUTPUT_LOCAL"
fi
