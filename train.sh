

export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=OFF

# Use the directory containing this script as the project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Supported tasks.
AVAILABLE_TASKS="uhdblur uhdrain uhdhaze uhdlol uhdsnow"

# 1. Determine task.
TASK=${1}
if [ -z "$TASK" ]; then
    echo "Available tasks: uhdblur, uhdrain, uhdhaze, uhdlol, uhdsnow"
    read -p "Enter task name (default: uhdblur): " TASK
    TASK=${TASK:-uhdblur}
fi

# Validate task.
if [[ ! " $AVAILABLE_TASKS " =~ " $TASK " ]]; then
    echo "Invalid task: $TASK"
    echo "Available tasks: uhdblur, uhdrain, uhdhaze, uhdlol, uhdsnow"
    exit 1
fi

CONFIG="options/train_CoDeSSM_${TASK}.yml"
DEFAULT_EXP_DIR="experiments/CoDeSSM_${TASK}"
TMP_RESUME_YML="/tmp/train_CoDeSSM_${TASK}_resume.yml"
MASTER_PORT=${MASTER_PORT:-4337}

# 2. Check required files.
echo "=========================================="
echo "Checking required files..."
echo "=========================================="

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [MISSING] $1"
        return 1
    fi
}

check_file "basicsr/archs/code_arch.py" || exit 1
check_file "$CONFIG" || {
    echo "Config file does not exist: $CONFIG"
    echo "Please create it based on an existing CoDeSSM config."
    exit 1
}
echo ""

# Resolve experiment directory from the YAML config.
EXP_DIR=$(python3 - "$CONFIG" "$DEFAULT_EXP_DIR" <<'EOF'
import os
import sys
import yaml

config_path = sys.argv[1]
default_exp_dir = sys.argv[2]

try:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    path_opt = cfg.get("path") or {}
    experiments_root = path_opt.get("experiments_root", "experiments")
    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]

    print(os.path.join(experiments_root, name))
except Exception:
    print(default_exp_dir)
EOF
)

echo "Task: $TASK"
echo "Config: $CONFIG"
echo "Experiment directory: $EXP_DIR"
echo ""

# 3. GPU configuration.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    read -p "Enter GPU IDs, comma separated (default: 0): " GPU_IDS
    GPU_IDS=${GPU_IDS:-"0"}
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
else
    GPU_IDS="$CUDA_VISIBLE_DEVICES"
fi

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "Using GPUs: $GPU_IDS ($NUM_GPUS device(s))"
echo ""

# 4. Interactive menu.
while true; do
    echo "=========================================="
    echo "Task: $TASK | Config: $CONFIG"
    echo "Experiment directory: $EXP_DIR"
    echo "Select an option:"
    echo "  1) Train from scratch"
    echo "  2) Resume from checkpoint"
    echo "  3) Exit"
    echo "=========================================="
    read -p "Enter option [1/2/3]: " choice

    case $choice in
        1)
            mkdir -p "$EXP_DIR"

            echo "Starting training from scratch..."
            torchrun \
                --standalone \
                --nproc_per_node="$NUM_GPUS" \
                --master_port="$MASTER_PORT" \
                basicsr/train.py -opt "$CONFIG" --launcher pytorch
            ;;

        2)
            STATE_DIR="$EXP_DIR/training_states"
            LATEST_STATE=""

            echo "Looking for .state files in: $STATE_DIR"

            if [ -d "$STATE_DIR" ]; then
                LATEST_STATE=$(ls -t "$STATE_DIR"/*.state 2>/dev/null | head -1)
            fi

            # If not found, search other experiment directories related to this task.
            if [ -z "$LATEST_STATE" ]; then
                SEARCH_ROOT=$(dirname "$EXP_DIR")
                echo "No .state file found in: $STATE_DIR"
                echo "Searching task-related checkpoints under: $SEARCH_ROOT"
                echo "Search pattern: *${TASK}*/training_states/*.state"

                LATEST_STATE=$(find "$SEARCH_ROOT" \
                    -type f \
                    -path "*${TASK}*/training_states/*.state" \
                    -printf '%T@ %p\n' 2>/dev/null \
                    | sort -nr \
                    | head -1 \
                    | sed 's/^[^ ]* //')
            fi

            if [ -n "$LATEST_STATE" ]; then
                echo "Found latest checkpoint: $LATEST_STATE"
                read -p "Use this checkpoint? [Y/n]: " use_latest

                if [[ "$use_latest" =~ ^[Nn]$ ]]; then
                    read -p "Enter .state checkpoint path: " resume_path
                else
                    resume_path="$LATEST_STATE"
                fi
            else
                read -p "No checkpoint found. Enter .state checkpoint path: " resume_path
            fi

            if [ -z "$resume_path" ] || [ ! -f "$resume_path" ]; then
                echo "Invalid checkpoint file."
                continue
            fi

            echo "Using checkpoint: $resume_path"

            echo "Generating temporary resume config..."
            python3 - "$CONFIG" "$TMP_RESUME_YML" "$resume_path" <<'EOF'
import os
import sys
import yaml

config_path = sys.argv[1]
tmp_config_path = sys.argv[2]
resume_path = sys.argv[3]

try:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = cfg.get("path") or {}
    cfg["path"]["resume_state"] = resume_path

    state_dir = os.path.dirname(resume_path)
    exp_dir = os.path.dirname(state_dir)
    basename = os.path.basename(resume_path)
    iter_num = basename.split(".")[0]

    model_path = os.path.join(exp_dir, "models", f"net_g_{iter_num}.pth")

    if os.path.exists(model_path):
        cfg["path"]["pretrain_network_g"] = model_path
        print(f"Found generator weights: {model_path}")
    else:
        print(f"Warning: generator weights not found: {model_path}", file=sys.stderr)
        cfg["path"]["pretrain_network_g"] = None

    cfg["path"]["pretrain_network_hq"] = None
    cfg["path"]["pretrain_network_d"] = None

    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

except Exception as e:
    print(f"Failed to process YAML: {e}", file=sys.stderr)
    sys.exit(1)
EOF

            if [ $? -ne 0 ]; then
                echo "Failed to generate resume config."
                continue
            fi

            echo "Temporary resume config saved to: $TMP_RESUME_YML"

            mkdir -p "$EXP_DIR"

            echo "Starting resume training..."
            torchrun \
                --standalone \
                --nproc_per_node="$NUM_GPUS" \
                --master_port="$MASTER_PORT" \
                basicsr/train.py -opt "$TMP_RESUME_YML" --launcher pytorch
            ;;

        3)
            echo "Exited."
            exit 0
            ;;

        *)
            echo "Invalid option. Please enter 1, 2, or 3."
            ;;
    esac
done