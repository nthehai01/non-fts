export DS_DIR='/Users/hainguyen/Documents/AI4Code_data'

export RAW_DATA_DIR="$DS_DIR/ai4code"
export PROCESSED_DATA_DIR="$DS_DIR/proc-ai4code"
export SEED=42
export PICKLE_PROTOCOL=4
export NON_MASKED_INDEX=-100
export MAX_GRADIENT=10.0
export MLM_PROBABILITY=0.15
export PICKLE_PROTOCOL=4


clearenv () {
    unset RAW_DATA_DIR
    unset PROCESSED_DATA_DIR
    unset SEED
    unset PICKLE_PROTOCOL
    unset NON_MASKED_INDEX
    unset MAX_GRADIENT
    unset MLM_PROBABILITY
    unset PICKLE_PROTOCOL
}
