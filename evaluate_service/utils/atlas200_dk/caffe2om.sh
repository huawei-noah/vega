MODEL_PATH=$1
WEIGHT_PATH=$2
OUTPUT_PATH=$3
omg  --model=$MODEL_PATH  --weight=$WEIGHT_PATH  --framework=0 --output=$OUTPUT_PATH/davinci_model  > $OUTPUT_PATH/omg.log
