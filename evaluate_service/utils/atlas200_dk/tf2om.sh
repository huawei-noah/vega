MODEL_PATH=$1
OUTPUT_PATH=$2
omg  --model=$MODEL_PATH --framework=3 --output=$OUTPUT_PATH/davinci_model  > $OUTPUT_PATH/omg.log
