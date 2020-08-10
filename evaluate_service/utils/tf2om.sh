MODEL_PATH=$1
OUTPUT_PATH=$2
atc --model=$MODEL_PATH  --framework=3  --input_format='NHWC'  --disable_reuse_memory=1 --head_stream=1 --output=$OUTPUT_PATH/tf_convert --soc_version=Ascend310 --core_type=AiCore   >$OUTPUT_PATH/omg.log
