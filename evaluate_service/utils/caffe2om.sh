MODEL_PATH=$1
WEIGHT_PATH=$2
OUTPUT_PATH=$3
atc --model=$MODEL_PATH --weight=$WEIGHT_PATH --framework=0 --input_fp16_nodes=data --output_type=FP16 --input_format='NCHW' --disable_reuse_memory=1 --head_stream=1 --output=$OUTPUT_PATH/caffe_convert --soc_version=Ascend310 --core_type=AiCore    >$OUTPUT_PATH/omg.log
