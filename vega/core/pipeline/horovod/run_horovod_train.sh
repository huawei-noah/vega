basepath=$(cd `dirname $0`; pwd)
SCRIPT_PATH=${basepath}/horovod_train.py
nps=$1
IP_ADDRESS=$3
PYTHON_COMMAND=$4

IFS=',' read -ra IP_ARRAY <<< "$IP_ADDRESS"

gpu_per_node=$(($nps / ${#IP_ARRAY[@]}))
for i in $(seq 0 $((${#IP_ARRAY[@]}-1)))
do
    IP_ARRAY[i]="${IP_ARRAY[i]}:$gpu_per_node"
done

IFS="," eval 'server_list="${IP_ARRAY[*]}"'

run_experiment() {
    local np=$1
    shift
    horovodrun --start-timeout 300 -np $np -H $server_list $@
}

run_experiment $nps $PYTHON_COMMAND $SCRIPT_PATH --cf_file $2
