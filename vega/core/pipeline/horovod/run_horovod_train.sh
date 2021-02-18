#!/usr/bin/env bash
# This script runs the Horovod training job on the modelarts platform or cluster.
# basepath=$(cd `dirname $0`; pwd)
# SCRIPT_PATH=${basepath}/horovod_train.py
# run_experiment() {
#     local np=$1
#     shift
#     mpirun -np $np \
#       --hostfile ${HOST_FILE_PATH} \
#       -bind-to socket \
#       -x NCCL_DEBUG=INFO -x MPI_HOME -x LD_LIBRARY_PATH -x PATH \
#       -x HOROVOD_MPI_THREADS_DISABLE=1 \
#       -mca plm_rsh_no_tree_spawn true \
#       $@
# }
# nps=$1
# run_experiment $nps python3 $SCRIPT_PATH --cf_file $2

basepath=$(cd `dirname $0`; pwd)
SCRIPT_PATH=${basepath}/horovod_train.py
nps=$1
IP_ADDRESS=$3

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
    horovodrun -np $np -H $server_list $@
}

run_experiment $nps python3 $SCRIPT_PATH --cf_file $2
