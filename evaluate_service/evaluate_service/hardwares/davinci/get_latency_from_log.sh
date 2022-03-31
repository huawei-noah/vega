LOG_FILE=$1
err_num=`cat $LOG_FILE | grep ERROR |wc -l`
if [[ $err_num == 0 ]];then
  cat $LOG_FILE |grep "Inference time:" | awk -F  ' '  '{print $NF}' |awk -F 'ms'  '{print $1}'
else
  echo None
fi
