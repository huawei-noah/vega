export ASCEND_HOME=/usr/local/Ascend
export PATH=/usr/local/python3.7/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/atc/python/site-packages/te.egg/lib:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:/usr/local/Ascend/atc/lib64/plugin/opskernel
export PYTHONPATH=$PYTHONPATH:$ASCEND_HOME/atc/python/site-packages/te.egg:$ASCEND_HOME/atc/python/site-packages/topi.egg:$ASCEND_HOME/atc/python/site-packages/auto_tune.egg
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export SLOG_PRINT_TO_STDOUT=1
#export DUMP_GE_GRAPH=1
#export DUMP_OP=1
