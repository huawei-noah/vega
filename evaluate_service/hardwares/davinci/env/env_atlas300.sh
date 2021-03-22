export ASCEND_HOME=/usr/local/Ascend
export PATH=/opt/cmake-3.14.5-Linux-x86_64/bin:/usr/local/python3.7.5/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/atc/python/site-packages/te.egg/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons
export PYTHONPATH=$PYTHONPATH:$ASCEND_HOME/atc/python/site-packages/te.egg:$ASCEND_HOME/atc/python/site-packages/topi.egg:$ASCEND_HOME/atc/python/site-packages/auto_tune.egg
export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/20.2.0/x86_64-linux/opp
export DDK_PATH=$ASCEND_HOME
export NPU_HOST_LIB=$ASCEND_HOME/ascend-toolkit/20.2.0/x86_64-linux/acllib/lib64/stub