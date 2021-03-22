EXAMPLE_DIR=$1
SAVE_PATH=$2

cd $EXAMPLE_DIR/
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make

cd ../../../out

mv ./main $SAVE_PATH
cp ../src/acl.json $SAVE_PATH