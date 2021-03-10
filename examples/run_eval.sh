
for i in $(eval echo {1..$2})
do
	echo "Loop $i arg: $1"
	python3 ./run_pipeline.py $1
done