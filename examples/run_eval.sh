


# example to run: ./run_eval.sh ./nas/mfasc/evaluation/mfasc_hf-random-no_seed.yml 10
for i in $(eval echo {1..$2})
do
	echo "Loop $i arg: $1"
	python3 ./run_example.py $1 pytorch
done