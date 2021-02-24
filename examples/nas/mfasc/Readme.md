# Multi-fidelity NAS with co-kriging.

It is an individual optimization method (no weight sharing), that works with arbitrary search spaces.
The search algorithm is described in the paper https://dl.acm.org/doi/10.1145/3292500.3330893
In this example we use MobileNetV2 search space.

To run the example, use:
```bash
cd examples
python3 ./run_example.py ./nas/mfasc/mfasc.yml pytorch
```

Search algorithm paramters:
* max_budget - the total number of epochs to be trained
* hf_epochs - the number of epochs to train for high-fidelity evaluation
* lf_epochs - the number of epochs to train for low-fidelity evaluation
* fidelity_ratio - amount of low-fidelity evaluations per one high-fidelity evaluation
* min_hf_sample_size - initial sample size of models evaluated with high-fidelity (must be >=2)
* min_lf_sample_size - initial sample size of models evaluated with low-fidelity (must be >=2)

The best model is saved in the file ```best_model_desc``` in the local task directory.

