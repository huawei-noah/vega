# Developer Guide

The Vega framework components are decoupled and the registration mechanism is used to combine functional components to facilitate function and algorithm expansion. For details about the Vega architecture and main mechanisms, see **[Developer Guide](./developer_guide.md)**.

You can also refer to **[Quick Start Guide](./quick_start.md)** to implement a simple CNN network search function and quickly enter Vega application development.

During Vega application development, the first problem encountered is how to import service data sets to Vega. For details, see **[Datasets Guide](./datasets.md)**.

For different algorithms, see **[Algorithm Development Guide](./new_algorithm.md)**. You can add new algorithms to Vega step by step based on the examples provided in this document.

In most Automl algorithms, the search space and network are strongly correlated. We will try to unify the definition of search space so that the same search space can adapt to different search algorithms. This is called **[Fine-grained Search Space Guide](./fine_grained_space.md)**. Welcome to use this method.
