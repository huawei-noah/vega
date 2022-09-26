**Evaluate Service ver1.8.5 released:**

**Introduction**

The evaluation service is a tool used to evaluate the performance of a 
model on specific hardware  developed by Noah's Ark Laboratory, the main features are as follows:
1. Multi-Backend: PyTorch, TensorFlow, MindSpore and Caffe. The input model can come form Pytorch, Tensorflow,
Mindspore and caffe.
2. Multi-hardware: Mobile Phone, Ascend 310, Kirinri 990 and etc. The model can be evaluated on multiple types of hardware.
3. Online real-time evaluation and offline evaluation. The evaluate service can combine with [Vega](https://github.com/huawei-noah/vega)
to implement network architecture search with hardware in the ring. One can also use the evaluate service independently.
4. Supports secure communication encryption. In security mode, communications are encrypted to secure the model and data.

**Installation**

Install evaluate service and the open source softwares that evaluate service depends on:

`pip3 install --user --upgrade evaluate-service`

**Cooperation and Contribution**

Welcome to use evaluate-service. If you have any questions or suggestions, need help, fix bugs, contribute new algorithms, 
or improve the documentation, submit an issue in the community. We will reply to and communicate with you in a timely manner.  
