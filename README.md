# MmAF
Multi-machine adaptive freezing algorithm code for enhanced pipeline parallel distributed training in 6G networks

#### 1、go to the specified address to download the pre-trained BERT model.

https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz

#### 2、Prepare three independent hosts.

In this experiment, we selected the RTX 1080 Ti, RTX 2080 Ti, and RTX 3070.
Copy the code to three different hosts separately.

#### 3、Modify the relevant configurations.

In `test1.py`, `test2.py`, and `test3.py`, modify the following parameters accordingly.

```python
os.environ['MASTER_ADDR'] = 'your_host_ip'  # Replace with your host IP
os.environ['MASTER_PORT'] = 'desired_port'  # Replace with the port you want to use for communication
os.environ['TP_SOCKET_IFNAME'] = 'your_network_interface'  # Replace with the network interface name used for communication2'
```

We recommend disabling the firewall between the three hosts.

#### 4、Run the code

Run `test1.py`, `test2.py`, and `test3.py` separately on each host.