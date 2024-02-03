# deep_learning
a common deep learning train framework using pytorch DDP

# usage
+ requiremments: [requirements.txt](./requirements.txt)
+ run
    - run like this
    ```
    usage: main [-h] [--config CONFIG] [--mode {train,test}] [--gpus GPUS] [--port PORT]

    options:
    -h, --help           show this help message and exit
    --config CONFIG      config file
    --mode {train,test}  run mode
    --gpus GPUS          gpu ids, such as: 0,1,2,3
    --port PORT          master port
    ```
+ to know more details, you can run the code step by step
