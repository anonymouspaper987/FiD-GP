resnet: python scripts/cifar_train.py --inference-config=configs/ffg_u_cifar100.json --num-epochs=200 --ml-epochs=100 --annealing-epochs=50 --lr=1e-3 --milestones=100 --resnet=18 --cifar=100  

imagenet: python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 scripts/imagenet_train.py --dist-url tcp://127.0.0.1:29500 --resnet 18 --batch-size 100 --use-prefetcher --num-workers 8 --multiprocessing-distributed --output-dir output_resnet18_imagenet_4gpu
