import os


config_files = ['base']
lr_vals = [0.0001, 0.0005, 0.001, 0.005, 0.01]
conv_kernel_size_vals = [4, 8, 16]
conv_stride_vals = [1]
pool_kernel_size_vals = [4, 8, 16]
pool_stride_vals = [1]

seeds = [1]

for file in config_files:
    for conv_kernel_size in conv_kernel_size_vals:
        for conv_stride in conv_stride_vals:
            for pool_kernel_size in pool_kernel_size_vals:
                for pool_stride in pool_stride_vals:
                    for lr in lr_vals:
                        for seed in seeds:
                            command = (
                                f"python C:\Users\caizi\Documents\GitHub\IIB_Project\main.py -c C:\Users\caizi\Documents\GitHub\IIB_Project\configs\{file}.yaml \
                                --seed {seed} --lr {lr} --conv_kernel_size {conv_kernel_size} --conv_stride {conv_stride} --pool_kernel_size {pool_kernel_size} --pool_stride {pool_stride}"
                            )
                            os.system(command)