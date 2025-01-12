import os


main_path = r"C:\Users\caizi\Documents\GitHub\IIB_Project\main.py"
config_path = r"C:\Users\caizi\Documents\GitHub\IIB_Project\configs\base.yaml"
lr_vals = [0.0001, 0.0005, 0.001]
conv_kernel_size_vals = [4, 8]
conv_stride_vals = [1, 2]
pool_kernel_size_vals = [4, 8]
pool_stride_vals = [1, 2]
freq_encoder_hidden_sizes = [[32], [128, 32], [256, 32]]
joint_head_hidden_sizes = [[16], [16, 16]]
act_fn = ['relu', 'tanh']


# Time Component
for conv_kernel_size in conv_kernel_size_vals:
    for conv_stride in conv_stride_vals:
        for pool_kernel_size in pool_kernel_size_vals:
            for pool_stride in pool_stride_vals:
                for hsizes in joint_head_hidden_sizes:
                    hsizes = " ".join(map(str, hsizes))
                    for fn in act_fn:
                        for lr in lr_vals:
                            command = (
                                f"python {main_path} \
                                -c {config_path} \
                                --seed {1} \
                                --conv_kernel_size {conv_kernel_size} \
                                --conv_stride {conv_stride} \
                                --pool_kernel_size {pool_kernel_size} \
                                --pool_stride {pool_stride}\
                                --joint_head_hidden_sizes {hsizes} \
                                --act_fn {fn} \
                                --lr {lr}")
                            os.system(command)


# Freq Component
# for fsizes in freq_encoder_hidden_sizes:
#     fsizes = " ".join(map(str, fsizes))
#     for hsizes in joint_head_hidden_sizes:
#         hsizes = " ".join(map(str, hsizes))
#         for fn in act_fn:
#             for lr in lr_vals:
#                 command = (
#                     f"python {main_path} \
#                     -c {config_path} \
#                     --seed {1} \
#                     --freq_encoder_hidden_sizes {fsizes} \
#                     --joint_head_hidden_sizes {hsizes} \
#                     --act_fn {fn} \
#                     --lr {lr}")
#                 os.system(command)


# Time and Freq Components
# for conv_kernel_size in conv_kernel_size_vals:
#     for conv_stride in conv_stride_vals:
#         for pool_kernel_size in pool_kernel_size_vals:
#             for pool_stride in pool_stride_vals:
#                 for fsizes in freq_encoder_hidden_sizes:
#                     fsizes = " ".join(map(str, fsizes))
#                     for hsizes in joint_head_hidden_sizes:
#                         hsizes = " ".join(map(str, hsizes))
#                         for fn in act_fn:
#                             for lr in lr_vals:
#                                 command = (
#                                     f"python {main_path} \
#                                     -c {config_path} \
#                                     --seed {1} \
#                                     --conv_kernel_size {conv_kernel_size} \
#                                     --conv_stride {conv_stride} \
#                                     --pool_kernel_size {pool_kernel_size} \
#                                     --pool_stride {pool_stride}\
#                                     --freq_encoder_hidden_sizes {fsizes} \
#                                     --joint_head_hidden_sizes {hsizes} \
#                                     --act_fn {fn} \
#                                     --lr {lr}")
#                                 os.system(command)