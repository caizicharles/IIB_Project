import os

main_path = r"C:\Users\ZC\Documents\GitHub\IIB_Project\main.py"
config_path = r"C:\Users\ZC\Documents\GitHub\IIB_Project\configs\base.yaml"

layer_num = [3, 4]
act_fn = ['relu', 'tanh']
lr_vals = [0.0001, 0.0005, 0.001]


for layer in layer_num:
    for fn in act_fn:
        for lr in lr_vals:
            command = (f"python {main_path} \
                -c {config_path} \
                --seed {1} \
                --layer_num {layer}\
                --act_fn {fn} \
                --lr {lr}")
            os.system(command)


# --joint_head_hidden_sizes {hsizes} \