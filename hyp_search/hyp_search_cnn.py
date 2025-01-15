import os

main_path = r"C:\Users\ZC\Documents\GitHub\IIB_Project\main.py"
config_path = r"C:\Users\ZC\Documents\GitHub\IIB_Project\configs\cnn.yaml"
lr_vals = [0.0001, 0.0005, 0.001]
layer_vals = [3, 4]

for layer in layer_vals:
    for lr in lr_vals:
        command = (f"python {main_path} \
            -c {config_path} \
            --seed {1} \
            --layer_num {layer} \
            --lr {lr}")
        os.system(command)
