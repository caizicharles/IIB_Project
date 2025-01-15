import os

main_path = r"C:\Users\ZC\Documents\GitHub\IIB_Project\main.py"
config_path = r"C:\Users\ZC\Documents\GitHub\IIB_Project\configs\transformer.yaml"
lr_vals = [0.0001, 0.0005, 0.001]
encoder_depths = [1, 2, 3]

for depth in encoder_depths:
    for lr in lr_vals:
        command = (f"python {main_path} \
            -c {config_path} \
            --seed {1} \
            --encoder_depth {depth} \
            --lr {lr}")
        os.system(command)
        exit()
