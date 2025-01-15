import os
import os.path as osp
import argparse


def create_project_structure(working_dir=""):
    datasets = ['Feas2', 'Trial']

    tasks = ['prediction']

    models = ['BaseModel', 'GRU', 'LSTM', 'Transformer']

    for dataset in datasets:
        path = osp.join(working_dir, 'data/raw', dataset)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
        path = osp.join(working_dir, 'data/processed', dataset)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

    for dataset in datasets:
        for task in tasks:
            for model in models:
                path = osp.join(working_dir, 'logs', 'bootstrap_results', dataset, task, model)
                os.makedirs(path, exist_ok=True)
                print(f"Created: {path}")
                path = osp.join(working_dir, 'logs', 'checkpoints', dataset, task, model)
                os.makedirs(path, exist_ok=True)
                print(f"Created: {path}")

    path = osp.join(working_dir, 'logs', 'mlflow')
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='')
    args = parser.parse_args()

    create_project_structure(working_dir=args.working_dir)
