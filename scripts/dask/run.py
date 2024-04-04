from functools import partial
import time
import torch
from torchvision import datasets, transforms
from dask_jobqueue import SLURMCluster
from distributed import Client, get_worker, print, as_completed
from dask import delayed
import dask
import os
import socket

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = datasets.MNIST(root='/home/aswerdlo/tmp/debug/test', train=False, transform=transforms.ToTensor(), download=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


def get_client(num_workers):
    cluster = SLURMCluster(
        cores=8,
        job_cpu=8,
        processes=1,
        memory="16GB", 
        walltime="02:00:00", 
        queue="kate_reserved",
        job_extra_directives=["--gres=gpu:1", "--exclude=matrix-0-18,matrix-0-22"], # matrix-0-36,matrix-3-26,matrix-3-28,matrix-1-22
        nanny=False,
        log_directory='/home/aswerdlo/repos/gen/outputs/dask',
        # interface='ib1'
        # env_extra=['module load anaconda', 'source activate mbircone'],
        # job_extra=[system_specific_args],
        # local_directory=local_directory,
    )
    cluster.scale(num_workers)
    print(cluster.job_script())

    # from dask.distributed import LocalCluster
    # cluster = LocalCluster(n_workers=2,  processes=True, threads_per_worker=1)
    
    client = Client(cluster)
    return client

def train(dataset, indices):
    assert os.getenv("IS_SSD") == "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    model = model.to(device)
    
    print(len(indices))
    print(type(indices))

    dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    job_id = os.getenv('SLURM_JOB_ID')

    addr = get_worker().address if hasattr(get_worker(), 'address') else None
    info_str = f"{os.getpid()} {socket.gethostname()} {device} {job_id} {addr}"
    print(f"Starting on {info_str}")

    predictions = []
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            # get_worker().log_event("runtimes", {"i": i})
            input = inputs[0].to(device)
            input = input.view(-1, 784)
            outputs = model(input)
            predictions.append(outputs)

    print(f"Finished on {info_str}")
    pred = torch.cat(predictions).cpu().numpy()
    return pred

if __name__ == '__main__':
    num_workers = 2
    client = get_client(num_workers=num_workers)
    try:
        client.wait_for_workers(num_workers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = CustomDataset()
        submission_list = range(len(dataset))

        dataset = client.scatter(dataset, broadcast=True)
        data = client.scatter(submission_list, broadcast=False)

        chunk_size = len(submission_list) // num_workers  # Adjust this based on the number of workers
        chunks = [submission_list[i:i + chunk_size] for i in range(0, len(submission_list), chunk_size)]
        data_chunks = client.scatter(chunks, broadcast=False)

        futures = [client.submit(train, dataset, chunk) for chunk in data_chunks]
        results = client.gather(futures)

        for result in results:
            for row in result:
                print(row)

        breakpoint()
    except:
        client.shutdown()