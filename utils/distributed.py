import os
import time
import torch
import pickle
import subprocess

import torch.distributed as dist


def apply_distributed(opt):
    if opt['rank'] == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_address = result.decode('utf-8').split()[0]
        master_port = opt['PORT']
    else:
        master_address = None
        master_port = None

    if torch.distributed.is_available() and opt['world_size'] > 1:
        from mpi4py import MPI
        master_address = MPI.COMM_WORLD.bcast(master_address, root=0)
        master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
        init_method_url = 'tcp://{}:{}'.format(master_address, master_port)
        backend = 'nccl'
        world_size = opt['world_size']
        rank = opt['rank']
        torch.distributed.init_process_group(backend=backend,
                                            init_method=init_method_url,
                                            world_size=world_size,
                                            rank=rank)
        
def apply_distributed_slurm(opt):
    ### UNVERIFIED FUNCTION FROM GPT4

    # Determine if the process is the master (rank 0) process
    if opt['rank'] == 0:
        # Get the IP address of the master node
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_address = result.decode('utf-8').split()[0]
        master_port = opt['PORT']
    else:
        # Non-master processes get the master address and port from environment variables
        master_address = os.getenv('MASTER_ADDR')
        master_port = os.getenv('MASTER_PORT')
        
        if master_address is None or master_port is None:
            raise RuntimeError("Master address and port must be set in environment variables for non-master processes.")

    # Broadcast the master address and port to all processes using SLURM variables
    if torch.distributed.is_available() and opt['world_size'] > 1:
        if opt['rank'] == 0:
            os.environ['MASTER_ADDR'] = master_address
            os.environ['MASTER_PORT'] = str(master_port)
        
        # Initialize the process group for distributed training
        init_method_url = f'tcp://{master_address}:{master_port}'
        backend = 'nccl'  # or 'gloo' if using CPU
        dist.init_process_group(backend=backend,
                                init_method=init_method_url,
                                world_size=opt['world_size'],
                                rank=opt['rank'])


def init_distributed(opt):
    opt['CUDA'] = opt.get('CUDA', True) and torch.cuda.is_available()
    if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
        # application was started without MPI
        # default to single node with single process
        opt['env_info'] = 'no MPI'
        opt['world_size'] = 1
        opt['local_size'] = 1
        opt['rank'] = 0
        opt['local_rank'] = 0
        opt['master_address'] = '127.0.0.1'
        opt['master_port'] = '8673'
    else:
        # application was started with MPI
        # get MPI parameters
        opt['world_size'] = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        opt['local_size'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        opt['rank'] = int(os.environ['OMPI_COMM_WORLD_RANK'])
        opt['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    # set up device
    if not opt['CUDA']:
        assert opt['world_size'] == 1, 'multi-GPU training without CUDA is not supported since we use NCCL as communication backend'
        opt['device'] = torch.device("cpu")
    else:
        torch.cuda.set_device(opt['local_rank'])
        opt['device'] = torch.device("cuda", opt['local_rank'])

    apply_distributed(opt)
    return opt

def init_distributed_slurm(opt):
    ### UNVERIFIED FUNCTION FROM GPT4

    # Check if CUDA is available and set it in the options
    opt['CUDA'] = opt.get('CUDA', True) and torch.cuda.is_available()

    if 'SLURM_JOB_ID' not in os.environ:
        # Application was started without SLURM (single node, single process)
        opt['env_info'] = 'no SLURM'
        opt['world_size'] = 1
        opt['local_size'] = 1
        opt['rank'] = 0
        opt['local_rank'] = 0
        opt['master_address'] = '127.0.0.1'
        opt['master_port'] = '8673'
    else:
        # Application was started with SLURM, get SLURM parameters
        opt['world_size'] = int(os.environ['SLURM_NTASKS'])
        opt['local_size'] = int(os.environ['SLURM_NTASKS_PER_NODE'])
        opt['rank'] = int(os.environ['SLURM_PROCID'])
        opt['local_rank'] = opt['rank'] % opt['local_size']
        opt['master_address'] = os.environ['SLURM_NODELIST'].split()[0]  # master node
        opt['master_port'] = '8673'  # You can use a fixed port or set it as an environment variable

    # Set up the device
    if not opt['CUDA']:
        assert opt['world_size'] == 1, 'multi-GPU training without CUDA is not supported since we use NCCL as the communication backend'
        opt['device'] = torch.device("cpu")
    else:
        torch.cuda.set_device(opt['local_rank'])
        opt['device'] = torch.device("cuda", opt['local_rank'])

    # Call the apply_distributed function to set up distributed training
    apply_distributed_slurm(opt)

    return opt

def is_main_process():
    rank = 0
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    return rank == 0

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device="cuda")
        else:
            tensor = torch.tensor(1, device="cuda")
        dist.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)

    _send_and_wait(0)
    # now sync on the main process
    _send_and_wait(1)