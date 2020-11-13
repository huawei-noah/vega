# -*- coding: utf-8 -*-
"""Deploy local cluster."""
import yaml
import subprocess
import os
import tempfile
import platform as pf


def copy_to_slaves(slaves, sub_path, user_name):
    """Copy files to slaves."""
    for slave in slaves:
        try:
            if user_name is None:
                subprocess.call(["scp", "-r", sub_path, "{}:{}/".format(slave, tempfile.gettempdir())])
            else:
                subprocess.call(["scp", "-r", sub_path, "{}@{}:{}/".format(user_name, slave, tempfile.gettempdir())])
        except Exception as ex:
            raise ValueError("Copying file to slave node failed! Maybe slave_ip ({}) is wrong!".format(slave))


def load_config():
    """Load deploy.yml."""
    with open("./deploy.yml", "r") as f:
        deploy = yaml.load(f, Loader=yaml.FullLoader)
    try:
        if "master" in deploy.keys():
            print("Master_ip is ({})".format(deploy["master"]))
        else:
            raise ValueError("Master_ip is not assignation")
        if "slaves" in deploy.keys():
            print("slaves_ip is ({})".format(deploy["slaves"]))
        else:
            raise ValueError("Slaves_ip is not assignation")
        if "listen_port" in deploy.keys():
            print("Listen_port is ({})".format(deploy["listen_port"]))
        else:
            deploy["listen_port"] = "8786"
        if "user_name" in deploy.keys():
            print("User_name is ({})".format(deploy["user_name"]))
        else:
            deploy["user_name"] = None
    except Exception:
        raise ValueError("Format of yaml file is wrong! Keywods is 'master' or 'slaves'")
    return deploy


def init_master(master_ip, master_port):
    """Initialize master node and start dask-scheduler."""
    try:
        subprocess.Popen(["dask-scheduler", "--port", str(master_port)], close_fds=True)
        print("Master scheduler is running.")
    except Exception:
        raise ValueError("Master scheduler failed to start!")


def start_dask_worker(master_ip, slaves_ip, master_port, user_name):
    """Initialize slaves node."""
    try:
        subprocess.Popen(["dask-worker", "{}:{}".format(master_ip, master_port)], close_fds=True)
        for slave_ip in slaves_ip:
            if user_name is None:
                dst = "{}".format(slave_ip)
            else:
                dst = "{}@{}".format(user_name, slave_ip)
            start_file = os.path.join(tempfile.gettempdir(), "deploy/start_slave_worker.py")
            subprocess.Popen(
                ["ssh", dst, "python3", start_file, str(master_ip), str(master_port)],
                close_fds=True)
    except Exception:
        raise ValueError("Worker ({}) failed to start!".format(slave_ip))


def nfs_stat():
    """Check nfs stat."""
    stat = os.popen("nfsstat -s")
    info = stat.readlines()
    print("nfs info is {}".format(info))
    for inf in info:
        if "rpc" in inf:
            print("nfs is installed")
            return True
        else:
            return False


def compare_version(user_version, default_version):
    """Check version of package."""
    uv = user_version.split(".")
    dv = default_version.split(".")
    uv.extend(abs(max(len(uv), len(dv)) - len(uv)) * [0])
    dv.extend(abs(max(len(uv), len(dv)) - len(dv)) * [0])
    for i in range(max(len(uv), len(dv))):
        if uv[i] < dv[i]:
            return False
        elif uv[i] > dv[i]:
            return True
        else:
            continue


def env_check():
    """Check environment."""
    default_sys_version = "16.04"
    default_python_version = "3.6.5"
    system = pf.system()
    sys_version = pf.dist()[1]
    python_version = pf.python_version()
    if system != "Linux":
        raise ValueError("System ({}) is not Linux".format(pf.system()))
    if not compare_version(sys_version, default_sys_version):
        raise ValueError("System ({}) is lower than {}".format(sys_version, default_sys_version))
    if not compare_version(python_version, default_python_version):
        raise ValueError("Python version ({}) is lower than {}".format(sys_version, default_python_version))


def network_check(slaves_ip):
    """Check network."""
    for slave_ip in slaves_ip:
        result = subprocess.call(["ping", "-c", "4", "{}".format(slave_ip)])
        if result != 0:
            raise ValueError("Slave ({}) is not connetion".format(slave_ip))


def package_check(master_ip, node_ip):
    """Check the package and install if not."""
    if isinstance(node_ip, list):
        for node in node_ip:
            cmd = "ssh {} -t '/bin/bash {}/deploy/install_dependencies.sh'".format(node, tempfile.gettempdir())
            result = subprocess.call(cmd, shell=True)
            if result != 0:
                raise ValueError("Node ({}) has no package and can't install".format(node))
    # master
    info = subprocess.call("/bin/bash ./install_dependencies.sh", shell=True)
    if info != 0:
        raise ValueError("Master ({}) can't install vega".format(master_ip))


def precheck():
    """Check everything."""
    env_check()
    conf = load_config()
    network_check(conf["slaves"])
    nfs_stat()
    return conf


def deploy(deploy):
    """Deploy cluster."""
    sub_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    copy_to_slaves(deploy["slaves"], sub_path, deploy["user_name"])
    # fix torch and psutil version conflict.
    package_check(deploy["master"], deploy["slaves"])
    init_master(deploy["master"], deploy["listen_port"])
    start_dask_worker(deploy["master"], deploy["slaves"], deploy["listen_port"], deploy["user_name"])


def verify(conf):
    """Verify cluster."""
    master_ip = conf["master"]
    master_port = conf["listen_port"]
    slaves_ip = conf["slaves"]
    user_name = conf["user_name"]
    try:
        from dask.distributed import Client
        Client("{}:{}".format(master_ip, master_port))
        for slave_ip in slaves_ip:
            if user_name is None:
                dst = "{}".format(slave_ip)
            else:
                dst = "{}@{}".format(master_ip, slave_ip)
            verify_file = "{}/deploy/verify_local_cluster.py".format(tempfile.gettempdir())
            result = subprocess.call(
                ["ssh", dst, "python3", verify_file, "{}:{}".format(master_ip, master_port)])
            if result != 0:
                raise ValueError("Slave ({}) verification failed!".format(slave_ip))
    except Exception:
        raise ValueError("Initializing cluster failed")


if __name__ == "__main__":
    conf = precheck()
    deploy(conf)
    verify(conf)
