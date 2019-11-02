import os
import src.utils

install_path = "/cr/users/koepke/install"

def get_install_path():
    return install_path

def set_install_path(path):
    global install_path
    install_path = os.path.expanduser(path)

