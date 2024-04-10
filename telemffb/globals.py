from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PyQt5.QtCore import QSettings
    from .LogWindow import LogWindow
    from .IPCNetworkThread import IPCNetworkThread
    from .utils import SystemSettings
    from .settingsmanager import SettingsWindow
    from .telem.TelemManager import TelemManager
    from main import MainWindow

settings_mgr : 'SettingsWindow' = None
userconfig_rootpath = None
userconfig_path = None
defaults_path = None
defaults_path = None

# main window instance
main_window : 'MainWindow' = None

device_type : str = None
device_usbpid : str = None
device_usbvidpid : str = None # "FFFF:2055"

launched_instances = []
child_ipc_ports = []
master_instance = False
ipc_instance : 'IPCNetworkThread' = None
child_instance = None

# systems settings
system_settings : 'SystemSettings' = None

#parsed startup arguments
args = None

# telemetry manager instance
telem_manager : 'TelemManager' = None

# configurator gains read at startup
startup_configurator_gains = None

# function reference to stop/init sims
stop_sims = None
init_sims = None

log_window : 'LogWindow' = None