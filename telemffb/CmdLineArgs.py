#
# This file is part of the TelemFFB distribution (https://github.com/walmis/TelemFFB).
# Copyright (c) 2023 Valmantas Palikša.
# Copyright (c) 2023 Micah Frisby
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#


from typing import Optional, List
import argparse

class CmdLineArgs:
    def __init__(
        self,
        teleplot: Optional[str] = None,
        plot: Optional[List[str]] = None,
        device: Optional[str] = None,
        reset: Optional[bool] = False,
        configfile: str = 'config.ini',
        overridefile: Optional[str] = 'None',
        sim: str = 'None',
        type: Optional[str] = None,
        headless: Optional[bool] = False,
        child: Optional[bool] = False,
        masterport: Optional[str] = None,
        minimize: Optional[bool] = False
    ) -> None:
        self.teleplot = teleplot
        self.plot = plot
        self.device = device
        self.reset = reset
        self.configfile = configfile
        self.overridefile = overridefile
        self.sim = sim
        self.type = type
        self.headless = headless
        self.child = child
        self.masterport = masterport
        self.minimize = minimize

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser(description='Send telemetry data over USB')

        # Add destination telemetry address argument
        parser.add_argument('--teleplot', type=str, metavar="IP:PORT", default=None,
                            help='Destination IP:port address for teleplot.fr telemetry plotting service')

        parser.add_argument('-p', '--plot', type=str, nargs='+',
                            help='Telemetry item names to send to teleplot, separated by spaces')

        parser.add_argument('-D', '--device', type=str, help='Rhino device USB VID:PID', default=None)
        parser.add_argument('-r', '--reset', help='Reset all FFB effects', action='store_true')

        # Add config file argument, default config.ini
        parser.add_argument('-c', '--configfile', type=str, help='Config ini file (default config.ini)', default='config.ini')
        parser.add_argument('-o', '--overridefile', type=str, help='User config override file (default = config.user.ini', default='None')
        parser.add_argument('-s', '--sim', type=str, help='Set simulator options DCS|MSFS|IL2 (default DCS', default="None")
        parser.add_argument('-t', '--type', help='FFB Device Type | joystick (default) | pedals | collective', default=None)
        parser.add_argument('--headless', action='store_true', help='Run in headless mode')
        parser.add_argument('--child', action='store_true', help='Is a child instance')
        parser.add_argument('--masterport', type=str, help='master instance IPC port', default=None)

        parser.add_argument('--minimize', action='store_true', help='Minimize on startup')

        args = parser.parse_args()

        return cls(**vars(args))

