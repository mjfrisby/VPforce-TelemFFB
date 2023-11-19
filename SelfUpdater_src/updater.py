import json
import logging
import shutil
import ssl
import tempfile
import urllib

import requests
import os
from io import BytesIO
from zipfile import ZipFile
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QDesktopServices, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QProgressBar, QPushButton, QLabel, QWidget, QHBoxLayout, QCheckBox, QMessageBox
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description='TelemFFB Updater App')
parser.add_argument('--url', help='TelemFFB Zip File URL', default=None)
parser.add_argument('--path', help='TelemFFB install directory')
parser.add_argument('--current_version', help='Version of TelemFFB currently installed', default='dummy_text')

args = parser.parse_args()

test = False
g_latest_version = None
g_latest_url = None
g_current_version = args.current_version

if getattr(sys, 'frozen', False):
    g_application_path = os.path.dirname(sys.executable)
elif __file__:
    g_application_path = os.path.dirname(__file__)

g_executable_name = 'VPforce-TelemFFB.exe'

def fetch_latest_version():

    ctx = ssl._create_unverified_context()

    current_version = args.current_version
    latest_version = None
    latest_url = None
    url = "https://vpforcecontrols.com/downloads/TelemFFB/"
    file = "latest.json"
    send_url = url + file

    try:
        with urllib.request.urlopen(send_url, context=ctx) as req:
            latest = json.loads(req.read().decode())
            latest_version = latest["version"]
            latest_url = url + latest["filename"]
    except:
        logging.exception(f"Error checking latest version status: {url}")

    if current_version != latest_version and latest_version is not None and latest_url is not None:
        logging.debug(f"Current version: {current_version} | Latest version: {latest_version}")
        return latest_version, latest_url
    elif current_version == latest_version:
        return False
    else:
        return None




class Downloader(QThread):
    update_progress = pyqtSignal(int)
    update_status_label = pyqtSignal(str)
    download_complete = pyqtSignal()
    extract_complete = pyqtSignal()

    def __init__(self, url, destination):
        super().__init__()
        self.url = url
        self.destination = destination

    def download(self):
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        current_size = 0
        self.update_status_label.emit("Downloading Update:")

        zip_path = os.path.join(self.destination, "temp.zip")

        with open(zip_path, "wb") as zip_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    current_size += len(chunk)
                    progress_percentage = int((current_size / total_size) * 100)
                    self.update_progress.emit(progress_percentage)
                    zip_file.write(chunk)

        self.update_progress.emit(100)
        self.update_status_label.emit("Download Completed:")
        self.download_complete.emit()
        self.extract()

    def extract(self):
        # Simulate extraction process
        self.update_status_label.emit("Extracting Update Files:")
        self.sleep(1)

        # Create a temporary directory inside the script's folder
        temp_dir = tempfile.mkdtemp(dir=g_application_path)

        try:
            # Extract the contents of the zip file to the temporary directory
            with ZipFile(os.path.join(self.destination, "temp.zip"), 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Copy and overwrite files from the temporary directory to the root directory
            for item in os.listdir(temp_dir):
                src = os.path.join(temp_dir, item)
                dest = os.path.join(self.destination, item)

                # Rename a specific file during the copy process
                if item == 'old_name.txt':
                    dest = os.path.join(self.destination, 'new_name.txt')

                if os.path.isdir(src):
                    shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dest, follow_symlinks=False)

        except Exception as e:
            print(f"Error during extraction: {e}")
        finally:
            # Clean up: remove the temporary directory
            shutil.rmtree(temp_dir)

        self.update_status_label.emit("Update Completed!")
        os.remove('temp.zip')
        self.extract_complete.emit()

    def sleep(self, seconds):
        # A simple sleep method for simulation
        import time
        time.sleep(seconds)

    def run(self):
        self.download()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VPforce TelemFFB Updater")
        self.setGeometry(100, 100, 400, 250)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Row layout for version labels and image
        version_layout = QHBoxLayout()

        self.version_label = QLabel(f"Current Installed Version: {g_current_version}\n\nLatest Available Version: {g_latest_version}")
        self.version_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        version_layout.addWidget(self.version_label)
        icon_path = os.path.join(g_application_path, "image/vpforceicon.png")
        self.setWindowIcon(QIcon(icon_path))
        # Add QLabel for the image
        image_label = QLabel(self)
        pixmap = QPixmap(icon_path)  # Replace with the path to your image
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        version_layout.addWidget(image_label)

        # Add the row layout to the main layout
        layout.addLayout(version_layout)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready to Download")

        row_layout = QHBoxLayout()
        row_layout.addWidget(self.progress_label)
        row_layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        layout.addLayout(row_layout)
        layout.addStretch(1)

        # Checkboxes
        self.launch_checkbox = QCheckBox("Launch TelemFFB When Complete")
        self.launch_checkbox.setChecked(True)

        self.release_notes_checkbox = QCheckBox("Read Release Notes")
        self.release_notes_checkbox.setChecked(False)

        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.launch_checkbox)
        checkbox_layout.addWidget(self.release_notes_checkbox)
        checkbox_layout.setAlignment(Qt.AlignLeft)

        layout.addLayout(checkbox_layout)
        layout.addStretch(1)


        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.start_download)

        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self.finish_update)
        self.finish_button.setVisible(False)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.finish_button)

        layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def start_download(self):
        global g_latest_version, g_latest_url, g_application_path

        self.downloader = Downloader(args.url, g_application_path)
        self.downloader.update_progress.connect(self.update_progress)
        self.downloader.update_status_label.connect(self.update_status_label)
        self.downloader.download_complete.connect(self.download_completed)
        self.downloader.extract_complete.connect(self.extract_completed)
        self.update_button.setDisabled(True)
        self.exit_button.setDisabled(True)
        self.downloader.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status_label(self, status):
        self.progress_label.setText(status)

    def download_completed(self):
        # Download is complete, make the Finish button visible
        pass

    def extract_completed(self):
        # Download is complete, make the Finish button visible
        self.finish_button.setVisible(True)
        self.finish_button.setDisabled(False)
        self.exit_button.setDisabled(False)
        self.update_button.setVisible(False)

        # Make the checkboxes visible
        self.launch_checkbox.setVisible(True)
        self.release_notes_checkbox.setVisible(True)

        # If "Launch on Exit" is checked, launch cmd.exe (replace with actual executable path)
        if self.launch_checkbox.isChecked():
            # Specify the path to the executable

            full = os.path.join(g_application_path, g_executable_name)
            print(f"FULLL:{full}")
            # Use subprocess.Popen with the script's directory as the working directory
            subprocess.Popen([full], cwd=g_application_path, shell=False)
            if self.release_notes_checkbox.isChecked():
                rn_path = os.path.join(g_application_path, '_RELEASE_NOTES.txt')
                subprocess.Popen(['start', 'notepad.exe', rn_path], cwd=g_application_path, shell=True)

            sys.exit()

    def finish_update(self):
        # Show a message box if "Read Release Notes" is checked
        if self.launch_checkbox.isChecked():
            # Specify the path to the executable

            full = os.path.join(g_application_path, g_executable_name)
            print(f"FULLL:{full}")
            # Use subprocess.Popen with the script's directory as the working directory
            subprocess.Popen([full], cwd=g_application_path, shell=False)

        if self.release_notes_checkbox.isChecked():
            rn_path = os.path.join(g_application_path, '_RELEASE_NOTES.txt')
            subprocess.Popen(['start', 'notepad.exe', rn_path], cwd=g_application_path, shell=True)

        # Close the updater app
        self.close()

def check_runtime():
    if args.current_version == 'dummy_text' and getattr(sys, 'frozen', False):
        QMessageBox.critical(None, "ERROR", "This application is not intended to be run in a standalone fashion.\n\nIf an update is available, you will be prompted to update upon starting TelemFFB")
        sys.exit()


def main():
    global g_latest_version
    global g_latest_url
    g_latest_version, g_latest_url = fetch_latest_version()

    if args.url is None:
        args.url = g_latest_url

    app = QApplication([])
    window = App()
    window.show()
    print(g_latest_version, g_latest_url)
    check_runtime()
    app.exec_()


if __name__ == "__main__":
    # args.url = "https://vpforcecontrols.com/downloads/TelemFFB/VPforce-TelemFFB-wip-79e581e8.zip"
    args.path = r"C:\Users\micah\Desktop\VPforce-TelemFFB\test"
    main()
