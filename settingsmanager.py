import logging
import xml.etree.ElementTree as ET
import sys
import os
import shutil
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication,  QTableWidgetItem, QCheckBox, QLineEdit, QDialog, QLabel, QComboBox,
                             QVBoxLayout, QPushButton, QFileDialog)
from PyQt5.QtWidgets import QTableWidget, QTextEdit, QWidget, QSlider
from datetime import datetime
from PyQt5.QtCore import Qt, pyqtSignal
from settingswindow import Ui_SettingsWindow
import re
import xml.dom.minidom

print_debugs = True


def lprint(msg):
    if print_debugs:
        print(msg)

class SettingsWindow(QtWidgets.QMainWindow, Ui_SettingsWindow):
    defaults_path = 'defaults.xml'

    userconfig_rootpath = os.path.join(os.environ['LOCALAPPDATA'],"VPForce-TelemFFB")
    userconfig_path = os.path.join(userconfig_rootpath , 'userconfig.xml')

    input_sim = ""
    input_model_name = ""
    input_model_type = ""

    sim = ""                             # DCS, MSFS, IL2       -- set in get_current_model below
    model_name = "unknown airplane"    # full model name with livery etc
    crafttype = ""                       # suggested, send whatever simconnect finds

    data_list = []
    prereq_list = []

    model_type = ""     # holder for current type/class
    model_pattern = ""  # holder for current matching pattern found in config xmls
    edit_mode = '' # holder for current editing mode.

    allow_in_table_editing = False

    def __init__(self, datasource='Global', device = 'joystick'):
        super(SettingsWindow, self).__init__()
        self.setupUi(self)  # This sets up the UI from Ui_SettingsWindow
        # self.defaults_path = defaults_path
        # self.userconfig_path = userconfig_path
        self.device = device
        self.input_sim = datasource
        self.sim = self.input_sim
        self.b_browse.clicked.connect(self.choose_directory)
        self.b_update.clicked.connect(self.update_button)
        self.slider_float.valueChanged.connect(self.update_textbox)
        self.cb_enable.stateChanged.connect(self.cb_enable_setvalue)
        self.drp_valuebox.currentIndexChanged.connect(self.update_dropbox)
        self.buttonBox.rejected.connect(self.hide)
        self.clear_propmgr()
        self.backup_userconfig()
        self.init_ui()

    def get_current_model(self,the_sim=None, dbg_model_name=None, dbg_crafttype=None, ):

        # in the future, get from simconnect.
        if the_sim is not None:

            self.sim = the_sim
        else:
            self.sim = self.input_sim
        if dbg_model_name is not None:
            self.model_name = dbg_model_name     #type value in box for testing. will set textbox in future
        else:
            self.model_name = self.input_model_name
        if dbg_crafttype is not None:
            self.crafttype = dbg_crafttype  # suggested, send whatever simconnect finds
        else:
            self.crafttype = self.input_model_type

        lprint(f'get current model {self.sim}  {self.model_name} {self.crafttype}')

        self.tb_currentmodel.setText(self.model_name)
        self.table_widget.clear()
        self.setup_table()
        self.setup_class_list()
        self.setup_model_list()

        # output a single model
        self.model_type, self.model_pattern, self.data_list = self.read_single_model()
        self.drp_sim.blockSignals(True)
        self.drp_sim.setCurrentText(self.sim)
        self.drp_sim.blockSignals(False)
        self.drp_class.blockSignals(True)
        self.drp_class.setCurrentText(self.model_type)
        self.drp_class.blockSignals(False)

        if self.model_pattern != '':
            #self.rb_model.setChecked(True)
            self.set_edit_mode('Model')
            self.drp_models.blockSignals(True)
            self.drp_models.setCurrentText(self.model_pattern)
            self.drp_models.blockSignals(False)
        else:
            if self.model_type == '':
                self.set_edit_mode(self.sim)
            else:
                self.set_edit_mode('Class')


        lprint(f"\nCurrent: {self.sim}  model: {self.model_name}  pattern: {self.model_pattern}  class: {self.model_type}  device:{self.device}\n")

        # put model name and class into UI

        if self.model_type != '': self.drp_class.setCurrentText(self.model_type)

        self.populate_table()

    def init_ui(self):

        self.create_empty_userxml_file()

        self.tb_currentmodel.setText(self.model_name)

        self.get_current_model()
        self.b_getcurrentmodel.clicked.connect(self.currentmodel_click)

        print (f"init {self.sim}")
        # Your custom logic for table setup goes here
        self.setup_table()

        # Connect the stateChanged signal of the checkbox to the toggle_rows function
        self.cb_show_inherited.stateChanged.connect(self.toggle_rows)

        self.b_revert.clicked.connect(self.restore_userconfig_backup)

        self.l_device.setText(self.device)
        self.set_edit_mode(self.sim)
        # read models from xml files to populate dropdown
        self.setup_model_list()
        self.setup_class_list()

        # allow changing sim dropdown
        self.drp_sim.currentIndexChanged.connect(lambda index: self.update_table_on_sim_change())
        # change class dropdown
        self.drp_class.currentIndexChanged.connect(lambda index: self.update_table_on_class_change())
        #allow changing model dropdown
        self.drp_models.currentIndexChanged.connect(lambda index: self.update_table_on_model_change())

        # create model setting button
        self.b_createusermodel.clicked.connect(self.show_user_model_dialog)

        # Initial visibility of rows based on checkbox state
        self.toggle_rows()

    def currentmodel_click(self):
        self.get_current_model(self.input_sim)

    def update_current_aircraft(self, data_source, aircraft_name, ac_class=None):
        send_source = data_source
        if data_source == "MSFS2020":
            send_source = "MSFS"
        if send_source is not None:
            if '.' in send_source:        #  source came is as SIM.Aircrafttype, must split.
                input = send_source.split('.')
                self.sim = input[0]
                self.input_model_type = input[1]
            else:
                self.input_sim = send_source
                if ac_class is not None:
                    self.input_model_type = ac_class
                else:
                    self.input_model_type = ''

        self.input_model_name = aircraft_name


        if self.isVisible():
            self.b_getcurrentmodel.click()
        #    self.currentmodel_click()
        #    self.get_current_model()

    def backup_userconfig(self):
        # Ensure the userconfig.xml file exists
        self.create_empty_userxml_file()
        backup_path = self.userconfig_path + ".backup"
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path_time = f"{self.userconfig_path}_{timestamp}.backup"
        try:
            # Copy the userconfig.xml file to the backup location
            shutil.copy2(self.userconfig_path, backup_path)
            #shutil.copy2(self.userconfig_path, backup_path_time)        #  do we want lots of backups?
            lprint(f"Backup created: {backup_path}")
        except Exception as e:
            lprint(f"Error creating backup: {e}")

    def restore_userconfig_backup(self):
        # Ensure the backup file exists
        backup_path = self.userconfig_path + ".backup"

        if not os.path.isfile(backup_path):
            lprint(f"Backup file '{backup_path}' not found.")
            return

        try:
            # Copy the backup file to userconfig.xml
            shutil.copy2(backup_path, self.userconfig_path)
            lprint(f"Backup '{backup_path}' restored to userconfig.xml")
            self.get_current_model()

        except Exception as e:
            lprint(f"Error restoring backup: {e}")

    def show_user_model_dialog(self):
        current_aircraft = self.tb_currentmodel.text()
        dialog = UserModelDialog(self.sim,current_aircraft, self.model_type, self)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            # Handle accepted
            new_aircraft = dialog.tb_current_aircraft.text()
            new_combo_box_value = dialog.combo_box.currentText()
            print (f"New: {new_aircraft} {new_combo_box_value}")
            self.write_models_to_xml(new_aircraft,new_combo_box_value, 'type')
            self.model_name = new_aircraft
            self.tb_currentmodel.setText(new_aircraft)
            self.get_current_model(self.sim)
        else:
            # Handle canceled
            pass


    def setup_class_list(self):
        self.drp_class.blockSignals(True)
        print_debug = False

        match self.sim:
            case 'Global':
                # Assuming drp_class is your QComboBox
                for disable in {'PropellerAircraft', 'TurbopropAircraft', 'JetAircraft', 'GliderAircraft', 'Helicopter', 'HPGHelicopter'}:
                    if print_debug: print (f"disable {disable}")
                    self.drp_class.model().item(self.drp_class.findText(disable)).setEnabled(False)

            case 'DCS':
                for disable in { 'TurbopropAircraft', 'GliderAircraft', 'HPGHelicopter'}:
                    if print_debug: lprint(f"disable {disable}")
                    self.drp_class.model().item(self.drp_class.findText(disable)).setEnabled(False)
                for enable in {'PropellerAircraft', 'JetAircraft', 'Helicopter'}:
                    if print_debug: lprint(f"enable {enable}")
                    self.drp_class.model().item(self.drp_class.findText(enable)).setEnabled(True)

            case 'IL2':
                for disable in {'TurbopropAircraft', 'GliderAircraft', 'Helicopter', 'HPGHelicopter'}:
                    if print_debug: lprint(f"disable {disable}")
                    self.drp_class.model().item(self.drp_class.findText(disable)).setEnabled(False)
                for enable in {'PropellerAircraft', 'JetAircraft'}:
                    if print_debug: lprint(f"enable {enable}")
                    self.drp_class.model().item(self.drp_class.findText(enable)).setEnabled(True)

            case 'MSFS':
                for enable in {'PropellerAircraft', 'TurbopropAircraft', 'JetAircraft', 'GliderAircraft', 'Helicopter', 'HPGHelicopter'}:
                    if print_debug: lprint(f"enable {enable}")
                    self.drp_class.model().item(self.drp_class.findText(enable)).setEnabled(True)

        #self.drp_class.addItems(classes)
        self.drp_class.blockSignals(False)

    def setup_model_list(self):
        models = self.read_models(self.sim)
        self.drp_models.blockSignals(True)
        self.drp_models.clear()
        self.drp_models.addItems(models)
        self.drp_models.setCurrentText(self.model_pattern)
        self.drp_models.blockSignals(False)

    def setup_table(self):
        self.table_widget.setColumnCount(10)
        headers = ['Source', 'Grouping', 'Display Name', 'Value', 'Info', "name"]
        self.table_widget.setHorizontalHeaderLabels(headers)
        self.table_widget.setColumnWidth(0, 120)
        self.table_widget.setColumnWidth(1, 120)
        self.table_widget.setColumnWidth(2, 215)
        self.table_widget.setColumnWidth(3, 120)
        self.table_widget.setColumnHidden(4, True)
        self.table_widget.setColumnHidden(5, True)
        self.table_widget.setColumnHidden(6, True)
        self.table_widget.setColumnHidden(7, True)
        self.table_widget.setColumnHidden(8, True)
        self.table_widget.setColumnHidden(9, True)

    def populate_table(self):
        self.table_widget.blockSignals(True)
        sorted_data = sorted(self.data_list, key=lambda x: (x['grouping'] != 'Basic', x['grouping'], x['displayname']))
        list_length = len(self.data_list)
        pcount = 1
        for row, data_dict in enumerate(sorted_data):
            #
            # hide 'type' setting in class mode
            #
            if self.edit_mode == 'Class' and data_dict['name'] == 'type':
                self.table_widget.setRowHeight(row, 0)
                continue


            # hide prereqs not satisfied
            #
            found_prereq = False
            if data_dict['prereq'] != '':
                for pr in self.prereq_list:
                    if pr['prereq']==data_dict['prereq']:
                        if pr['value'].lower() == 'false':
                            lprint(f"name: {data_dict['displayname']} data: {data_dict['prereq']}  pr:{pr['prereq']}  value:{pr['value']}")
                            self.table_widget.setRowHeight(row, 0)
                            found_prereq = True
                            break
                if found_prereq: continue

            state = self.set_override_state(data_dict['replaced'])
            checkbox = QCheckBox()
            # Manually set the initial state
            checkbox.setChecked(state)


            checkbox.clicked.connect(
                lambda state, trow=row, tdata_dict=data_dict: self.override_state_changed(trow, tdata_dict, state))

            item = QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            #item.setStyleSheet("margin-left:50%; margin-right:50%;")
            item.setData(QtCore.Qt.UserRole, row)  # Attach row to the item
            item.setData(QtCore.Qt.CheckStateRole, QtCore.Qt.Unchecked)  # Set initial state

            grouping_item = QTableWidgetItem(data_dict['grouping'])
            displayname_item = QTableWidgetItem(data_dict['displayname'])
            value_item = self.create_datatype_item(data_dict['datatype'], data_dict['value'], data_dict['unit'], checkbox.checkState())
            info_item = QTableWidgetItem(data_dict['info'])
            replaced_item = QTableWidgetItem("      " + data_dict['replaced'])
            unit_item = QTableWidgetItem(data_dict['unit'])
            valid_item = QTableWidgetItem(data_dict['validvalues'])
            datatype_item = QTableWidgetItem(data_dict['datatype'])

            # store name for use later, not shown
            name_item = QTableWidgetItem(data_dict['name'])
            name_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            state_item = QTableWidgetItem(str(state))


            # Connect the itemChanged signal to your custom function
            value_item.setData(Qt.UserRole, row)  # Attach row to the item
            value_item.setData(Qt.UserRole + 1, data_dict['name'])  # Attach name to the item
            value_item.setData(Qt.UserRole + 2, data_dict['value'])  # Attach original value to the item
            value_item.setData(Qt.UserRole + 3, data_dict['unit'])  # Attach unit to the item
            value_item.setData(Qt.UserRole + 4, data_dict['datatype'])  # Attach datatype to the item
            value_item.setData(Qt.UserRole + 5, data_dict['validvalues'])  # Attach datatype to the item
            value_item.setData(Qt.UserRole + 6, str(state))  # Attach datatype to the item


            #lprint(f"Row {row} - Grouping: {data_dict['grouping']}, Display Name: {data_dict['displayname']}, Unit: {data_dict['unit']}, Ovr: {data_dict['replaced']}")

            # Check if replaced is an empty string and set text color accordingly
            for item in [grouping_item, displayname_item, value_item, info_item, replaced_item]:
                match data_dict['replaced']:
                    case 'Global':
                        item.setForeground(QtGui.QColor('gray'))
                    case 'Global (user)':
                        item.setForeground(QtGui.QColor('black'))
                    case 'Sim Default':
                        item.setForeground(QtGui.QColor('darkblue'))
                    case 'Sim (user)':
                        item.setForeground(QtGui.QColor('blue'))
                    case 'Class Default':
                        item.setForeground(QtGui.QColor('darkGreen'))
                    case 'Class (user)':
                        item.setForeground(QtGui.QColor('green'))
                    case 'Model Default':
                        item.setForeground(QtGui.QColor('darkMagenta'))
                    case 'Model (user)':
                        item.setForeground(QtGui.QColor('magenta'))

            # Make specific columns read-only
            grouping_item.setFlags(grouping_item.flags() & ~Qt.ItemIsEditable)
            displayname_item.setFlags(displayname_item.flags() & ~Qt.ItemIsEditable)
            info_item.setFlags(info_item.flags() & ~Qt.ItemIsEditable)
            replaced_item.setFlags(replaced_item.flags() & ~Qt.ItemIsEditable)

            #
            # disable in-table value editing here
            # if not self.allow_in_table_editing:
            #     value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)

            # Set the row count based on the actual data
            self.table_widget.setRowCount(list_length)

            self.table_widget.insertRow(row)
            self.table_widget.setItem(row, 0, item)
            try:
                self.table_widget.setCellWidget(row, 0, checkbox)
            except Exception as e:
                logging.error(f"EXCEPTION: {e}")
            self.table_widget.setItem(row, 1, grouping_item)
            self.table_widget.setItem(row, 2, displayname_item)
            self.table_widget.setItem(row, 3, value_item)
            self.table_widget.setItem(row, 4, info_item)
            self.table_widget.setItem(row, 5, name_item)
            self.table_widget.setItem(row, 6, valid_item)
            self.table_widget.setItem(row, 7, datatype_item)
            self.table_widget.setItem(row, 8, unit_item)
            self.table_widget.setItem(row, 9, state_item)
            #self.connected_rows.add(row)


            # row click for property manager
            #self.table_widget.itemSelectionChanged.connect(self.handle_item_click)
            self.table_widget.setSelectionBehavior(QtWidgets.QTableView.SelectRows)

            # make unselectable in not checked
            if not state:
                for col in range(self.table_widget.columnCount()):
                    unselitem = self.table_widget.item(row, col)
                    unselitem.setFlags(unselitem.flags() & ~Qt.ItemIsSelectable)

            # if row not in self.connected_rows:
            #     value_item.dataChanged.connect(self.handle_item_change)
            #     self.connected_rows.add(row)

        # this is for handling clicking the actual value cell..
        self.table_widget.itemSelectionChanged.connect(self.handle_item_click)

        # disable in-table value editing here
        if not self.allow_in_table_editing:
            self.table_widget.itemChanged.connect(self.handle_item_change)  # Connect to the custom function

        self.table_widget.blockSignals(False)

    def toggle_rows(self):
        show_inherited = self.cb_show_inherited.isChecked()

        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)  # Assuming the column with 'user' is the sixth column

            if item is not None and 'user' in item.text() and self.edit_mode in item.text():
                self.table_widget.setRowHidden(row, False)

            else:
                self.table_widget.setRowHidden(row, not show_inherited)

    def clear_propmgr(self):
        self.l_displayname.setText("Select a Row to Edit")
        self.cb_enable.hide()
        self.t_info.hide()
        self.l_validvalues.hide()
        self.l_value.hide()
        self.slider_float.hide()
        self.b_update.hide()
        self.drp_valuebox.hide()
        self.tb_value.hide()
        self.b_browse.hide()

    def handle_item_click(self):
        selected_items = self.table_widget.selectedItems()

        if selected_items:
            # Get the row number of the first selected item
            row = selected_items[0].row()
            if row is not None:

                #lprint(f"Clicked on Row {row + 1}")

                for col in range(self.table_widget.columnCount()):

                    source_item = self.table_widget.item(row, 0)
                    source = source_item.text()
                    displayname_item = self.table_widget.item(row, 2, )
                    displayname = displayname_item.text()
                    value_item = self.table_widget.item(row, 3 )
                    value = value_item.text()
                    info_item = self.table_widget.item(row, 4 )
                    info = info_item.text()
                    name_item = self.table_widget.item(row, 5 )
                    name = name_item.text()
                    valid_item = self.table_widget.item(row, 6 )
                    validvalues = valid_item.text()
                    datatype_item = self.table_widget.item(row, 7 )
                    datatype = datatype_item.text()
                    unit_item = self.table_widget.item(row, 8 )
                    unit = unit_item.text()
                    state_item = self.table_widget.item(row, 9 )
                    state = state_item.text()

                    self.populate_propmgr(name, displayname, value, unit, validvalues, datatype, info, state)
            else:
                self.clear_propmgr()

    def populate_propmgr(self, name, displayname, value, unit, validvalues, datatype, info, state):
        self.clear_propmgr()
        if state.lower() == 'false':
            return
        self.l_displayname.setText(displayname)
        if info != 'None' and info != '':
            self.t_info.setText(info)
            self.t_info.show()
        self.l_name.setText(name)
        self.tb_value.setText(value)
        self.l_name.show()
        self.b_update.show()
        match datatype:
            case 'bool':
                self.cb_enable.show()
                if value == '': value = 'false'
                self.cb_enable.setCheckState(self.strtobool(value))
                #self.tb_value.show()
                self.tb_value.setText(value)
            case 'float':
                self.slider_float.setMinimum(0)
                self.slider_float.setMaximum(100)
                self.l_value.show()
                self.slider_float.show()
                self.tb_value.show()
                if '%' in value:
                    pctval = int(value.replace('%', ''))
                else:
                    pctval = int(float(value) * 100)
                self.slider_float.setValue(pctval)
                self.tb_value.setText(str(pctval) + '%')

            case 'negfloat':
                self.slider_float.setMinimum(-100)
                self.slider_float.setMaximum(100)
                self.l_value.show()
                self.slider_float.show()
                self.tb_value.show()
                if '%' in value:
                    pctval = int(value.replace('%', ''))
                else:
                    pctval = int(float(value) * 100)
                self.slider_float.setValue(pctval)
                self.tb_value.setText(str(pctval) + '%')

            case 'int' | 'text' | 'anyfloat':
                self.l_value.show()
                self.tb_value.show()

            case 'list':
                self.l_value.show()
                self.tb_value.show()
                self.drp_valuebox.show()
                self.drp_valuebox.blockSignals(True)
                self.drp_valuebox.clear()
                valids = validvalues.split(',')
                self.drp_valuebox.addItems(valids)
                self.drp_valuebox.blockSignals(False)
            case 'path':
                self.b_browse.show()
                self.l_value.show()

    def cb_enable_setvalue(self):
        state = self.cb_enable.checkState()
        strstate = 'false' if state == 0 else 'true'
        self.tb_value.setText(strstate)

    def update_button(self,):
        self.write_values(self.l_name.text(),self.tb_value.text())

    def update_textbox(self, value):
        pct_value = int(value)  # Convert slider value to a float (adjust the division factor as needed)
        self.tb_value.setText(str(pct_value)+'%')

    def update_dropbox(self):
        self.tb_value.setText(self.drp_valuebox.currentText())

    def choose_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog

        # Open the directory browser dialog
        directory = QFileDialog.getExistingDirectory(self, "Choose Directory", options=options)

        if directory:
            lprint(f"Selected Directory: {directory}")
            self.tb_value.setText(directory)

    def handle_item_change(self,  item):

        #print (f"{item.column()} : {self.value_previous} ")
        if item.column() == 3:  # Assuming column 3 contains the 'value' items

            row = item.data(Qt.UserRole)
            name = item.data(Qt.UserRole + 1)
            original_value = item.data(Qt.UserRole + 2)
            unit = item.data(Qt.UserRole + 3)
            datatype = item.data(Qt.UserRole + 4)
            valid = item.data(Qt.UserRole + 5)
            state = item.data(Qt.UserRole + 6)
            new_value = item.text()

            if datatype == 'bool':
                newbool = not self.strtobool(original_value)
                new_value = 'true' if newbool else 'false'

            if original_value == '': original_value = "(blank)"


            if new_value != original_value:
                lprint(f"{item.column()} : CHANGED ")

                self.write_values(name, new_value)

                lprint(
                            f"Row {row} - Name: {name}, Original: {original_value}, New: {new_value}, Unit: {unit}, Datatype: {datatype}, valid values: {valid}")

    def write_values(self, name, new_value):
        mysim = self.drp_sim.currentText()
        myclass = self.drp_class.currentText()
        mymodel = self.drp_models.currentText()
        match self.edit_mode:
            case 'Global' | 'Sim':
                self.write_sim_to_xml(self.device, self.sim, new_value, name)
                self.drp_sim.setCurrentText('')
                self.drp_sim.setCurrentText(mysim)
            case 'Class':
                self.write_class_to_xml(self.device, self.sim, self.model_type, new_value, name)
                self.drp_class.setCurrentText('')
                self.drp_class.setCurrentText(myclass)
            case 'Model':
                self.write_models_to_xml(self.device, self.sim, self.model_pattern, new_value, name)
                self.drp_models.setCurrentText('')
                self.drp_models.setCurrentText(mymodel)
        self.reload_table()

    def strtobool(self,val):
        """Convert a string representation of truth to true (1) or false (0).
        True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
        are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
        'val' is anything else.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError("invalid truth value %r" % (val,))


    # Slot function to handle checkbox state changes
    # blows up 0x0000005 after 3 clicks

    def override_state_changed(self, row, data_dict, state):
        lprint(f"Override - Row: {row}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
        mysim = self.drp_sim.currentText()
        myclass = self.drp_class.currentText()
        mymodel = self.drp_models.currentText()

        self.table_widget.blockSignals(True)
        if state:

            # add row to userconfig
            match self.edit_mode:
                case 'Global' | 'Sim':
                    lprint(f"Override - {self.sim}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.write_sim_to_xml(self.device, self.sim,data_dict['value'],data_dict['name'])
                    self.sort_elements()
                    self.drp_sim.setCurrentText('')
                    self.drp_sim.setCurrentText(mysim)
                case 'Class':
                    lprint(f"Override - {self.sim}.{self.drp_class.currentText()}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.write_class_to_xml(self.device, self.sim, myclass, data_dict['value'],data_dict['name'])
                    self.sort_elements()
                    self.drp_class.setCurrentText('')
                    self.drp_class.setCurrentText(myclass)
                case 'Model':
                    lprint(f"Override - {self.sim}.{self.drp_models.currentText()}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.write_models_to_xml(self.device, self.sim, self.drp_models.currentText(),data_dict['value'],data_dict['name'])
                    self.sort_elements()
                    self.drp_models.setCurrentText('')
                    self.drp_models.setCurrentText(mymodel)
        # make value editable & reset view

            self.reload_table()
            self.table_widget.selectRow(row)
            self.handle_item_click()

        else:
            match self.edit_mode:
                case 'Global':
                    lprint(
                        f"Remove - {self.sim}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.erase_sim_from_xml(data_dict['value'], data_dict['name'])
                    self.drp_sim.setCurrentText('')
                    self.drp_sim.setCurrentText(mysim)
                case 'Sim':
                    lprint(
                        f"Remove - {self.sim}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.erase_sim_from_xml(data_dict['value'], data_dict['name'])
                    self.drp_sim.setCurrentText('')
                    self.drp_sim.setCurrentText(mysim)
                case 'Class':
                    lprint(
                        f"Remove - {self.sim}.{self.drp_class.currentText()}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.erase_class_from_xml(self.drp_class.currentText(), data_dict['value'], data_dict['name'])
                    self.drp_class.setCurrentText('')
                    self.drp_class.setCurrentText(myclass)
                case 'Model':
                    lprint(
                        f"Remove - {self.sim}.{self.drp_models.currentText()}, Name: {data_dict['name']}, value: {data_dict['value']}, State: {state}, Edit: {self.edit_mode}")
                    self.erase_models_from_xml(self.drp_models.currentText(), data_dict['value'], data_dict['name'])
                    self.drp_models.setCurrentText('')
                    self.drp_models.setCurrentText(mymodel)
                # make value editable & reset view
            self.reload_table()
            self.clear_propmgr()
            self.table_widget.blockSignals(False)


    def sort_elements(self):
        # Parse the XML file
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        # Extract all elements
        all_elements = root.findall('')

        # Sort the elements based on their tag names
        sorted_elements = sorted(all_elements, key=lambda x: x.tag)

# warning!  deletes everything

         # Replace existing elements with sorted elements
        # for elem in root:
        #     root.remove(elem)
        #
        #     # Add sorted elements back to the parent
        # for elem in sorted_elements:
        #     root.append(elem)
###

        # Prettify the XML
        xml_str = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml()
        with open(self.userconfig_path, 'w') as xml_file:
            xml_file.write(xml_str)



    def update_table_on_model_change(self):
        # Get the selected model from the combo box
        self.set_edit_mode('Model')

        self.model_name = self.drp_models.currentText()

        if self.model_name != '':

            # Replace the following line with your actual XML reading logic
            self.model_type, self.model_pattern, self.data_list = self.read_single_model()
            lprint(f"\nmodel change for: {self.sim}  model: {self.model_name}  pattern: {self.model_pattern}  class: {self.model_type}  device:{self.device}\n")

            # Update the table with the new data
            self.drp_class.blockSignals(True)
            self.drp_class.setCurrentText(self.model_type)
            self.drp_class.blockSignals(False)

        else:
            lprint("model cleared")
            self.set_edit_mode('Class')
            old_model_type = self.model_type
            lprint(self.model_type)
            self.drp_class.setCurrentText('')
            self.drp_class.setCurrentText(old_model_type)

        self.reload_table()

    def update_table_on_class_change(self):
        # Get the selected model from the combo box
        self.drp_models.blockSignals(True)
        self.drp_models.setCurrentText('')
        self.model_name = ''
        self.drp_models.blockSignals(False)
        self.set_edit_mode('Class')
        self.model_type = self.drp_class.currentText()
        if self.model_type != '':

            self.reload_table()

            lprint(
                f"\nclass change for: {self.sim}  model: ---  pattern: {self.model_pattern}  class: {self.model_type}  device:{self.device}\n")

        else:
            lprint("class cleared")
            self.drp_class.setCurrentText('')
            self.set_edit_mode('Sim')
            old_sim = self.sim
            lprint(self.model_type)
            self.drp_sim.setCurrentText('Global')
            self.drp_sim.setCurrentText(old_sim)

        self.table_widget.clear()
        self.setup_table()
        self.populate_table()
        self.toggle_rows()

    def update_table_on_sim_change(self):
        # Get the selected sim from the radio buttons

        self.sim = self.drp_sim.currentText()

        if self.sim == 'Global':
            self.set_edit_mode('Global')
        else:
            self.set_edit_mode('Sim')

        self.setup_class_list()
        self.setup_model_list()

        self.reload_table()

        lprint(f"\nsim change for: {self.sim}  model: {self.model_name}  pattern: {self.model_pattern}  class: {self.model_type}  device:{self.device}\n")

    def reload_table(self):
        self.table_widget.blockSignals(True)
        # Read all the data
        self.model_type, self.model_pattern, self.data_list = self.read_single_model()
        # Update the table with the new data
        self.table_widget.clear()
        self.setup_table()
        self.populate_table()
        self.toggle_rows()
        self.table_widget.blockSignals(False)

    def create_datatype_item(self, datatype, value, unit, checkstate):
        #lprint(f"{datatype} {value}")
        if datatype == 'bool':
            toggle = QCheckBox()
            toggle.setChecked(value.lower() == 'true')
            #checkbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            toggle.setStyleSheet("margin-left:50%; margin-right:50%;")
            item = QTableWidgetItem()
            #item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            boolvalue = self.strtobool(value)
            item.setData(Qt.CheckStateRole, Qt.Checked if boolvalue else Qt.Unchecked)
            if not checkstate:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)   # no editing if not allowed in this mode
            # disable in-table value editing here
            if not self.allow_in_table_editing:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)  #
            return item
        elif datatype == 'int' or datatype == 'text' or datatype == 'float' or datatype == 'negfloat':
            line_edit = QLineEdit(str(value) + str(unit))
            item = QTableWidgetItem(line_edit.text())  # Set the widget
            if not checkstate:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)   # no editing if not allowed in this mode
            return item
        # making float numeric for now...
        # elif datatype == 'float':
        #     slider = QSlider(Qt.Horizontal)
        #     slider.setValue(int(float(value) * 100))  # Assuming float values between 0 and 1
        #     item = QTableWidgetItem()
        #     item.setData(Qt.DisplayRole, slider)
        #     return item
        else:
            item = QTableWidgetItem(value)
            if not checkstate:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)   # no editing if not allowed in this mode
            return item

    def set_override_state(self, override_text):
        state = False
        if '(user)' not in override_text:
            state = False
        else:
            match self.edit_mode:
                case 'Global':
                    state = (override_text == 'Global (user)')
                case 'Sim':
                    state = (override_text == 'Sim (user)')
                case 'Class':
                    state = (override_text == 'Class (user)')
                case 'Model':
                    state =(override_text == 'Model (user)')
        return state

    def set_edit_mode(self,mode):
        oldmode = self.edit_mode

        if mode != oldmode:
            match mode:
                case 'MSFS' | 'IL2' | 'DCS':
                    mode = 'Sim'

            self.l_mode.setText(mode)
            self.edit_mode = mode
            self.setup_class_list()
            match mode:
                case 'Global':

                    self.drp_class.blockSignals(True)
                    self.drp_models.blockSignals(True)
                    self.drp_class.setCurrentText('')
                    self.drp_models.setCurrentText('')
                    self.model_name = ''
                    self.model_type = ''
                    self.drp_class.blockSignals(False)
                    self.drp_models.blockSignals(False)
                    self.drp_class.setEnabled(False)
                    self.drp_models.setEnabled(False)
                    self.b_createusermodel.setEnabled(False)
                case 'Sim':
                    self.drp_class.setEnabled(True)
                    self.drp_models.setEnabled(True)
                    self.drp_class.blockSignals(True)
                    self.drp_models.blockSignals(True)
                    self.drp_class.setCurrentText('')
                    self.drp_models.setCurrentText('')
                    self.model_name = ''
                    self.model_type = ''
                    self.drp_class.blockSignals(False)
                    self.drp_models.blockSignals(False)
                    self.b_createusermodel.setEnabled(True)

                case 'Class':
                    self.drp_class.setEnabled(True)
                    self.drp_models.setEnabled(True)
                    self.drp_models.blockSignals(True)
                    self.drp_models.setCurrentText('')
                    self.model_name = ''
                    self.drp_models.blockSignals(False)
                    self.b_createusermodel.setEnabled(True)

                case 'Model':
                    self.drp_class.setEnabled(True)
                    self.drp_models.setEnabled(True)
                    self.b_createusermodel.setEnabled(True)

                case _:
                    self.drp_class.setEnabled(True)
                    self.drp_models.setEnabled(True)

        lprint(f"{mode} Mode")


    def get_craft_attributes(file_path, sim, device):
        craft_attributes = set()
        craft_attributes.add('Aircraft')

        tree = ET.parse(file_path)
        root = tree.getroot()

        for defaults_elem in root.findall(f'.//defaults[{sim}="true"][{device}="true"]'):
            # for defaults_elem in root.findall(f'.//defaults[{sim}="true" and {device}="true"]'):
            for value_elem in defaults_elem.findall('.//value'):
                craft_attr = value_elem.get('Craft')
                if craft_attr is not None:
                    craft_attributes.add(craft_attr)

        return sorted(list(craft_attributes))

#  END GUI STUFF

    def read_xml_file(self, the_sim):
        tree = ET.parse(self.defaults_path)
        root = tree.getroot()

        # Collect data in a list of dictionaries
        data_list = []
        for defaults_elem in root.findall(f'.//defaults[{the_sim}="true"][{self.device}="true"]'):

            grouping = defaults_elem.find('Grouping').text
            name = defaults_elem.find('name').text
            # lprint(name)
            displayname = defaults_elem.find('displayname').text
            datatype = defaults_elem.find('datatype').text
            unit_elem = defaults_elem.find('unit')
            unit = unit_elem.text if unit_elem is not None else ""
            value_elem = defaults_elem.find('value')
            value = value_elem.text if value_elem is not None else ""
            if value is None: value = ""
            valid_elem = defaults_elem.find('validvalues')
            validvalues = valid_elem.text if valid_elem is not None else ""
            info_elem = defaults_elem.find('info')
            info = (f"{info_elem.text}") if info_elem is not None else ""
            prereq_elem = defaults_elem.find('prereq')
            prereq = (f"{prereq_elem.text}") if prereq_elem is not None else ""

            if the_sim == 'Global':
                replaced = 'Global'
            else:
                replaced = 'Sim Default'

            # Store data in a dictionary
            data_dict = {
                'grouping': grouping,
                'name': name,
                'displayname': displayname,
                'value': value,
                'unit': unit,
                'datatype': datatype,
                'validvalues': validvalues,
                'replaced': replaced,
                'prereq': prereq,
                'info': info
            }

            data_list.append(data_dict)

            # lprint(data_list)
        # Sort the data by grouping and then by name

        sorted_data = sorted(data_list, key=lambda x: (x['grouping'] != 'Basic', x['grouping'], x['displayname']))
        # lprint(sorted_data)
        # printconfig(sim, craft, sorted_data)
        return sorted_data

    def read_prereqs(self):
        tree = ET.parse(self.defaults_path)
        root = tree.getroot()

        # Collect data in a list of dictionaries
        data_list = []
        for defaults_elem in root.findall(f'.//defaults'):

            name = defaults_elem.find('name').text
            prereq_elem = defaults_elem.find('prereq')
            prereq = (f"{prereq_elem.text}") if prereq_elem is not None else ""

            # Store data in a dictionary
            data_dict = {
                'prereq': prereq,
                'value': 'False'
            }

            if data_dict is not None and data_dict['prereq'] != '' and data_dict not in data_list:
                data_list.append(data_dict)

            # lprint(data_list)

        # lprint(sorted_data)
        # printconfig(sim, craft, sorted_data)
        return data_list

    def check_prereq_value(self,datalist):
        for item in datalist:
            for prereq in self.prereq_list:
                if prereq['prereq'] == item['name']:
                    prereq['value'] = item['value']

    def is_prereq_satisfied(self,setting_name):
        tree = ET.parse(self.defaults_path)
        root = tree.getroot()
        result = True
        # Collect data in a list of dictionaries
        prereq_list = []
        for defaults_elem in root.findall(f'.//defaults[sim="{self.sim}"][device="{self.device}"]'):

            prereq_elem = defaults_elem.find('prereq')
            prereq = (f"{prereq_elem.text}") if prereq_elem is not None else ""
            prereq_dict = {'prereq': prereq}
            # Append to the list
            prereq_list.append(prereq_dict)

        if prereq_list != []:
            if setting_name in prereq_dict:
                print (prereq_dict['name'])

        return result

    def skip_bad_combos(self, craft):
        if self.sim == 'DCS' and craft == 'HPGHelicopter': return True
        if self.sim == 'DCS' and craft == 'TurbopropAircraft': return True
        if self.sim == 'DCS' and craft == 'GliderAircraft': return True
        if self.sim == 'IL2' and craft == 'GliderAircraft': return True
        if self.sim == 'IL2' and craft == 'HPGHelicopter': return True
        if self.sim == 'IL2' and craft == 'Helicopter': return True
        if self.sim == 'IL2' and craft == 'TurbopropAircraft': return True

        if self.sim == 'Global' and craft != 'Aircraft': return True
        return False


    def printconfig(self, sorted_data):
        # lprint("printconfig: " +sorted_data)
        show_source = False
        lprint("#############################################")

        # Print the sorted data with group names and headers
        current_group = None
        current_header = None
        for item in sorted_data:
            if item['grouping'] != current_group:
                current_group = item['grouping']
                if current_header is not None:
                    lprint("\n\n")  # Separate sections with a blank line
                lprint(f"\n# {current_group}")
            tabstring = "\t\t"
            replacestring = ''
            if show_source:
                if item['replaced'] == "Global": replacestring = " G"
                if item['replaced'] == "Global (user)": replacestring = "UG"
                if item['replaced'] == "Sim Default": replacestring =  "SD"
                if item['replaced'] == "Sim (user)": replacestring = "UD"
                if item['replaced'] == "Class Default": replacestring = "SC"
                if item['replaced'] == "Class (user)": replacestring = "UC"
                if item['replaced'] == "Model Default": replacestring = "DM"
                if item['replaced'] == "Model (user)": replacestring = "UM"
            spacing = 50 - (len(item['name']) + len(item['value']) + len(item['unit']))
            space = " " * spacing + " # " + replacestring + " # "

            lprint(f"{tabstring}{item['name']} = {item['value']} {item['unit']} {space} {item['info']}")


    def read_models_data(self,file_path, full_model_name):
        # runs on both defaults and userconfig xml files
        tree = ET.parse(file_path)
        root = tree.getroot()

        model_data = []
        found_pattern = ''

        # Iterate through models elements
        #for model_elem in root.findall(f'.//models[sim="{self.sim}"][device="{self.device}"]'):
        for model_elem in root.findall(f'.//models[sim="{self.sim}"][device="{self.device}"]') + \
                          root.findall(f'.//models[sim="any"][device="{self.device}"]') + \
                          root.findall(f'.//models[sim="{self.sim}"][device="any"]') + \
                          root.findall(f'.//models[sim="any"][device="any"]'):

            # Assuming 'model' is the element containing the wildcard pattern

            unit_pattern = model_elem.find('model')
            if unit_pattern is not None:
                pattern = unit_pattern.text
                if pattern is not None:
                    # Check if the full_model_name matches the pattern using re.match
                    if re.match(pattern, full_model_name):
                        name = model_elem.find('name').text
                        value = model_elem.find('value').text
                        unit_elem = model_elem.find('unit')
                        unit = unit_elem.text if unit_elem is not None else ""
                        model_dict = {
                            'name': name,
                            'value': value,
                            'unit': unit
                        }
                        found_pattern = pattern
                        model_data.append(model_dict)

        return model_data, found_pattern

    def read_default_class_data(self, the_sim, the_class):
        tree = ET.parse(self.defaults_path)
        root = tree.getroot()

        class_data = []

        # Iterate through models elements
        #for model_elem in root.findall(f'.//classdefaults[sim="{the_sim}"][type="{the_class}"][device="{self.device}"]'):
        #for model_elem in root.findall(f'.//classdefaults[sim="{the_sim}"][type="{the_class}"][device="{self.device}"]'):
        for model_elem in root.findall(f'.//classdefaults[sim="{the_sim}"][type="{the_class}"][device="{self.device}"]') + \
                          root.findall(f'.//classdefaults[sim="any"][type="{the_class}"][device="{self.device}"]') + \
                          root.findall(f'.//classdefaults[sim="{the_sim}"][type="{the_class}"][device="any"]') + \
                          root.findall(f'.//classdefaults[sim="any"][type="{the_class}"][device="any"]'):

            if model_elem.find('name') is not None:

                name = model_elem.find('name').text
                value = model_elem.find('value').text
                unit_elem = model_elem.find('unit')
                unit = unit_elem.text if unit_elem is not None else ""

                model_dict = {
                    'name': name,
                    'value': value,
                    'unit': unit,
                    'replaced': 'Class Default'
                }

                class_data.append(model_dict)

        return class_data

    def read_user_model_data(self, the_sim):
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        model_data = []

        # Iterate through models elements
        #for model_elem in root.findall(f'.//models[sim="{the_sim}"][device="{self.device}"]'):
        for model_elem in root.findall(f'.//models[sim="{the_sim}"][device="{self.device}"]') + \
                           root.findall(f'.//models[sim="any"][device="{self.device}"]') + \
                           root.findall(f'.//models[sim="{the_sim}"][device="any"]') + \
                           root.findall(f'.//models[sim="any"][device="any"]'):
            if model_elem.find('type') is None:
                if model_elem.find('name') is not None:

                    setting = model_elem.find('name').text
                    value = model_elem.find('value').text


                    model_dict = {
                        'name': setting,
                        'value': value,
                        'unit': '',
                        'replaced': 'Model (user)'
                    }

                    model_data.append(model_dict)

        return model_data

    def read_user_sim_data(self, the_sim):
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        sim_data = []

        # Iterate through models elements
        #for model_elem in root.findall(f'.//simSettings[sim="{the_sim}" or sim="any"][device="{self.device}" or device="any"]'):
        for model_elem in root.findall(f'.//simSettings[sim="{the_sim}"][device="{self.device}"]'):   # + \
                          # root.findall(f'.//simSettings[sim="any"][device="{self.device}"]') + \
                          # root.findall(f'.//simSettings[sim="{the_sim}"][device="any"]') + \
                          # root.findall(f'.//simSettings[sim="any"][device="any"]'):

            if model_elem.find('name') is not None:

                name = model_elem.find('name').text
                value = model_elem.find('value').text
                if the_sim == 'Global':
                    replaced = 'Global'
                else:
                    replaced = 'Sim (user)'
                model_dict = {
                    'name': name,
                    'value': value,
                    'unit': '',
                    'replaced': replaced
                }

                sim_data.append(model_dict)

        return sim_data

    def read_user_class_data(self, the_sim, crafttype):
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        model_data = []

        # Iterate through models elements
        #for model_elem in root.findall(f'.//models[sim="{the_sim}"][device="{self.device}"]'):
        for model_elem in root.findall(f'.//classSettings[sim="{the_sim}"][device="{self.device}"]'):     # + \
                          # root.findall(f'.//classSettings[sim="any"][device="{self.device}"]') + \
                          # root.findall(f'.//classSettings[sim="{the_sim}"][device="any"]') + \
                          # root.findall(f'.//classSettings[sim="any"][device="any"]'):
            if model_elem.find('type') is not None:
                # Assuming 'model' is the element containing the wildcard pattern
                pattern = model_elem.find('type').text

                if pattern is not None:
                    # Check if the craft type matches the pattern using re match
                    if re.match(pattern, crafttype):
                        name = model_elem.find('name').text
                        value = model_elem.find('value').text
                        unit_elem = model_elem.find('unit')
                        unit = unit_elem.text if unit_elem is not None else ""
                        model_dict = {
                            'name': name,
                            'value': value,
                            'unit': unit,
                            'replaced': 'Class (user)'
                        }

                        model_data.append(model_dict)

        return model_data

    def read_single_model(self, sim=None, aircraft_name=None):
        if sim is not None:
            if '.' in sim:
                input = sim.split('.')
                sim_temp = input[0]
                self.sim = sim_temp.replace('2020','')
                self.model_type = input[1]
            else:
                self.sim = sim
        if aircraft_name is not None:
            self.model_name = aircraft_name
        print_counts = False
        print_each_step = False  # for debugging

        # Read models data first
        model_data, def_model_pattern = self.read_models_data(self.defaults_path, self.model_name)
        user_model_data, usr_model_pattern = self.read_models_data(self.userconfig_path, self.model_name)

        model_pattern = def_model_pattern
        if usr_model_pattern != '':
            model_pattern = usr_model_pattern

        # Extract the type from models data, if name is blank then use the class.  otherwise assume no type is set.
        if self.model_name == '':
            model_class = self.model_type
        else:
            model_class = ''   #self.model_type

        for model in model_data:
            if model['name'] == 'type':
                model_class = model['value']
                break
        # check if theres an override
        if user_model_data is not None:
            for model in user_model_data:
                if model['name'] == 'type':
                    model_class = model['value']
                    break
        if model_class == '':
            model_class = self.input_model_type
        #lprint(f"class: {model_class}")

        # # get default settings for all sims and device
        # globaldata = self.read_xml_file('Global')
        #
        # if print_counts: lprint(f"globaldata count {len(globaldata)}")
        #
        # # see what we got
        # if print_each_step:
        #     lprint(f"\nGlobal result: Global  type: ''  device:{self.device}\n")
        #     self.printconfig(globaldata)


        # get default Aircraft settings for this sim and device
        simdata = self.read_xml_file(self.sim)

        if print_counts:  lprint(f"simdata count {len(simdata)}")

        # see what we got
        if print_each_step:
            lprint(f"\nSimresult: {self.sim} type: ''  device:{self.device}\n")
            self.printconfig(simdata)

        # combine base stuff
        defaultdata = simdata
        # if self.sim != 'Global':
        #     for item in simdata: defaultdata.append(item)

        if print_counts:  lprint(f"defaultdata count {len(defaultdata)}")


        # get additional class default data
        if model_class != "":
            # Use the extracted type in read_xml_file
            craftresult = self.read_default_class_data(self.sim,model_class)

            if craftresult is not None:

                # merge if there is any
                default_craft_result = self.update_default_data_with_craft_result(defaultdata, craftresult)
            else:
                default_craft_result = defaultdata

            if print_counts:  lprint(f"default_craft_result count {len(default_craft_result)}")

            # see what we got
            if print_each_step:
                lprint(f"\nDefaultsresult: {self.sim} type: {model_class}  device:{self.device}\n")
                self.printconfig(default_craft_result)
        else:
            default_craft_result = defaultdata


        # get userconfig global overrides
        userglobaldata = self.read_user_sim_data( 'Global')
        if userglobaldata is not None:
            # merge if there is any
            def_craft_userglobal_result = self.update_data_with_models(default_craft_result, userglobaldata,'Global (user)')
        else:
            def_craft_userglobal_result = default_craft_result

        if print_counts:  lprint(f"def_craft_userglobal_result count {len(def_craft_userglobal_result)}")


        # get userconfig sim overrides
        if self.sim != 'Global':
            user_default_data = self.read_user_sim_data(self.sim)
            if user_default_data is not None:
                # merge if there is any
                def_craft_user_default_result = self.update_data_with_models(def_craft_userglobal_result, user_default_data, 'Sim (user)')
            else:
                def_craft_user_default_result = def_craft_userglobal_result

            if print_counts:  lprint(f"def_craft_user_default_result count {len(def_craft_user_default_result)}")
        else:
            def_craft_user_default_result = def_craft_userglobal_result

        if model_class != "":
            # get userconfg craft specific type overrides
            usercraftdata = self.read_user_class_data(self.sim, model_class)
            if usercraftdata is not None:
                # merge if there is any
                def_craft_usercraft_result = self.update_data_with_models(def_craft_user_default_result, usercraftdata, 'Class (user)')
            else:
                def_craft_usercraft_result = def_craft_user_default_result
        else:
            def_craft_usercraft_result = def_craft_user_default_result


        # Update result with default models data
        def_craft_models_result = self.update_data_with_models(def_craft_usercraft_result, model_data, 'Model Default')

        if print_counts:  lprint(f"def_craft_models count {len(def_craft_models_result)}")


        # finally get userconfig model specific overrides
        if user_model_data:
            final_result = self.update_data_with_models(def_craft_models_result, user_model_data, 'Model (user)')
        else:
            final_result = def_craft_models_result

        final_result = [item for item in final_result if item['value'] != '' or item['name'] == 'vpconf']

        self.prereq_list = self.read_prereqs()
        self.check_prereq_value(final_result)

        sorted_data = sorted(final_result, key=lambda x: (x['grouping'] != 'Basic', x['grouping'], x['name']))
        #lprint(f"final count {len(final_result)}")


        return model_class, model_pattern, sorted_data


    def write_models_to_xml(self, the_device, the_sim, the_model, the_value, setting_name):
        # Load the existing XML file or create a new one if it doesn't exist
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        # Check if an identical <models> element already exists
        model_elem = root.find(f'.//models' #[sim="{the_sim}"]'
                               f'[device="{the_device}"]'
                               f'[model="{the_model}"]'
                               f'[name="{setting_name}"]')

        if model_elem is not None:
            # Update the value of the existing element
            for child_elem in model_elem:
                if child_elem.tag == 'value':
                    child_elem.text = str(the_value)
                if child_elem.tag == 'sim':
                    if child_elem.text == 'any':
                        child_elem.text = the_sim
            tree.write(self.userconfig_path)
            lprint(f"Updated <models> element with values: sim={the_sim}, device={the_device}, "
                  f"value={the_value}, model={the_model}, name={setting_name}")

        else:
            # Check if an identical <models> element already exists; if so, skip
            model_elem_exists = any(
                all(
                    element.tag == tag and element.text == value
                    for tag, value in [
                        ("name", setting_name),
                        ("model", the_model),
                        ("value", the_value),
                        ("sim", the_sim),
                        ("device", the_device)
                    ]
                )
                for element in root.iter("models")
            )

            if model_elem_exists:
                lprint("<models> element with the same values already exists. Skipping.")
            else:
                # Create child elements with the specified content
                models = ET.SubElement(root, "models")
                for tag, value in [("name", setting_name),
                                   ("model", the_model),
                                   ("value", the_value),
                                   ("sim", the_sim),
                                   ("device", the_device)]:
                    ET.SubElement(models, tag).text = value

                # Write the modified XML back to the file
                tree = ET.ElementTree(root)
                tree.write(self.userconfig_path)
                lprint(f"Added <models> element with values: sim={the_sim}, device={the_device}, "
                      f"value={the_value}, model={the_model}, name={setting_name}")


    def write_class_to_xml(self, the_device, the_sim, the_class, the_value, setting_name):
        # Load the existing XML file or create a new one if it doesn't exist
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        # Check if an identical <classSettings> element already exists
        class_elem = root.find(f'.//classSettings[sim="{the_sim}"]'
                               f'[device="{the_device}"]'
                               f'[type="{the_class}"]'
                               f'[name="{setting_name}"]')

        if class_elem is not None:
            # Update the value of the existing element
            for child_elem in class_elem:
                if child_elem.tag == 'value':
                    child_elem.text = str(the_value)
            tree.write(self.userconfig_path)
            lprint(f"Updated <classSettings> element with values: sim={the_sim}, device={the_device}, "
                  f"value={the_value}, model={the_class}, name={setting_name}")

        else:
            # Create a new <classSettings> element
            classes = ET.SubElement(root, "classSettings")
            for tag, value in [("name", setting_name),
                               ("type", the_class),
                               ("value", the_value),
                               ("sim", the_sim),
                               ("device", the_device)]:
                ET.SubElement(classes, tag).text = value

            # Write the modified XML back to the file
            tree = ET.ElementTree(root)
            tree.write(self.userconfig_path)
            lprint(f"Added <classSettings> element with values: sim={the_sim}, device={the_device}, "
                  f"value={the_value}, type={the_class}, name={setting_name}")


    def write_sim_to_xml(self, the_device, the_sim, the_value, setting_name):
        # Load the existing XML file or create a new one if it doesn't exist
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        # Check if an identical <simSettings> element already exists
        sim_elem = root.find(f'.//simSettings[sim="{the_sim}"]'
                             f'[device="{the_device}"]'
                             f'[name="{setting_name}"]')

        if sim_elem is not None:
            # Update the value of the existing element
            for child_elem in sim_elem:
                if child_elem.tag == 'value':
                    child_elem.text = str(the_value)
            tree.write(self.userconfig_path)
            lprint(f"Updated <simSettings> element with values: sim={the_sim}, device={the_device}, "
                  f"value={the_value}, name={setting_name}")

        else:
            # Create a new <simSettings> element
            sims = ET.SubElement(root, "simSettings")
            for tag, value in [("name", setting_name),
                               ("value", the_value),
                               ("sim", the_sim),
                               ("device", the_device)]:
                ET.SubElement(sims, tag).text = value

            # Write the modified XML back to the file
            tree = ET.ElementTree(root)
            tree.write(self.userconfig_path)
            lprint(
                f"Added <simSettings> element with values: sim={the_sim}, device={the_device}, value={the_value}, name={setting_name}")


    def write_converted_to_xml(self, differences):
        sim_set = []
        class_set = []
        model_set = []

        for dif in differences:
            if dif['sim'] == 'any':
                model_set.append(dif)
            else:
                if dif['class'] != '':
                    class_set.append(dif)
                else:
                    sim_set.append(dif)
        for s in sim_set:
            self.write_sim_to_xml(s['device'], s['sim'], s['value'], s['name'])
        for c in class_set:
            self.write_class_to_xml(c['device'], c['sim'], c['class'], c['value'], c['name'])
        for m in model_set:
            self.write_models_to_xml(m['device'], m['sim'], m['model'], m['value'], m['name'])

    def erase_models_from_xml(self, the_model, the_value, setting_name):
        # Load the existing XML file or create a new one if it doesn't exist
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()
        elements_to_remove = []
        for model_elem in root.findall(f'models[sim="{self.sim}"]'
                                       f'[device="{self.device}"]'
                                       f'[value="{the_value}"]'
                                       f'[model="{the_model}"]'
                                       f'[name="{setting_name}"]'):

            if model_elem is not None:
                elements_to_remove.append(model_elem)
            else:
                print ("not found")

        # Remove the elements outside the loop
        for elem in elements_to_remove:
            root.remove(elem)
            # Write the modified XML back to the file
            tree.write(self.userconfig_path)
            lprint(f"Removed <models> element with values: sim={self.sim}, device={self.device}, "
                      f"value={the_value}, model={the_model}, name={setting_name}")


    def erase_class_from_xml(self, the_class, the_value, setting_name):
        # Load the existing XML file or create a new one if it doesn't exist
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()
        elements_to_remove = []
        for class_elem in root.findall(f'.//classSettings[sim="{self.sim}"]'
                                       f'[device="{self.device}"]'
                                       f'[value="{the_value}"]'
                                       f'[type="{the_class}"]'
                                       f'[name="{setting_name}"]'):

            if class_elem is not None:
                elements_to_remove.append(class_elem)
            else:
                print ("not found")

        # Remove the elements outside the loop
        for elem in elements_to_remove:
            root.remove(elem)
            # Write the modified XML back to the file
            tree.write(self.userconfig_path)
            lprint(f"Removed <classSettings> element with values: sim={self.sim}, device={self.device}, "
                      f"value={the_value}, type={the_class}, name={setting_name}")


    def erase_sim_from_xml(self, the_value, setting_name):
        # Load the existing XML file or create a new one if it doesn't exist
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()

        elements_to_remove = []
        for sim_elem in root.findall(f'.//simSettings[sim="{self.sim}"]'
                                       f'[device="{self.device}"]'
                                       f'[value="{the_value}"]'
                                       f'[name="{setting_name}"]'):

            if sim_elem is not None:
                elements_to_remove.append(sim_elem)
            else:
                print ("not found")

        # Remove the elements outside the loop
        for elem in elements_to_remove:
            root.remove(elem)
            # Write the modified XML back to the file
            tree.write(self.userconfig_path)
            lprint(f"Removed <simSettings> element with values: sim={self.sim}, device={self.device}, value={the_value}, name={setting_name}")

    def update_default_data_with_craft_result(self,defaultdata, craftresult):
        updated_defaultdata = defaultdata.copy()  # Create a copy to avoid modifying the original data


        #default_names = set(item['name'] for item in defaultdata)

        # Iterate through craftresult
        for craft_item in craftresult:
            name = craft_item['name']

            # Check if the item with the same name exists in defaultdata
            matching_item = next((item for item in updated_defaultdata if item['name'] == name), None)

            if matching_item:
                # If the item exists, update 'value' and 'unit'
                matching_item['value'] = craft_item['value']
                matching_item['unit'] = craft_item['unit']
                matching_item['replaced'] = "Class Default"  # Set the 'replaced' flag
            #else:
            # If the item doesn't exist, append it to defaultdata
                # no appending
                # defaultdata.append(craft_item)

        return updated_defaultdata

    def update_data_with_models(self,defaults_data, model_data, replacetext):
        updated_result = defaults_data.copy()

        # Create a dictionary mapping settings to their corresponding values and units
        model_dict = {model['name']: {'value': model['value'], 'unit': model['unit']} for model in model_data}

        for item in updated_result:
            name = item['name']

            # Check if the setting exists in the model_data
            if name in model_dict:
                # Update the value and unit in defaults_data with the values from model_data
                item['value'] = model_dict[name]['value']
                item['unit'] = model_dict[name]['unit']
                item['replaced'] = replacetext  # Set the 'replaced' text



        return updated_result
    def update_data_with_models_old(self,defaults_data, model_data, replacetext):
        updated_result = []

        # Create a dictionary mapping settings to their corresponding values and units
        model_dict = {model['name']: {'value': model['value'], 'unit': model['unit']} for model in model_data}

        for item in defaults_data:
            name = item['name']

            # Check if the setting exists in the model_data
            if name in model_dict:
                # Update the value and unit in defaults_data with the values from model_data
                item['value'] = model_dict[name]['value']
                item['unit'] = model_dict[name]['unit']
                item['replaced'] = replacetext  # Set the 'replaced' text

            updated_result.append(item)

        return updated_result

    def create_empty_userxml_file(self):
        # Create a backup directory if it doesn't exist
        # uncomment later when we use %localappdata%
        # os.makedirs(os.path.dirname(self.userconfig_path), exist_ok=True)

        if not os.path.isfile(self.userconfig_path):
            # Create an empty XML file with the specified root element
            root = ET.Element("TelemFFB")
            tree = ET.ElementTree(root)

            if not os.path.exists(self.userconfig_rootpath):
                os.makedirs(self.userconfig_rootpath)
            tree.write(self.userconfig_path)
            lprint(f"Empty XML file created at {self.userconfig_path}")
        else:
            lprint(f"XML file exists at {self.userconfig_path}")

    def read_models(self,the_sim):
        all_models = ['']
        tree = ET.parse(self.defaults_path)
        root = tree.getroot()
        #for model_elem in root.findall(f'.//models[sim="{mysim}"][device="{self.device}"]'):
        for model_elem in root.findall(f'.//models[sim="{the_sim}"][device="{self.device}"]') +\
            root.findall(f'.//models[sim="any"][device="{self.device}"]') +\
            root.findall(f'.//models[sim="{the_sim}"][device="any"]') +\
            root.findall(f'.//models[sim="any"][device="any"]'):

            pattern = model_elem.find('model')
            #print (pattern.text)
            if pattern is not None:
                if pattern.text not in all_models:
                    all_models.append(pattern.text)

        self.create_empty_userxml_file()
        tree = ET.parse(self.userconfig_path)
        root = tree.getroot()
        #for model_elem in root.findall(f'.//models[sim="{mysim}"][device="{self.device}"]'):
        for model_elem in root.findall(f'.//models[sim="{the_sim}"][device="{self.device}"]') + \
                          root.findall(f'.//models[sim="any"][device="{self.device}"]') + \
                          root.findall(f'.//models[sim="{the_sim}"][device="any"]') + \
                          root.findall(f'.//models[sim="any"][device="any"]'):

            pattern = model_elem.find('model')
            #print (pattern.text)
            if pattern is not None:
                if pattern.text not in all_models:
                    all_models.append(pattern.text)

        return sorted(all_models)


class UserModelDialog(QDialog):
    def __init__(self, sim, current_aircraft, current_type, parent=None):
        super(UserModelDialog, self).__init__(parent)
        self.combo_box = None
        self.tb_current_aircraft = None
        self.setWindowTitle("Create Model Setting")
        self.init_ui(sim, current_aircraft,current_type)

    def init_ui(self,sim,current_aircraft,current_type):


        layout = QVBoxLayout()

        label1 = QLabel("TelemFFB uses regex to match aircraft names")
        label2 = QLabel("Name.* will match anything starting with 'Name'")
        label3 = QLabel("^Name$ will match only the exact 'Name'")
        label4 = QLabel("(The )?Name.* matches starting with 'Name' or 'The Name'" )
        label5 = QLabel("Edit the match pattern below.")

        label6 = QLabel("And choose the aircraft class:")

        classes = []
        match sim:
            case 'DCS':
                classes = ["PropellerAircraft", "JetAircraft", "Helicopter"]
            case 'IL2':
                classes = ["PropellerAircraft", "JetAircraft"]
            case 'MSFS':
                classes = ['PropellerAircraft', 'TurbopropAircraft', 'JetAircraft', 'GliderAircraft', 'Helicopter', 'HPGHelicopter']

        label_aircraft = QtWidgets.QLabel("Current Aircraft:")
        self.tb_current_aircraft = QtWidgets.QLineEdit()
        self.tb_current_aircraft.setText(current_aircraft)
        self.tb_current_aircraft.setAlignment(Qt.AlignHCenter)

        self.combo_box = QComboBox()
        self.combo_box.addItems(classes)
        self.combo_box.setStyleSheet("QComboBox::view-item { align-text: center; }")
        self.combo_box.setCurrentText(current_type)

        ok_button = QPushButton("OK")
        ok_button.setStyleSheet("text-align:center;")
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("text-align:center;")

        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addWidget(label3)
        layout.addWidget(label4)
        layout.addWidget(label5)

        layout.addWidget(self.tb_current_aircraft)

        layout.addWidget(label6)
        layout.addWidget(self.combo_box)

        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)

        self.setLayout(layout)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)


# #  for printing config.ini text file
# def print_all_defaults():    # none of this may work outside teh class
#     device = "joystick"
#     for the_sim in "Global", "DCS", "MSFS", "IL2":
#         crafts = SettingsWindow.get_craft_attributes(defaults_path, the_sim, device)
#         for craft in crafts:
#             skip = SettingsWindow.skip_bad_combos(the_sim, craft)
#             if skip == True: continue
#             mydata = SettingsWindow.read_xml_file(the_sim, craft)
#             # lprint("main: "+ mydata)
#             SettingsWindow.printconfig( mydata)


if __name__ == "__main__":
    # defaults_path = "defaults.xml"  # defaults file
    # userconfig_path = "userconfig.xml"  # user config overrides stored here, will move to %localappdata% at some point
    # # defaults_path = "debug_defaults.xml"  # defaults file
    # # userconfig_path = "debug_userconfig.xml"  # user config overrides stored here, will move to %localappdata% at some point
    # device = "joystick"  # joystick, pedals, collective.  essentially permanent per session

    # stuff below is fo printing config.ini, no longer used.

    # defaultdata = []
    # mydata = []
    #
    # # output a single aircraft class config
    #
    # defaultdata = read_xml_file(xml_file_path,sim,"Aircraft",device)
    # printconfig(sim, "Aircraft", defaultdata)
    #
    # #defaultdata = read_xml_file(xml_file_path,sim,crafttype,device)
    # #printconfig(sim, crafttype, defaultdata)

    # output a single model
    # model_type, model_pattern, mydata = read_single_model(defaults_path, sim, model_name, device, crafttype)

    # lprint(f"\nData for: {sim}  model: {model_name}  pattern: {model_pattern}  class: {model_type}  device:{device}\n")

    # printconfig(sim, model_type, mydata)

    # output all default configs for device
    # print_all_defaults()

    #  GUI stuff below

    app = QApplication(sys.argv)
    sw = SettingsWindow(defaults_path='defaults.xml', userconfig_path='userconfig.xml', device='joystick')

    sw.show()

    sys.exit(app.exec_())

