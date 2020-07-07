#!/usr/bin/env python3
# -*- coding: utf-8 -*

import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QCursor, QFont
import os
import pickle
import numpy as np
import glob
import time
from pathlib import Path


import music21
from music21 import instrument, note, stream, chord, duration
import tensorflow.keras.callbacks
import tensorflow.keras.backend as K
from tensorflow.keras import utils
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape, \
    Flatten, RepeatVector, Permute, TimeDistributed, Multiply, Lambda, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model


class LoadFilesWindow(QtWidgets.QDialog):
    """First Window with elements to load midi files, select model parameters,
    load pretrained models and to start the training
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Load midi files')
        self.horizontal_layout = QtWidgets.QHBoxLayout(self)
        self.horizontal_layout.setContentsMargins(10, 10, 10, 10)
        self.horizontal_layout.setSpacing(10)
        self.summary_string = ""
        self.model, self.model_with_att = None, None
        self.network_data = []
        self.file_dialog = QtWidgets.QFileDialog()
        self.folder_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.file_list_widget = QtWidgets.QListWidget()
        self.make_file_list_layout()
        self.spinbox_seq_length = QtWidgets.QSpinBox()
        self.spinbox_embed_size = QtWidgets.QSpinBox()
        self.spinbox_rnn_unit = QtWidgets.QSpinBox()
        self.make_right_layout()
        self.model_name_string = ""
        self.n_notes = 0
        self.n_durations = 0
        self.embed_size = 0
        self.rnn_units = 0
        self.use_attention = False
        self.model_parameters = []
        self.notes_lookups = []
        self.durations_lookups = []
        self.file_list = [
            '/home/carlo/PycharmProjects/compose_tool/samples/cello/cs1-6gig.mid',
            '/home/carlo/PycharmProjects/compose_tool/samples/cello/cs1-5men.mid',
            '/home/carlo/PycharmProjects/compose_tool/samples/cello/cs1-2all.mid',
            '/home/carlo/PycharmProjects/compose_tool/samples/cello/cs1-3cou.mid',
            '/home/carlo/PycharmProjects/compose_tool/samples/cello/cs1-1pre.mid',
            '/home/carlo/PycharmProjects/compose_tool/samples/cello/cs1-4sar.mid'
                          ]
        self.confirmation_window = None

    def skip_loading(self):
        folder = '/home/carlo/PycharmProjects/compose_tool/data/Peter'

        self.model_name_string = folder.split('/')[-1]
        folder = Path(folder)
        
        try:
            with open(folder / 'store' / 'lookups', 'rb') as saved_lookups:
                lookups = pickle.load(saved_lookups)
                self.notes_lookups, self.durations_lookups = lookups
        except FileNotFoundError:
            pass
        try:
            with open(folder / 'store' / 'model_parameters', 'rb') as saved_parameters:
                self.model_parameters = pickle.load(saved_parameters)
            self.model, self.model_with_att = LoadFilesWindow.make_network(
                self.model_parameters[0], self.model_parameters[1], self.model_parameters[2],
                self.model_parameters[3], self.model_parameters[4]
                                                           )
            weights_folder = Path(__file__).parent / 'data' / self.model_name_string / 'weights'
            self.model.load_weights(str(weights_folder / 'weights.h5'))
            compose_window = ComposeWindow(self)
            compose_window.show()

        except FileNotFoundError:
            pass

    def make_file_list_layout(self):
        """Create and arrange the layout for the file list elements on the left side.
        """
        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.setContentsMargins(10, 10, 10, 10)
        vertical_layout.setSpacing(10)
        self.horizontal_layout.addLayout(vertical_layout)
        push_button_load_files = QtWidgets.QPushButton()
        push_button_load_files.setMinimumSize(QSize(100, 30))
        push_button_load_files.setText('Add midi files')
        push_button_load_files.clicked.connect(self.open_file_dialog)
        push_button_load_folder = QtWidgets.QPushButton()
        push_button_load_folder.setMinimumSize(QSize(100, 30))
        push_button_load_folder.setText('Add folder')
        push_button_load_folder.clicked.connect(self.open_folder_dialog)
        label_list_widget = QtWidgets.QLabel('Midi files to train the network:')
        push_button_remove_file = QtWidgets.QPushButton()
        push_button_remove_file.setText('Remove selected file')
        push_button_remove_file.clicked.connect(self.remove_file)

        vertical_layout.addWidget(push_button_load_files)
        vertical_layout.addWidget(push_button_load_folder)
        vertical_layout.addWidget(label_list_widget)

        vertical_layout.addWidget(self.file_list_widget)
        vertical_layout.addWidget(push_button_remove_file)

    def make_right_layout(self):
        """Create and arrange the layout for the parameters and the buttons on the right side.
        """
        vertical_layout_right = QtWidgets.QVBoxLayout()
        self.horizontal_layout.addLayout(vertical_layout_right)
        self.spinbox_seq_length.setMaximum(10000)
        self.spinbox_embed_size.setMaximum(10000)
        self.spinbox_rnn_unit.setMaximum(10000)
        self.spinbox_seq_length.setValue(32)
        self.spinbox_embed_size.setValue(100)
        self.spinbox_rnn_unit.setValue(256)
        label_seq_length = QtWidgets.QLabel()
        label_seq_length.setText('Sequence length')
        label_embed_size = QtWidgets.QLabel()
        label_embed_size.setText('Embedding size')
        label_rnn_unit = QtWidgets.QLabel()
        label_rnn_unit.setText('rnn unit')

        button_start_training = QtWidgets.QPushButton()
        button_start_training.setText('Prepare Training')
        button_start_training.clicked.connect(self.prepare_training)
        
        button_load_pretrained_model = QtWidgets.QPushButton('Load pretrained model')
        button_load_pretrained_model.clicked.connect(self.load_model)
        #button_load_pretrained_model.clicked.connect(self.skip_loading)
        
        vertical_layout_right.addWidget(label_seq_length)
        vertical_layout_right.addWidget(self.spinbox_seq_length)
        vertical_layout_right.addWidget(label_embed_size)
        vertical_layout_right.addWidget(self.spinbox_embed_size)
        vertical_layout_right.addWidget(label_rnn_unit)
        vertical_layout_right.addWidget(self.spinbox_rnn_unit)

        vertical_layout_right.addWidget(button_start_training)
        vertical_layout_right.addWidget(button_load_pretrained_model)

    @staticmethod
    def midi_converter(file_list, seq_length):
        """Converts the given list of midi_files a the tupel of lists (notes, durations).
        """
        parser = music21.converter
        interval = 0
        notes = []
        durations = []

        if file_list:
            for file in file_list:
                chordified_score = parser.parse(file).chordify()
                notes.extend(['START'] * seq_length)
                durations.extend([0] * seq_length)

                score_new = chordified_score.transpose(interval)

                for element in score_new.flat:

                    if isinstance(element, music21.note.Note):
                        if element.isRest:
                            notes.append(str(element.name))
                            durations.append(element.duration.quarterLength)
                        else:
                            notes.append(str(element.nameWithOctave))
                            durations.append(element.duration.quarterLength)

                    if isinstance(element, music21.chord.Chord):
                        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                        durations.append(element.duration.quarterLength)

            return notes, durations

    @staticmethod
    def prepare_sequences(element, seq_length):
        """Gets the input and output sequences of the given element to the right format of the neural network.
        Creates lookup-tables (note/duration <-> int) for the given element.
        Takes list of notes/durations and sequence length as argument.
        Returns tuple of (input_sequence, output_sequence, element_to_int, int_to_element)
        """
        input_sequence = []
        output_sequence = []
        element_to_int, int_to_element = LoadFilesWindow.make_element_int_dicts(element)

        for i in range(len(element) - seq_length):
            sequence_in = element[i:i + seq_length]
            sequence_out = element[i + seq_length]
            input_sequence.append([element_to_int[char] for char in sequence_in])
            output_sequence.append(element_to_int[sequence_out])

        input_sequence = np.reshape(input_sequence, (len(input_sequence), seq_length))
        output_sequence = utils.to_categorical(output_sequence, num_classes=len(element))

        return input_sequence, output_sequence, element_to_int, int_to_element

    @staticmethod
    def make_element_int_dicts(element):
        """Returns tuple of dictionaries (element to int, int to element) for given element (notes, durations).
        """
        element_names = sorted(set(element))
        element_to_int = dict((element, number) for number, element in enumerate(element_names))
        int_to_element = dict((number, element) for number, element in enumerate(element_names))
        return element_to_int, int_to_element

    @staticmethod
    def make_network(n_notes, n_durations, embed_size=100, rnn_units=256, use_attention=False):
        """Creates the structure of the neural network.
        Returns tuple of model and model with attention.
        """
        notes_in = Input(shape=(None,))
        durations_in = Input(shape=(None,))

        embedding_layer_notes = Embedding(n_notes, embed_size)(notes_in)
        embedding_layer_durations = Embedding(n_durations, embed_size)(durations_in)

        concatenate = Concatenate()([embedding_layer_notes, embedding_layer_durations])
        concatenate = LSTM(rnn_units, return_sequences=True)(concatenate)

        if use_attention:

            concatenate = LSTM(rnn_units, return_sequences=True)(concatenate)
            layer_dense = Dense(1, activation='tanh')(concatenate)
            layer_dense = Reshape([-1])(layer_dense)
            activation = Activation('softmax')(layer_dense)
            activation_repeated = Permute([2, 1])(RepeatVector(rnn_units)(activation))

            layer2 = Multiply()([concatenate, activation_repeated])
            layer2 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(layer2)

        else:
            layer2 = LSTM(rnn_units)(concatenate)

        notes_out = Dense(n_notes, activation='softmax', name='pitch')(layer2)
        durations_out = Dense(n_durations, activation='softmax', name='duration')(layer2)

        model = Model([notes_in, durations_in], [notes_out, durations_out])

        if use_attention:
            model_with_att = Model([notes_in, durations_in], activation)
        else:
            model_with_att = None

        optimizer = RMSprop(lr=0.001)
        model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer)

        return model, model_with_att

    def prepare_training(self):
        """Start training preparation. Load StartTrainingWindow when finished."""
        if self.file_list:
            embed_size = self.spinbox_embed_size.value()
            rnn_units = self.spinbox_rnn_unit.value()
            use_attention = True
            seq_length = self.spinbox_seq_length.value()

            notes, durations = LoadFilesWindow.midi_converter(self.file_list, seq_length)
            notes_input_sequence, notes_output_sequence, *self.notes_lookups = \
                LoadFilesWindow.prepare_sequences(notes, seq_length)
            durations_input_sequence, durations_output_sequence, *self.durations_lookups = \
                LoadFilesWindow.prepare_sequences(durations, seq_length)
            self.model, self.model_with_att = LoadFilesWindow.make_network(
                len(notes), len(durations), embed_size, rnn_units, use_attention
                                                           )
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            self.summary_string = "\n".join(summary_list)

            self.confirmation_window = StartTrainingWindow(self)
            self.confirmation_window.label = self.summary_string

            self.network_data = [[notes_input_sequence, durations_input_sequence], [notes_output_sequence, durations_output_sequence]]
            self.model_parameters = [len(notes), len(durations), embed_size, rnn_units, use_attention]
            self.confirmation_window.show()

        else:
            message_box = QtWidgets.QMessageBox(self)
            message_box.setText('No midi file selected')
            message_box.show()
            
    def load_model(self):
        """Loads pretrained model and shows ComposeWindow"""
        load_model_dialog = QtWidgets.QFileDialog()
        load_model_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        folder = load_model_dialog.getExistingDirectory(self, 'Select folder with model data')
        self.model_name_string = folder.split('/')[-1]
        folder = Path(folder)
        
        try:
            with open(folder / 'store' / 'lookups', 'rb') as saved_lookups:
                lookups = pickle.load(saved_lookups)
                self.notes_lookups, self.durations_lookups = lookups
        except FileNotFoundError:
            pass
        try:
            with open(folder / 'store' / 'model_parameters', 'rb') as saved_parameters:
                self.model_parameters = pickle.load(saved_parameters)
            self.model, self.model_with_att = LoadFilesWindow.make_network(
                self.model_parameters[0], self.model_parameters[1], self.model_parameters[2],
                self.model_parameters[3], self.model_parameters[4]
                                                           )
            weights_folder = Path(__file__).parent / 'data' / self.model_name_string / 'weights'
            self.model.load_weights(str(weights_folder / 'weights.h5'))
            compose_window = ComposeWindow(self)
            compose_window.progress_dialog.close()
            compose_window.label_model_name.setText(f'Model name: {main.load_files_window.model_name_string}')
            compose_window.show()

        except FileNotFoundError:
            pass

    def open_file_dialog(self):
        """Opens file dialog for the midi files.
        """
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        files = self.file_dialog.getOpenFileNames(self, 'Select midi files!', "", 'Midi files (*.mid)')[0]
        self.file_list += files
        self.refresh_list_widget()

    def open_folder_dialog(self):
        """Opens folder dialog to get all containing midi files (subfolders included).
        """
        self.folder_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        folder = self.folder_dialog.getExistingDirectory(self, 'Select folder!')
        if folder:
            self.get_all_files(folder)
            self.refresh_list_widget()

    def get_all_files(self, folder):
        """Adds all files in the given folder and subfolders to self.file_list.
        """
        for item in Path(folder).iterdir():
            if item.is_file() and item.suffix == '.mid':
                self.file_list.append(str(item))
            elif item.is_dir():
                self.get_all_files(item)

    def remove_file(self):
        """Removes the selected file from the list widget and the related list self.file_list.
        """
        try:
            selected_item = self.file_list_widget.selectedItems()[0]
            self.file_list.remove(selected_item.text())
            self.file_list_widget.takeItem(self.file_list_widget.row(selected_item))

        except IndexError:
            pass

    def refresh_list_widget(self):
        """Updates the list widget with the related list self.file_list. Removes double entries.
        """
        self.file_list = list(set(self.file_list))
        self.file_list_widget.clear()
        self.file_list_widget.addItems(self.file_list)


class StartTrainingWindow(QtWidgets.QDialog):
    """Start Training Window with options to change the training parameters and to start the training.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Prepare Training')
        self.horizontal_layout = QtWidgets.QHBoxLayout(self)
        self.horizontal_layout.setContentsMargins(10, 10, 10, 10)
        self.horizontal_layout.setSpacing(10)

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.setContentsMargins(10, 10, 10, 10)
        vertical_layout.setSpacing(10)
        label = QtWidgets.QLabel(main.load_files_window.summary_string)
        font = QFont()
        font.setStyleHint(QFont().Monospace)
        font.setFamily('monospace')
        label.setFont(font)

        grid_layout = QtWidgets.QGridLayout()
        self.horizontal_layout.addWidget(label)
        label_model_name = QtWidgets.QLabel('Model name:')
        self.line_edit_model_name = QtWidgets.QLineEdit('Peter')
        label_epochs = QtWidgets.QLabel('Epochs:')
        self.spinbox_epochs = QtWidgets.QSpinBox()
        self.spinbox_epochs.setMaximum(999999999)
        self.spinbox_epochs.setValue(20)
        label_batch_size = QtWidgets.QLabel('Batch size:')
        self.spinbox_batch_size = QtWidgets.QSpinBox()
        self.spinbox_batch_size.setMaximum(99999)
        self.spinbox_batch_size.setValue(32)
        label_validation_split = QtWidgets.QLabel('Validation split:')
        self.spinbox_validation_split = QtWidgets.QDoubleSpinBox()
        self.spinbox_validation_split.setMaximum(1)
        self.spinbox_validation_split.setSingleStep(0.1)
        self.spinbox_validation_split.setValue(0.2)
        label_early_stop_patience = QtWidgets.QLabel('Early stop patience:')
        self.spinbox_early_stop_patience = QtWidgets.QSpinBox()
        self.spinbox_early_stop_patience.setMaximum(100)
        self.spinbox_early_stop_patience.setValue(10)
        self.cancel_flag = False
        self.compose_window = None

        push_button_start_training = QtWidgets.QPushButton()
        push_button_start_training.setText('Start Training')
        push_button_start_training.clicked.connect(self.start_training)
        grid_layout.addWidget(label_model_name, 0, 0)
        grid_layout.addWidget(self.line_edit_model_name, 0, 1)
        grid_layout.addWidget(label_epochs, 1, 0)
        grid_layout.addWidget(self.spinbox_epochs, 1, 1)
        grid_layout.addWidget(label_batch_size, 2, 0)
        grid_layout.addWidget(self.spinbox_batch_size, 2, 1)
        grid_layout.addWidget(label_validation_split, 3, 0)
        grid_layout.addWidget(self.spinbox_validation_split, 3, 1)
        grid_layout.addWidget(label_early_stop_patience, 4, 0)
        grid_layout.addWidget(self.spinbox_early_stop_patience, 4, 1)

        grid_layout.addWidget(push_button_start_training, 5, 1)

        self.horizontal_layout.addLayout(grid_layout)

    def cancel_action(self):
        """Action called by cancel button of the progress dialog.
        """
        self.cancel_flag = True

    def start_training(self):
        """Starts the training process. Creates a folder named by the project name.
        Saves the model data in the subfolder 'store' and the weights in 'weights'.
        """
        self.cancel_flag = False
        main.load_files_window.model_name_string = self.line_edit_model_name.text()
        self.compose_window = ComposeWindow(self)
        self.compose_window.label_model_name.setText(f'Model name: {main.load_files_window.model_name_string}')
        epochs = self.spinbox_epochs.value()

        self.compose_window.progress_dialog.reset()
        self.compose_window.progress_dialog.setRange(0, epochs)
        cancel_button = QtWidgets.QPushButton(self)
        cancel_button.setText('Cancel training')
        cancel_button.clicked.connect(self.cancel_action)
        self.compose_window.progress_dialog.setCancelButton(cancel_button)

        self.compose_window.progress_dialog.setWindowModality(Qt.WindowModal)
        self.compose_window.progress_dialog.setWindowTitle('Training')
        self.compose_window.progress_dialog.setAutoClose(False)

        self.close()
        self.compose_window.progress_dialog.show()

        weights_folder = Path(__file__).parent / 'data' / main.load_files_window.model_name_string / 'weights'
        store_folder = Path(__file__).parent / 'data' / main.load_files_window.model_name_string / 'store'
        weights_folder.mkdir(parents=True, exist_ok=True)
        store_folder.mkdir(parents=True, exist_ok=True)
        checkpoint1 = tensorflow.keras.callbacks.ModelCheckpoint(
            str(weights_folder / 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5'),
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
                                      )

        checkpoint2 = tensorflow.keras.callbacks.ModelCheckpoint(
            str(weights_folder / 'weights.h5'),
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
                                      )

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(
            monitor='loss',
            restore_best_weights=True,
            patience=self.spinbox_early_stop_patience.value()
                                       )

        progress_callback = ProgressCallback()

        callbacks_list = [
            checkpoint1,
            checkpoint2,
            early_stopping,
            progress_callback
                          ]
        main.load_files_window.model.save_weights(str(weights_folder / 'weights.h5'))

        _history = main.load_files_window.model.fit(
            main.load_files_window.network_data[0], main.load_files_window.network_data[1],
            epochs=epochs,
            batch_size=self.spinbox_batch_size.value(),
            validation_split=self.spinbox_validation_split.value(),
            callbacks=callbacks_list,
            shuffle=True
                                                    )
        with open(store_folder / 'lookups', 'wb') as lookups:
            pickle.dump([main.load_files_window.notes_lookups, main.load_files_window.durations_lookups], lookups)
        with open(store_folder / 'model_parameters', 'wb') as parameters:
            pickle.dump(main.load_files_window.model_parameters, parameters)

        cancel_button.setText('Close')
        self.compose_window.progress_dialog.setFocus()
        self.compose_window.progress_dialog.setValue(epochs)
        self.compose_window.progress_dialog.setLabelText('Training finished')

        if self.cancel_flag:
            main.load_files_window.confirmation_window.show()
            self.compose_window.progress_dialog.close()
            self.cancel_flag = False
        else:
            self.compose_window.show()


class ProgressCallback(tensorflow.keras.callbacks.Callback):
    """Callback to update the progress bar and the label during the training process.
    """
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        main.load_files_window.confirmation_window.compose_window.progress_dialog.setValue(1)
        text = f'Epoch: 0 / {self.params["epochs"]}'
        main.load_files_window.confirmation_window.compose_window.progress_dialog.setLabelText(text)

    def on_epoch_begin(self, epoch, logs=None):
        main.load_files_window.confirmation_window.compose_window.progress_dialog.setValue(epoch)
        text = f'Epoch: {int(epoch)+1} / {self.params["epochs"]}'
        main.load_files_window.confirmation_window.compose_window.progress_dialog.setLabelText(text)
        if main.load_files_window.confirmation_window.compose_window.progress_dialog.wasCanceled():
            self.model.stop_training = True


class ComposeWindow(QtWidgets.QDialog):
    """Compose Window. Shows options for composing midi files with the loaded model,
    to load another pretrained model or to improve the training for the current model.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Compose')
        self.horizontal_layout = QtWidgets.QHBoxLayout(self)
        self.horizontal_layout.setContentsMargins(10, 10, 10, 10)
        self.horizontal_layout.setSpacing(10)

        self.label_model_name = QtWidgets.QLabel(self)
        self.spinbox_notes_temp = QtWidgets.QDoubleSpinBox(self)
        self.spinbox_durations_temp = QtWidgets.QDoubleSpinBox(self)
        self.spinbox_max_extra_notes = QtWidgets.QSpinBox(self)
        self.spinbox_max_seq_length = QtWidgets.QSpinBox(self)
        self.spinbox_seq_length = QtWidgets.QSpinBox(self)
        self.text_entry_start_phrase_notes = QtWidgets.QLineEdit(self)
        self.text_entry_start_phrase_durations = QtWidgets.QLineEdit(self)

        self.make_layout()
        self.progress_dialog = QtWidgets.QProgressDialog('Training...', 'Cancel', 0, 100, self)
        
    def make_layout(self):
        """Create and arrange the layout for the parameters and the buttons on the right side.
        """
        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.setContentsMargins(10, 10, 10, 10)
        vertical_layout.setSpacing(10)
        self.horizontal_layout.addLayout(vertical_layout)

        self.spinbox_notes_temp.setMaximum(99)
        self.spinbox_notes_temp.setValue(0.5)
        self.spinbox_notes_temp.setSingleStep(0.25)
        self.spinbox_durations_temp.setMaximum(99)
        self.spinbox_durations_temp.setValue(0.5)
        self.spinbox_durations_temp.setSingleStep(0.25)
        self.spinbox_max_extra_notes.setMaximum(999)
        self.spinbox_max_extra_notes.setValue(50)
        self.spinbox_max_seq_length.setMaximum(999)
        self.spinbox_max_seq_length.setValue(32)
        self.spinbox_seq_length.setMaximum(999)
        self.spinbox_seq_length.setValue(32)
        
        button_load_pretrained_model = QtWidgets.QPushButton('Load pretrained Model')
        button_load_pretrained_model.clicked.connect(self.load_pretrained_model)
        button_improve_training = QtWidgets.QPushButton('Improve training for this model')
        button_improve_training.clicked.connect(self.reload_training_window)
        label_notes_temp = QtWidgets.QLabel()
        label_notes_temp.setText('Notes temperature:')
        label_durations_temp = QtWidgets.QLabel()
        label_durations_temp.setText('Durations temperature:')
        label_max_extra_notes = QtWidgets.QLabel()
        label_max_extra_notes.setText('Maximum extra notes:')
        label_max_seq_length = QtWidgets.QLabel()
        label_max_seq_length.setText('Maximum prediction sequence length:')
        label_seq_length = QtWidgets.QLabel()
        label_seq_length.setText('Start Phrase length (Remaining notes will be filled with \'START\':')
        label_start_phrase_notes = QtWidgets.QLabel()
        label_start_phrase_notes.setText('Starting phrase notes (e.g. F#3, C3, A2, ...)')
        label_start_phrase_durations = QtWidgets.QLabel()
        label_start_phrase_durations.setText('Starting phrase durations (e.g. 0.5, 0.75, 1, ...)')

        vertical_layout.addWidget(self.label_model_name)
        vertical_layout.addWidget(button_load_pretrained_model)
        vertical_layout.addWidget(button_improve_training)
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.addWidget(label_notes_temp, 0, 0)
        grid_layout.addWidget(self.spinbox_notes_temp, 0, 1)
        grid_layout.addWidget(label_durations_temp, 1, 0)
        grid_layout.addWidget(self.spinbox_durations_temp, 1, 1)
        grid_layout.addWidget(label_max_extra_notes, 2, 0)
        grid_layout.addWidget(self.spinbox_max_extra_notes, 2, 1)
        grid_layout.addWidget(label_max_seq_length, 3, 0)
        grid_layout.addWidget(self.spinbox_max_seq_length, 3, 1)
        grid_layout.addWidget(label_seq_length, 4, 0)
        grid_layout.addWidget(self.spinbox_seq_length, 4, 1)
        vertical_layout.addLayout(grid_layout)
        vertical_layout.addWidget(label_start_phrase_notes)
        vertical_layout.addWidget(self.text_entry_start_phrase_notes)
        vertical_layout.addWidget(label_start_phrase_durations)
        vertical_layout.addWidget(self.text_entry_start_phrase_durations)

        button_start_composing = QtWidgets.QPushButton('Start Composing')
        button_start_composing.clicked.connect(self.start_composing)
        vertical_layout.addWidget(button_start_composing)
        
    def reload_training_window(self):
        """Reloads training window.
        """
        main.load_files_window.confirmation_window.show()
        self.close()

    def load_pretrained_model(self):
        """Loads another pretrained model.
        """
        main.load_files_window.load_model()
        self.close()

    def start_composing(self):
        """Starts composing process. Shows messagebox when finished.
        """
        notes, durations = self.make_start_phrase()
        output = self.generate_notes(notes, durations)
        if output is not None:
            ComposeWindow.make_midi_file(output)
            message_box = QtWidgets.QMessageBox(self)
            message_box.setText(
                f'Finished! A midi file with a sequence of {len(output)} notes was saved in the subfolder "output".'
                                )
            message_box.show()

    def make_start_phrase(self):
        """Creates starting phrase by the input of the line edit.
        Returns tuple of notes, durations.
        """
        if self.text_entry_start_phrase_notes.text():
            notes = ['START'] + self.text_entry_start_phrase_notes.text().replace(' ','').split(',')
        else:
            notes = ['START']
        if self.text_entry_start_phrase_durations.text():
            durations = [0.] + [float(item) for item in self.text_entry_start_phrase_durations.text().split(',')]
        else:
            durations = [0.]
        seq_length = self.spinbox_seq_length.value()
        
        if seq_length is not None and seq_length > len(notes):
            notes = ['START'] * (seq_length - len(notes)) + notes
            durations = [0.] * (seq_length - len(durations)) + durations

        return notes, durations

    def generate_notes(self, notes, durations):
        """Takes the tuple (notes, durations) of the starting phrase as arguments.
        Returns predicted sequence list of [note, duration].
        """
        prediction_output = []
        notes_input_sequence = []
        durations_input_sequence = []
        
        max_extra_notes = self.spinbox_max_extra_notes.value()
        sequence_length = len(notes)
        for note_element, duration_element in zip(notes, durations):
            try:
                note_int = main.load_files_window.notes_lookups[0][note_element]
            except KeyError:
                message_box = QtWidgets.QMessageBox(self)
                message_box.setText(f'The note "{note_element}" is not in the lookup table')
                message_box.show()
                return
            try:
                duration_int = main.load_files_window.durations_lookups[0][duration_element]
            except KeyError:
                message_box = QtWidgets.QMessageBox(self)
                message_box.setText(f'The duration "{duration_element}" is not in the lookup table')
                message_box.show()
                return

            notes_input_sequence.append(note_int)
            durations_input_sequence.append(duration_int)
            prediction_output.append([note_element, duration_element])
            
            if note_element != 'START':
                try:
                    midi_note = note.Note(note_element)
                except music21.pitch.AccidentalException:
                    message_box = QtWidgets.QMessageBox(self)
                    message_box.setText(f'Chords are not supported in the starting phrase unfortunately')
                    message_box.show()
                    return
        
                new_note = np.zeros(128)
                new_note[midi_note.pitch.midi] = 1
        att_matrix = np.zeros(shape=(max_extra_notes+sequence_length, max_extra_notes))
        
        for note_index in range(max_extra_notes):
            prediction_input = [
                np.array([notes_input_sequence]),
                np.array([durations_input_sequence])
                                ]
        
            notes_prediction, durations_prediction = main.load_files_window.model.predict(prediction_input, verbose=0)
            if main.load_files_window.model_parameters:       
                att_prediction = main.load_files_window.model_with_att.predict(prediction_input, verbose=0)[0]
                att_matrix[
                (note_index-len(att_prediction)+sequence_length): (note_index+sequence_length), note_index
                           ] = att_prediction
            
            new_note = np.zeros(128)

            for idx, n_i in enumerate(notes_prediction[0]):
                try:
                    note_name = main.load_files_window.notes_lookups[1][idx]
                    midi_note = note.Note(note_name)
                    new_note[midi_note.pitch.midi] = n_i
                    
                except music21.pitch.AccidentalException:
                    pass
                except music21.pitch.PitchException:
                    pass
                except KeyError:
                    pass
                
            i1 = ComposeWindow.sample_with_temp(notes_prediction[0], self.spinbox_notes_temp.value())
            i2 = ComposeWindow.sample_with_temp(durations_prediction[0], self.spinbox_durations_temp.value())

            note_result = main.load_files_window.notes_lookups[1][i1]
            duration_result = main.load_files_window.durations_lookups[1][i2]
            
            prediction_output.append([note_result, duration_result])
        
            notes_input_sequence.append(i1)
            durations_input_sequence.append(i2)
            
            if len(notes_input_sequence) > self.spinbox_max_seq_length.value():
                notes_input_sequence = notes_input_sequence[1:]
                durations_input_sequence = durations_input_sequence[1:]

            if note_result == 'START':
                break

        return prediction_output

    @staticmethod
    def sample_with_temp(preds, temperature):
        """Choose temperature.
        """
        if temperature == 0:
            return np.argmax(preds)
        else:
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            return np.random.choice(len(preds), p=preds)

    @staticmethod
    def make_midi_file(sequence):
        """Creates a midi file for the given sequence.
        """
        output_folder = Path(__file__).parent / 'data' / main.load_files_window.model_name_string / 'output'
        output_folder.mkdir(parents=True, exist_ok=True)
        midi_stream = stream.Stream()
        
        for pattern in sequence:
            note_pattern, duration_pattern = pattern
            if '.' in note_pattern:
                notes_in_chord = note_pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    new_note.duration = duration.Duration(duration_pattern)
                    new_note.storedInstrument = instrument.Piano()
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                midi_stream.append(new_chord)
            elif note_pattern == 'rest':
                new_note = note.Rest()
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Piano()
                midi_stream.append(new_note)
            elif note_pattern != 'START':
                new_note = note.Note(note_pattern)
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Piano()
                midi_stream.append(new_note)

        midi_stream = midi_stream.chordify()
        file_name_string = f'output-{main.load_files_window.model_name_string}-{time.strftime("%H%M%S")}.mid'
        midi_stream.write('midi', fp=str(output_folder / file_name_string))


def main():
    app = QtWidgets.QApplication(sys.argv)
    main.load_files_window = LoadFilesWindow()
    main.load_files_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
