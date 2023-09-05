import pytest
import numpy as np
from nex5file.filedata import FileData
from nex5file.variables import (
    Variable,
    NexFileVarType,
    EventVariable,
    NeuronVariable,
    IntervalVariable,
    MarkerVariable,
    WaveformVariable,
    ContinuousVariable,
    PopulationVector,
)

from nex5file.headers import VariableHeader


def test_compare_events():
    ev = EventVariable(VariableHeader(Type=NexFileVarType.EVENT, Name="event"))
    ev1 = EventVariable(VariableHeader(Type=NexFileVarType.EVENT, Name="event"))
    assert ev1 == ev
    ev1.timestamps = np.array([2])
    assert ev1 != ev
    ev2 = EventVariable(VariableHeader(Type=NexFileVarType.EVENT, Name="event2"))
    assert ev2 != ev


def test_compare_neurons():
    nr = NeuronVariable(VariableHeader(Type=NexFileVarType.NEURON, Name="neuron"))
    nr1 = NeuronVariable(VariableHeader(Type=NexFileVarType.NEURON, Name="neuron"))
    assert nr1 == nr
    nr1.timestamps = np.array([2])
    assert nr1 != nr
    nr2 = NeuronVariable(VariableHeader(Type=NexFileVarType.NEURON, Name="neuron2"))
    assert nr2 != nr


def test_compare_interval_vars():
    var = IntervalVariable(VariableHeader(Type=NexFileVarType.INTERVAL, Name="int"))
    var1 = IntervalVariable(VariableHeader(Type=NexFileVarType.INTERVAL, Name="int"))
    assert var1 == var
    var1.interval_starts = np.array([0.5])
    var1.interval_ends = np.array([1.5])
    assert var1 != var
    var2 = IntervalVariable(VariableHeader(Type=NexFileVarType.INTERVAL, Name="int2"))
    assert var2 != var


def test_compare_marker_vars():
    var = MarkerVariable(VariableHeader(Type=NexFileVarType.MARKER, Name="int"))
    var1 = MarkerVariable(VariableHeader(Type=NexFileVarType.MARKER, Name="int"))
    assert var1 == var
    var1.timestamps = np.array([0.5])
    assert var1 != var
    var2 = MarkerVariable(VariableHeader(Type=NexFileVarType.MARKER, Name="int2"))
    assert var2 != var


def test_compare_continuous_vars():
    var = ContinuousVariable(VariableHeader(Type=NexFileVarType.CONTINUOUS, Name="int"))
    var1 = ContinuousVariable(VariableHeader(Type=NexFileVarType.CONTINUOUS, Name="int"))
    assert var1 == var
    var1.fragment_timestamps = np.array([0.5])
    assert var1 != var
    var2 = ContinuousVariable(VariableHeader(Type=NexFileVarType.CONTINUOUS, Name="int2"))
    assert var2 != var
