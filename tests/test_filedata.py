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


# Create a fixture to initialize a sample FileData instance for testing
@pytest.fixture
def sample_file_data():
    return FileData(10000, "Sample File Data")


# Test cases for the FileData class
def test_init_timestamp_frequency():
    with pytest.raises(ValueError):
        FileData(0)


def test_getitem_existing_variable(sample_file_data):
    event_var = EventVariable(VariableHeader(Type=NexFileVarType.EVENT, Name="EventVar"))
    sample_file_data.variables.append(event_var)
    retrieved_var = sample_file_data["EventVar"]
    assert retrieved_var == event_var


def test_getitem_nonexistent_variable(sample_file_data):
    with pytest.raises(ValueError, match='unable to find variable "Var2" in file data'):
        sample_file_data["Var2"]


def test_delete_variable(sample_file_data):
    event_var = EventVariable(VariableHeader(Type=NexFileVarType.EVENT, Name="event_var"))
    sample_file_data.variables.append(event_var)
    sample_file_data.DeleteVariable("event_var")
    assert len(sample_file_data.variables) == 0


def test_get_doc_comment(sample_file_data):
    assert sample_file_data.GetDocComment() == "Sample File Data"


def test_get_timestamp_frequency(sample_file_data):
    assert sample_file_data.GetTimestampFrequency() == 10000


def test_get_doc_start_time(sample_file_data):
    assert sample_file_data.GetDocStartTime() == 0


def test_get_doc_end_time(sample_file_data):
    assert sample_file_data.GetDocEndTime() == 0


def test_neuron_names(sample_file_data):
    neuron_var = NeuronVariable(VariableHeader(Type=NexFileVarType.NEURON, Name="NeuronVar"))
    sample_file_data.variables.append(neuron_var)
    assert sample_file_data.NeuronNames() == ["NeuronVar"]


def test_event_names(sample_file_data):
    event_var = EventVariable(VariableHeader(Type=NexFileVarType.EVENT, Name="EventVar"))
    sample_file_data.variables.append(event_var)
    assert sample_file_data.EventNames() == ["EventVar"]


def test_interval_names(sample_file_data):
    interval_var = IntervalVariable(VariableHeader(Type=NexFileVarType.INTERVAL, Name="IntervalVar"))
    sample_file_data.variables.append(interval_var)
    assert sample_file_data.IntervalNames() == ["IntervalVar"]


def test_wave_names(sample_file_data):
    wave_var = WaveformVariable(VariableHeader(Type=NexFileVarType.WAVEFORM, Name="WaveVar"))
    sample_file_data.variables.append(wave_var)
    assert sample_file_data.WaveNames() == ["WaveVar"]


def test_marker_names(sample_file_data):
    marker_var = MarkerVariable(VariableHeader(Type=NexFileVarType.MARKER, Name="MarkerVar"))
    sample_file_data.variables.append(marker_var)
    assert sample_file_data.MarkerNames() == ["MarkerVar"]


def test_continuous_names(sample_file_data):
    continuous_var = ContinuousVariable(VariableHeader(Type=NexFileVarType.CONTINUOUS, Name="ContVar"))
    sample_file_data.variables.append(continuous_var)
    assert sample_file_data.ContinuousNames() == ["ContVar"]


def test_pop_vector_names(sample_file_data):
    pop_vector_var = PopulationVector(VariableHeader(Type=NexFileVarType.POPULATION_VECTOR, Name="PopVectorVar"))
    sample_file_data.variables.append(pop_vector_var)
    assert sample_file_data.PopVectorNames() == ["PopVectorVar"]


def test_maximum_timestamp(sample_file_data):
    neuron_var1 = NeuronVariable(VariableHeader(Type=NexFileVarType.NEURON, Name="NeuronVar1"))
    neuron_var1.timestamps = np.array([1.0, 2.0, 3.0])

    neuron_var2 = NeuronVariable(VariableHeader(Type=NexFileVarType.NEURON, Name="NeuronVar2"))
    neuron_var2.timestamps = np.array([4.0, 5.0, 6.0])

    sample_file_data.variables.extend([neuron_var1, neuron_var2])
    assert sample_file_data.MaximumTimestamp() == 6.0


def test_add_event(sample_file_data):
    sample_file_data.AddEvent("new event", [1.2, 3.3, 4.4])
    assert sample_file_data.EventNames() == ["new event"]
    assert sample_file_data["new event"].Timestamps().tolist() == [1.2, 3.3, 4.4]


def test_add_neuron(sample_file_data):
    sample_file_data.AddNeuron("new neuron", [0.04, 1.2, 1.7])
    assert sample_file_data.NeuronNames() == ["new neuron"]
    assert sample_file_data["new neuron"].Timestamps().tolist() == [0.04, 1.2, 1.7]


def test_add_interval(sample_file_data):
    intervals = [(0.0, 1.0), (1.5, 2.0)]
    sample_file_data.AddIntervalAsPairsStartEnd("IntervalVariable", intervals)
    assert sample_file_data.IntervalNames() == ["IntervalVariable"]
    assert sample_file_data["IntervalVariable"].Intervals() == [[0, 1.5], [1, 2]]


def test_add_cont_single_fragment(sample_file_data):
    sample_file_data.AddContVarWithFloatsSingleFragment("ContVariable", 1000.0, 0.05, [1.0, 2.0, 3.7])
    assert sample_file_data.ContinuousNames() == ["ContVariable"]
    assert sample_file_data["ContVariable"].ContinuousValues().tolist() == [1.0, 2.0, 3.7]
    assert sample_file_data["ContVariable"].FragmentTimestamps().tolist() == [0.05]
    assert sample_file_data["ContVariable"].FragmentCounts().tolist() == [3]


def test_add_cont_single_fragment_values16(sample_file_data):
    sample_file_data.AddContSingleFragmentValuesInt16("ContVariable", 1000.0, 0.03, [5, 3, 4], 0.015, 0.2)
    assert sample_file_data.ContinuousNames() == ["ContVariable"]
    assert sample_file_data["ContVariable"].ContinuousValues().tolist() == (np.array([5, 3, 4]) * 0.015 + 0.2).tolist()
    assert sample_file_data["ContVariable"].FragmentTimestamps().tolist() == [0.03]
    assert sample_file_data["ContVariable"].FragmentCounts().tolist() == [3]


def test_add_cont_all_timestamps(sample_file_data):
    sample_file_data.AddContVarWithFloatsAllTimestamps(
        "ContVariable", 1000.0, [0.003, 0.004, 0.005, 0.1, 0.101], [1.0, 2.0, 3.7, 9, 5]
    )
    assert sample_file_data.ContinuousNames() == ["ContVariable"]
    assert sample_file_data["ContVariable"].ContinuousValues().tolist() == [1.0, 2.0, 3.7, 9, 5]
    assert sample_file_data["ContVariable"].FragmentTimestamps().tolist() == [0.003, 0.1]
    assert sample_file_data["ContVariable"].FragmentCounts().tolist() == [3, 2]
