"""
Tests in this module require NeuroExplorer to be installed and to be running. 
Also, the tests use some fixed file names.
"""
import pytest
from nex5file.reader import Reader
import numpy as np
import nex


@pytest.fixture(scope="module", params=[r"C:\Data\File1.nex5", r"C:\Data\File1.nex"])
def load(request):
    fpath = request.param
    doc = nex.OpenDocument(fpath)
    r = Reader()
    data = r.ReadNexFile(fpath)
    return [doc, data]


def test_doc_prop_equal(load):
    doc, data = load
    assert nex.GetDocComment(doc) == data.GetDocComment()
    assert nex.GetTimestampFrequency(doc) == data.GetTimestampFrequency()
    assert nex.GetDocStartTime(doc) == data.GetDocStartTime()
    assert nex.GetDocEndTime(doc) == data.GetDocEndTime()


def test_var_names_equal(load):
    doc, data = load
    assert doc.NeuronNames() == data.NeuronNames()
    assert doc.EventNames() == data.EventNames()
    assert doc.IntervalNames() == data.IntervalNames()
    assert doc.MarkerNames() == data.MarkerNames()
    assert doc.WaveNames() == data.WaveNames()
    assert doc.ContinuousNames() == data.ContinuousNames()


def test_ts_equal(load):
    doc, data = load
    for name in doc.NeuronNames():
        assert doc[name].Timestamps() == data[name].Timestamps().tolist()
    for name in doc.EventNames():
        assert doc[name].Timestamps() == data[name].Timestamps().tolist()
    for name in doc.MarkerNames():
        assert doc[name].Timestamps() == data[name].Timestamps().tolist()
    for name in doc.WaveNames():
        assert doc[name].Timestamps() == data[name].Timestamps().tolist()


def test_intervals_equal(load):
    doc, data = load
    for name in doc.IntervalNames():
        assert doc[name].Intervals() == data[name].Intervals()


def test_markers_equal(load):
    doc, data = load
    for name in doc.MarkerNames():
        assert doc[name].MarkerFieldNames() == data[name].MarkerFieldNames()
        assert doc[name].Markers() == data[name].Markers()


def test_waves_equal(load):
    doc, data = load
    for name in doc.WaveNames():
        assert doc[name].SamplingRate() == data[name].SamplingRate()
        assert doc[name].NumPointsInWave() == data[name].NumPointsInWave()
        assert doc[name].WaveformValues() == data[name].WaveformValues().tolist()
        assert doc[name].PreThresholdTime() == data[name].PreThresholdTime()


def test_cont_equal(load):
    doc, data = load
    for name in doc.ContinuousNames():
        assert doc[name].FragmentCounts() == data[name].FragmentCounts().tolist()
        assert doc[name].SamplingRate() == data[name].SamplingRate()
        assert np.array_equal(np.array(doc[name].ContinuousValues()), data[name].ContinuousValues())


def test_meta_equal(load):
    doc, data = load
    for name in doc.NeuronNames():
        assert doc[name].Metadata() == data[name].Metadata()
