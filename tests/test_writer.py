"""
Tests in this module require NeuroExplorer to be installed and to be running. 
Also, the tests use some fixed file names.
"""
from nex5file.reader import Reader
from nex5file.writer import Writer
from nex5file.filedata import FileData
import numpy as np
import nex


def test_write_empty_data_nex():
    freq = 1000
    dataOriginal = FileData(freq)
    fpath = r"C:\Data\empty_test.nex"
    w = Writer()
    w.WriteNexFile(dataOriginal, fpath)
    r = Reader()
    data = r.ReadNexFile(fpath)
    assert data == dataOriginal

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    # if no data, nex will set end time to 1 and add StartStop and AllFile
    assert nex.GetDocEndTime(doc) == 1.0
    assert doc.NeuronNames() == []
    assert doc.EventNames() == ["StartStop"]
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == []
    assert doc.ContinuousNames() == []
    nex.CloseDocument(doc)


def test_write_empty_data_nex5():
    freq = 1000
    dataOriginal = FileData(freq)
    fpath = r"C:\Data\empty_test.nex5"
    w = Writer()
    w.WriteNexFile(dataOriginal, fpath)
    r = Reader()
    data = r.ReadNexFile(fpath)
    assert data == dataOriginal

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    # if no data, nex will set end time to 1 and add StartStop and AllFile
    # assert nex.GetDocEndTime(doc) == 1.0
    assert doc.NeuronNames() == []
    assert doc.EventNames() == ["StartStop"]
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == []
    assert doc.ContinuousNames() == []
    nex.CloseDocument(doc)


def test_write_timestamp_data_nex():
    freq = 100000
    data = FileData(freq)
    eventTs = [1, 2, 3.5]
    eName = "event name with spaces"
    neuronTs = [0.001, 2.54, 8.99]
    nName = "neuron 12345"
    data.AddEvent(eName, np.array(eventTs))
    data.AddNeuron(nName, np.array(neuronTs), wire=15, unit=2, xPosition=25.34, yPosition=35)

    fpath = r"C:\Data\ats_test.nex"
    w = Writer()
    w.WriteNexFile(data, fpath)
    r = Reader()
    data1 = r.ReadNexFile(fpath)

    assert data1 == data

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    assert nex.GetDocEndTime(doc) == 8.99
    assert doc.NeuronNames() == [nName]
    assert doc.EventNames() == [eName]
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == []
    assert doc.ContinuousNames() == []
    assert doc[eName].Timestamps() == eventTs
    assert doc[nName].Timestamps() == neuronTs

    nex.CloseDocument(doc)


def test_write_timestamp_data_nex5():
    freq = 100000
    data = FileData(freq)
    eventTs = [1, 2, 3.5]
    eName = "event name with spaces"
    neuronTs = [0.001, 2.54, 8.99]
    nName = "neuron 12345"
    data.AddEvent(eName, np.array(eventTs))
    data.AddNeuron(nName, np.array(neuronTs), wire=15, unit=2, xPosition=25.34, yPosition=35)

    fpath = r"C:\Data\ats_test.nex5"
    w = Writer()
    w.WriteNex5File(data, fpath)
    r = Reader()
    data1 = r.ReadNexFile(fpath)

    assert data1 == data

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    assert nex.GetDocEndTime(doc) == 8.99
    assert doc.NeuronNames() == [nName]
    assert doc.EventNames() == [eName]
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == []
    assert doc.ContinuousNames() == []
    assert doc[eName].Timestamps() == eventTs
    assert doc[nName].Timestamps() == neuronTs

    nex.CloseDocument(doc)


def test_write_cont_data_nex5():
    freq = 100000
    data = FileData(freq)
    cName = "cont name with spaces"
    v = [1.1, 2.2, 3.3]
    sr = 1000
    first = 0.5
    data.AddContVarWithFloatsSingleFragment(cName, sr, first, np.array(v))

    fpath = r"C:\Data\acont_test.nex5"
    w = Writer()
    w.WriteNex5File(data, fpath)
    r = Reader()
    data1 = r.ReadNexFile(fpath)

    assert data1 == data

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    assert nex.GetDocEndTime(doc) == 0.5 + (len(v) - 1) / sr
    assert doc.NeuronNames() == []
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == []
    assert doc.ContinuousNames() == [cName]
    assert doc[cName].SamplingRate() == sr
    assert doc[cName].ContinuousValues() == np.array(v).astype(np.float32).tolist()
    assert doc[cName].FragmentTimestamps() == [first]
    assert doc[cName].FragmentCounts() == [len(v)]
    nex.CloseDocument(doc)


def test_write_cont_data_nex():
    freq = 100000
    data = FileData(freq)
    cName = "cont name with spaces"
    v = [1, 20, 30]
    sr = 1000
    first = 0.5
    scale = 0.1
    offset = 0.02
    data.AddContSingleFragmentValuesInt16(cName, sr, first, v, scale, offset)

    fpath = r"C:\Data\acont_test.nex"
    w = Writer()
    w.WriteNex5File(data, fpath)
    r = Reader()
    data1 = r.ReadNexFile(fpath)

    assert data1 == data

    scaledValues = np.array(v) * scale + offset
    assert data1[cName].ContinuousValues().tolist() == scaledValues.tolist()

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    assert nex.GetDocEndTime(doc) == 0.5 + (len(v) - 1) / sr
    assert doc.NeuronNames() == []
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == []
    assert doc.ContinuousNames() == [cName]
    assert doc[cName].SamplingRate() == sr
    assert doc[cName].ContinuousValues() == scaledValues.tolist()
    assert doc[cName].FragmentTimestamps() == [first]
    assert doc[cName].FragmentCounts() == [len(v)]
    nex.CloseDocument(doc)


def test_write_wave_data_nex5():
    freq = 100000
    data = FileData(freq)
    wv = [[2, 3, 4, 1], [5, 6, 7, 2]]
    sr = 10000
    ts = [10, 20]
    wv = [[2, 3, 4, 1], [5, 6, 7, 2]]
    waveName = "ScriptGenerated"
    data.AddWaveVarWithFloats(waveName, sr, ts, wv)
    fpath = r"C:\Data\awave_test.nex5"
    w = Writer()
    w.WriteNex5File(data, fpath)
    r = Reader()
    data1 = r.ReadNexFile(fpath)
    assert data1["ScriptGenerated"].Timestamps().tolist() == ts
    assert data1["ScriptGenerated"].WaveformValues().tolist() == wv

    doc = nex.OpenDocument(fpath)
    assert nex.GetDocComment(doc) == ""
    assert nex.GetTimestampFrequency(doc) == freq
    assert nex.GetDocStartTime(doc) == 0
    assert nex.GetDocEndTime(doc) == ts[-1] + (len(wv[0]) - 1) / sr
    assert doc.NeuronNames() == []
    assert doc.IntervalNames() == ["AllFile"]
    assert doc.MarkerNames() == []
    assert doc.WaveNames() == [waveName]
    assert doc.ContinuousNames() == []
    assert doc[waveName].SamplingRate() == sr
    assert doc[waveName].WaveformValues() == np.array(wv).astype(np.float32).tolist()
    assert doc[waveName].Timestamps() == ts
    nex.CloseDocument(doc)
