# nex5file
Python package to read, write and edit data stored in NeuroExplorer .nex and .nex5 files

## Getting Started

### Install nex5file

Use pip to install nex5file
```
pip install nex5file
```

### Read .nex and .nex5 Files

`ReadNexFile` method of `nex5file.reader.Reader` class reads and parses contents of a .nex or a .nex5 file and returns an instance of `nex5file.filedata.FileData` object. This method reads all the data in the file.

```python
from nex5file.reader import Reader
from nex5file.filedata import FileData
nexFilePath = r"C:\path\to\mydata.nex"
reader = Reader()
data = reader.ReadNexFile(nexFilePath)
```

If you need to read only some channels from a .nex or .nex5 file, use `ReadNexFileVariables` method:

```python
# read only two continuous channels: cont1 and cont2
from nex5file.reader import Reader
from nex5file.filedata import FileData
nexFilePath = r"C:\path\to\mydata.nex"
reader = Reader()
data = reader.ReadNexFileVariables(nexFilePath, ['cont1', 'cont2'])
```

To retrieve channel names from a file, use `ReadNexHeadersOnly` method. Here is the code to read only continuous channels:

```python
# read only all continuous channels
from nex5file.reader import Reader
from nex5file.filedata import FileData
reader = Reader()
data = reader.ReadNexHeadersOnly(r"C:\path\to\mydata.nex5")
contNames = data.ContinuousNames()
data_cont = reader.ReadNexFileVariables(r"C:\path\to\mydata.nex5", contNames)
```

### Access Data in a FileData Object

Retrieving information fromm `FileData` object is similar to retrieving values using `nex` package. The difference is that `nex` package requires `NeuroExplorer` to be installed and running, while `nex5file` package if pure Python.

The syntax for accessing data is similar to `nex` package syntax. Many method names in `nex5file` package are the same as in `nex` package.

Here is a script to get continuous channels information using `nex`:

```python
import nex
doc = nex.OpenDocument(nexFilePath)
# print continuous channel name, sampling rates and continuous values
for name in doc.ContinuousNames():
    rate = doc[name].SamplingRate()
    values = doc[name].ContinuousValues()
    print(name, rate, values)
```

Here is the same functionality using `nex5file` package:

```python
from nex5file.reader import Reader
from nex5file.filedata import FileData

reader = Reader()
data = reader.ReadNexFile(nexFilePath)
# print continuous channel name, sampling rates and continuous values
for name in data.ContinuousNames():
    rate = data[name].SamplingRate()
    values = doc[name].ContinuousValues()
    print(name, rate, values)
```

### Modify Data in a FileData Object

You can use the following `FileData` methods to modify data:

- `DeleteVariable`
- `AddEvent`
- `AddNeuron`
- `AddIntervalAsPairsStartEnd`
- `AddMarker`
- `AddContVarWithFloatsSingleFragment`
- `AddContSingleFragmentValuesInt16`
- `AddContVarWithFloatsAllTimestamps`
- `AddWaveVarWithFloats`

### Write .nex and .nex5 Files

Use `WriteNexFile` method of `nex5file.writer.Writed` class

```python
from nex5file.writer import Writer
from nex5file.filedata import FileData

freq = 100000
data = FileData(freq)
eventTs = [1, 2, 3.5]
eName = "event 001"
neuronTs = [0.001, 2.54, 8.99]
nName = "neuron 002"
data.AddEvent(eName, np.array(eventTs))

nexFilePath = r"C:\Data\example.nex"
writer = Writer()
writer.WriteNexFile(data, nexFilePath)
```
