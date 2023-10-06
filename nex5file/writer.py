from nex5file.filedata import (
    Variable,
    NeuronVariable,
    EventVariable,
    IntervalVariable,
    MarkerVariable,
    WaveformVariable,
    ContinuousVariable,
    PopulationVector,
)
from nex5file.filedata import FileData, NexFileVarType
from nex5file.headers import _ToString, FileHeader, DataFormat
from struct import pack
import os
import struct
import json
import numpy as np
from copy import deepcopy


class Writer:
    """
    .nex and .nex5 writer class

    This class provides methods for writing data to .nex and .nex5 files.
    """

    theFile: any = None
    timestamp_frequencyHz: float = 0
    tsAs64: int = 0

    def __init__(self):
        """
        Initialize a new Writer instance.
        """
        self.var_headers = []

    def WriteNexFile(self, data: FileData, filePath: str) -> None:
        """
        Write data to a .nex file.

        This method writes data from a FileData object to a .nex file located at the specified 'filePath'.

        If filePath ends with '.nex5', writes data using nex5 data format.

        Parameters:
            data (FileData): A FileData object containing the data to be written.
            filePath (str): The path to the .nex file to be created or updated.

        Raises:
            ValueError: If the maximum timestamp exceeds the 32-bit range, preventing it from being saved as a .nex file.

        Example:
            To write data to a .nex file:

        ::

            writer = Writer()
            writer.WriteNexFile(file_data, r"C:\\path\\to\\your\\file.nex")
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex5":
            self.WriteNex5File(data, filePath)
            return

        if len(data.variables) > 0 and data._NumberOfBytesInData() == 0:
            raise ValueError(
                "unable to save FileData object if all variables have no data. NeuroExplorer will reject .nex file with no data."
            )
        self.timestamp_frequencyHz = data.timestamp_frequencyHz
        maxTs = data._MaximumTimestamp()
        if round(maxTs * data.timestamp_frequencyHz) > pow(2, 31):
            raise ValueError(
                "unable to save as .nex file: max timestamp exceeds 32-bit range; you can save as .nex5 file instead"
            )
        self.theFile = open(filePath, "wb")
        fh = self._PrepareNexFileHeader(data, round(maxTs * data.timestamp_frequencyHz))
        fh.WriteToNexFile(self.theFile)
        self._WriteNexVarHeaders(data)
        self._WriteVariableData(data)

        self.theFile.close()

    def WriteNex5File(self, data: FileData, filePath: str) -> None:
        """
        Write data to a .nex5 file.

        This method writes data from a FileData object to a .nex5 file located at the specified 'filePath'.

        If filePath ends with '.nex', writes data using nex data format.

        Parameters:
            data (FileData): A FileData object containing the data to be written.
            filePath (str): The path to the .nex5 file to be created or updated.

        Example:
            To write data to a .nex5 file:

        ::

            writer = Writer()
            writer.WriteNex5File(file_data, r"C:path\\to\\your\\file.nex5")
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex":
            self.WriteNexFile(data, filePath)
            return

        if len(data.variables) > 0 and data._NumberOfBytesInData() == 0:
            raise ValueError(
                "unable to save FileData object if all variables have no data. NeuroExplorer will reject .nex file with no data."
            )

        self.timestamp_frequencyHz = data.timestamp_frequencyHz
        self.theFile = open(filePath, "wb")
        fh = self._PrepareNex5FileHeader(data)
        fh.WriteToNex5File(self.theFile)

        self._WriteNex5VarHeaders(data)
        self._WriteVariableData(data)
        self._WriteMetaData(data)
        self.theFile.close()

    def _WriteMetaData(self, data: FileData) -> None:
        """
        This method is used internally by the Writer class to prepare the prepare and write metadata to .nex5 file.
        """
        metaData = {}
        metaData["file"] = {}
        metaData["file"]["writerSoftware"] = {}
        metaData["file"]["writerSoftware"]["name"] = "nex5file"
        metaData["file"]["writerSoftware"]["version"] = "0.1.0"
        metaData["variables"] = []
        for v in data.variables:
            varMeta = {"name": v.header.Name}
            if v.header.Type == NexFileVarType.NEURON or v.header.Type == NexFileVarType.WAVEFORM:
                varMeta["unitNumber"] = v.header.Unit
                varMeta["probe"] = {}
                varMeta["probe"]["wireNumber"] = v.header.Wire
                varMeta["probe"]["position"] = {}
                varMeta["probe"]["position"]["x"] = v.header.XPos
                varMeta["probe"]["position"]["y"] = v.header.YPos
            metaData["variables"].append(varMeta)

        metaString = json.dumps(metaData).encode("utf-8")
        pos = self.theFile.tell()
        self.theFile.write(metaString)
        metaPosInHeader = 284
        self.theFile.seek(metaPosInHeader, 0)
        self.theFile.write(struct.pack("<Q", pos))

    def _PrepareNexFileHeader(self, data, maxTsTicks: int) -> FileHeader:
        """
        This method is used internally by the Writer class to prepare the header for writing a .nex file.
        """
        fh = FileHeader()
        fh.MagicNumber = 827868494
        fh.NexFileVersion = 106
        fh.Comment = data.comment
        fh.Frequency = data.timestamp_frequencyHz
        fh.BegTicks = int(round(data.beg_seconds * data.timestamp_frequencyHz))
        fh.EndTicks = maxTsTicks
        fh.NumVars = len(data.variables)
        return fh

    def _PrepareNex5FileHeader(self, data) -> FileHeader:
        """
        This method is used internally by the Writer class to prepare the header for writing a .nex5 file.
        """
        fh = FileHeader()
        fh.MagicNumber = 894977358
        fh.NexFileVersion = 501
        fh.Comment = data.comment
        fh.Frequency = data.timestamp_frequencyHz
        fh.NumVars = len(data.variables)
        maxTs = data._MaximumTimestamp()
        maxTsTicks = int(round(maxTs * fh.Frequency))
        self.tsAs64 = 0
        if maxTsTicks > pow(2, 31):
            self.tsAs64 = 1
            fh.NexFileVersion = 502

        fh.BegTicks = int(round(data.beg_seconds * data.timestamp_frequencyHz))
        fh.EndTicks = maxTsTicks
        return fh

    def _WriteNexVarHeaders(self, data: FileData) -> None:
        """
        This method is used internally by the Writer class to prepare .nex variable headers.
        """
        dataOffset = 544 + len(data.variables) * 208
        for v in data.variables:
            # CalcMarkerLength will also set MarkerDataType
            if isinstance(v, MarkerVariable):
                v._CalcMarkerLength()
            v.header.Count = v._CountForHeader()
            vh = deepcopy(v.header)
            vh.TsDataType = 0
            vh.Version = 102
            vh.ContDataType = 0
            vh.ContFragIndexType = 0
            if isinstance(v, ContinuousVariable) or isinstance(v, WaveformVariable):
                scaling = v._ContScaling(vh.ContDataType)
                vh.ADtoMV = scaling[0]
                vh.MVOffset = scaling[1]
            # overwrite MarkerDataType for .nex files
            vh.MarkerDataType = 0
            vh.DataOffset = dataOffset
            v.header_for_writing = vh
            dataOffset += v._BytesInData()
            vh.WriteToNexFile(self.theFile)

    def _WriteNex5VarHeaders(self, data: FileData) -> None:
        """
        This method is used internally by the Writer class to prepare .nex5 variable headers.
        """
        dataOffset = 356 + len(data.variables) * 244
        for v in data.variables:
            # CalcMarkerLength will also set MarkerDataType
            if isinstance(v, MarkerVariable):
                v._CalcMarkerLength()
            v.header.Count = v._CountForHeader()
            vh = deepcopy(v.header)
            vh.TsDataType = self.tsAs64
            vh.Version = 500
            # we support only 32-bit fragment indexes
            vh.ContFragIndexType = 0
            if isinstance(v, ContinuousVariable) or isinstance(v, WaveformVariable):
                scaling = v._ContScaling(vh.ContDataType)
                vh.ADtoMV = scaling[0]
                vh.MVOffset = scaling[1]
            vh.DataOffset = dataOffset
            v.header_for_writing = vh
            dataOffset += v._BytesInData()
            vh.WriteToNex5File(self.theFile)

    def _WriteVariableData(self, data: FileData) -> None:
        """
        This method is used internally by the Writer class to write .variable data.
        """
        WriteFunctions = {
            NexFileVarType.EVENT: self._WriteTimestamps,
            NexFileVarType.NEURON: self._WriteTimestamps,
            NexFileVarType.INTERVAL: self._WriteIntervals,
            NexFileVarType.MARKER: self._WriteMarkers,
            NexFileVarType.CONTINUOUS: self._WriteContinuousVariableData,
            NexFileVarType.WAVEFORM: self._WriteWaveformVariableData,
            NexFileVarType.POPULATION_VECTOR: self._WritePopVectorData,
        }
        for var in data.variables:
            WriteFunctions[var.header.Type](var)

    def _WritePopVectorData(self, var: PopulationVector) -> None:
        """
        This method is used internally by the Writer class.
        """
        var.weights.astype(np.float64).tofile(self.theFile)

    def _WriteTimestamps(self, var: Variable) -> None:
        """
        This method is used internally by the Writer class.
        """
        self._WriteTimestampsArray(var.header_for_writing.BytesInTimestamp(), var.Timestamps())

    def _WriteIntervals(self, var: IntervalVariable) -> None:
        """
        This method is used internally by the Writer class.
        """
        self._WriteTimestampsArray(var.header_for_writing.BytesInTimestamp(), var.interval_starts)
        self._WriteTimestampsArray(var.header_for_writing.BytesInTimestamp(), var.interval_ends)

    def _WriteTimestampsArray(self, bytesInTs: int, timestamps) -> None:
        """
        This method is used internally by the Writer class.
        """
        if bytesInTs == 4:
            np.round(np.array(timestamps) * self.timestamp_frequencyHz).astype(np.int32).tofile(self.theFile)
        else:
            np.round(np.array(timestamps) * self.timestamp_frequencyHz).astype(np.int64).tofile(self.theFile)

    def _WriteMarkers(self, var: MarkerVariable) -> None:
        """
        This method is used internally by the Writer class.
        """
        self._WriteTimestamps(var)
        for i, name in enumerate(var.marker_field_names):
            self._WriteField("64s", name)
            if var.header_for_writing.MarkerDataType == 1:
                self._WriteList(var.marker_fields[i], DataFormat.UINT32)
            else:
                for v in var.marker_fields[i]:
                    if isinstance(v, int):
                        sv = "{0:05d}".format(v) + "\x00"
                    else:
                        sv = v
                        while len(sv) < var.header_for_writing.MarkerLength:
                            sv += "\x00"
                    self.theFile.write(sv.encode("utf-8"))

    def _WriteContinuousVariableData(self, var: ContinuousVariable):
        """
        This method is used internally by the Writer class.
        """
        self._WriteTimestampsArray(var.header_for_writing.BytesInTimestamp(), var.fragment_timestamps)
        # we support only 32-bit fragment indexes in this package
        np.array(var.fragment_indexes).astype(np.uint32).tofile(self.theFile)
        if var.header_for_writing.ContDataType == 0:
            np.round(np.array(var.continuous_values) / var.header_for_writing.ADtoMV).astype(np.int16).tofile(
                self.theFile
            )
        else:
            np.array(var.continuous_values).astype(np.float32).tofile(self.theFile)

    def _WriteWaveformVariableData(self, var: WaveformVariable):
        """
        This method is used internally by the Writer class.
        """
        self._WriteTimestampsArray(var.header_for_writing.BytesInTimestamp(), var.timestamps)
        if var.header_for_writing.ContDataType == 0:
            np.round(np.array(var.waveform_values) / var.header_for_writing.ADtoMV).astype(np.int16).tofile(
                self.theFile
            )
        else:
            np.array(var.waveform_values).astype(np.float32).tofile(self.theFile)

    def _ConvertStringToBytesIfNeeded(self, stringOrBytes) -> bytes:
        """
        This method is used internally by the Writer class.
        """
        if isinstance(stringOrBytes, bytes):
            return stringOrBytes
        else:
            return stringOrBytes.encode("utf-8")

    def _WriteField(self, theFormat, theField) -> None:
        """
        This method is used internally by the Writer class.
        """
        if theFormat.endswith("s"):
            theField = self._ConvertStringToBytesIfNeeded(theField)
        self.theFile.write(struct.pack(theFormat, theField))

    def _WriteList(self, theList, valueType) -> None:
        """
        This method is used internally by the Writer class.
        """
        self.theFile.write(struct.pack(DataFormat.StructTypeFromDataType(valueType) * len(theList), *theList))
