from nex5file.variables import (
    NeuronVariable,
    EventVariable,
    IntervalVariable,
    MarkerVariable,
    WaveformVariable,
    ContinuousVariable,
    PopulationVector,
)
from nex5file.filedata import (
    FileData,
    NexFileVarType,
    MakeFileVar,
)

from nex5file.headers import _ToString, FileHeader, VariableHeader, DataFormat

# from typing import List
import os
import struct
import json
import numpy as np


class Reader:
    """
    .nex and .nex5 reader class.

    This class provides functionality for reading both .nex and .nex5 files and extracting their data.
    """

    file_header: FileHeader
    data: FileData

    def __init__(self):
        """
        Initialize a new Reader instance.
        """
        self.file_header = FileHeader()
        self.data = FileData()
        self.theFile = None

    def ReadNexFileVariables(self, filePath: str, varNames: list) -> FileData:
        """
        Read specified variables from .nex file and return a FileData object.

        Parameters:
            filePath (str): Path to the .nex file.
            varNames list(str): a list of variable names.

        Returns:
            FileData: A FileData object containing variable data from the file.

        Raises:
            ValueError: If the file format is invalid.

        Example:
            Read only continuous channels from .nex file

            ```python
            reader = Reader()
            data = reader.ReadNexHeadersOnly("example.nex")
            contNames = data.ContinuousNames()
            data_cont = reader.ReadNexFileVariables("example.nex", contNames)
            ```

        """

        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex5":
            return self.ReadNex5FileVariables(filePath)

        self.data = FileData()
        self.file_header = FileHeader()
        self.theFile = open(filePath, "rb")
        self._ReadNexFileHeader()
        self._ReadNexVariableHeaders(varNames)
        self._ReadNexOrNex5VariableData()

        self.theFile.close()
        return self.data

    def ReadNex5FileVariables(self, filePath: str, varNames: list) -> FileData:
        """
        Read specified variables from .nex5 file and return a FileData object.

        Parameters:
            filePath (str): Path to the .nex5 file.
            varNames list(str): a list of variable names.

        Returns:
            FileData: A FileData object containing variable data from the file.

        Raises:
            ValueError: If the file format is invalid.

        Example:
            Read only continuous channels from .nex file

            ```python
            reader = Reader()
            data = reader.ReadNex5HeadersOnly("example.nex5")
            contNames = data.ContinuousNames()
            data_cont = reader.ReadNex5FileVariables("example.nex5", contNames)
            ```

        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex":
            return self.ReadNexFileVariables(filePath)

        self.data = FileData()
        self.file_header = FileHeader()
        self.theFile = open(filePath, "rb")
        self._ReadNex5FileHeader()
        self._ReadNex5VariableHeaders(varNames)
        self._ReadNexOrNex5VariableData()

        self.theFile.close()
        return self.data

    def ReadNexHeadersOnly(self, filePath: str) -> FileData:
        """
        Read .nex file headers and return a FileData object with no data.
        The returned FileData object can be used to obtain the names of variables in the file.
        The names can then be used to load only specific variables
        from a .nex file using ReadNexFileVariables method.

        Parameters:
            filePath (str): Path to the .nex file.

        Returns:
            FileData: A FileData object containing variable headers from the file.

        Raises:
            ValueError: If the file format is invalid.

        Example:
            Read only continuous channels from .nex file

            ```python
            reader = Reader()
            data = reader.ReadNexHeadersOnly("example.nex5")
            contNames = data.ContinuousNames()
            data_cont = reader.ReadNexFileVariables("example.nex5", contNames)
            ```

        """

        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex5":
            return self.ReadNex5HeadersOnly(filePath)

        self.data = FileData()
        self.file_header = FileHeader()
        self.theFile = open(filePath, "rb")
        self._ReadNexFileHeader()
        self._ReadNexVariableHeaders()
        self.theFile.close()
        return self.data

    def ReadNex5HeadersOnly(self, filePath: str) -> FileData:
        """
        Read .nex5 file headers and return a FileData object with no data.
        The returned FileData object can be used to obtain the names of variables in the file.
        The names can then be used to load only specific variables
        from a .nex5 file using ReadNex5FileVariables method.

        Parameters:
            filePath (str): Path to the .nex5 file.

        Returns:
            FileData: A FileData object containing variable headers from the file.

        Raises:
            ValueError: If the file format is invalid.

        Example:
            ```python
            # read only continuous channels from .nex file
            reader = Reader()
            data = reader.ReadNex5HeadersOnly("example.nex5")
            contNames = data.ContinuousNames()
            data_cont = reader.ReadNex5FileVariables("example.nex5", contNames)
            ```

        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex":
            return self.ReadNexHeadersOnly(filePath)

        self.data = FileData()
        self.file_header = FileHeader()
        self.theFile = open(filePath, "rb")
        self._ReadNex5FileHeader()
        self._ReadNex5VariableHeaders()
        self.theFile.close()
        return self.data

    def ReadNex5File(self, filePath: str) -> FileData:
        """
        Read a .nex5 file and return its data.

        Parameters:
            filePath (str): Path to the .nex5 file.

        Returns:
            FileData: A FileData object containing the data from the file.

        Raises:
            ValueError: If the file format is invalid.

        Example:

            ```python
            reader = Reader()
            data = reader.ReadNex5File("example.nex5")
            ```
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex":
            return self.ReadNexFile(filePath)

        self.data = FileData()
        self.file_header = FileHeader()
        self.theFile = open(filePath, "rb")
        self._ReadNex5FileHeader()
        self._ReadNex5VariableHeaders()
        self._ReadNexOrNex5VariableData()
        self._ReadMetadata()
        self.theFile.close()
        return self.data

    def ReadNexFile(self, filePath: str) -> FileData:
        """
        Read a .nex file and return its data.

        Parameters:
            filePath (str): Path to the .nex file.

        Returns:
            FileData: A FileData object containing the data from the file.

        Raises:
            ValueError: If the file format is invalid.

        Example:

            ```python
            reader = Reader()
            data = reader.ReadNexFile("example.nex")
            ```
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == ".nex5":
            return self.ReadNex5File(filePath)

        self.data = FileData()
        self.file_header = FileHeader()
        self.theFile = open(filePath, "rb")
        self._ReadNexFileHeader()
        self._ReadNexVariableHeaders()
        self._ReadNexOrNex5VariableData()
        self.theFile.close()
        return self.data

    def _ReadNex5VariableHeaders(self, varNames: list = []):
        """
        Read .nex5 variable headers and append them to the data object.

        Raises:
            ValueError: If the variable header contains invalid information.

        This method is intended to be used internally by the Reader class.
        """
        for varNum in range(self.file_header.NumVars):
            vh = self._ReadNex5VarHeader()
            if len(varNames) > 0:
                if not vh.Name in varNames:
                    continue
            self._VerifyVariableHeader(vh)
            self.data.variables.append(MakeFileVar[vh.Type](vh))

    def _ReadNexVariableHeaders(self, varNames: list = []):
        """
        Read .nex variable headers and append them to the data object.

        Raises:
            ValueError: If the variable header contains invalid information.

        This method is intended to be used internally by the Reader class.
        """
        vh = VariableHeader()
        for varNum in range(self.file_header.NumVars):
            vh = self._ReadNexVarHeader()
            if len(varNames) > 0:
                if not vh.Name in varNames:
                    continue
            self._VerifyVariableHeader(vh)
            self.data.variables.append(MakeFileVar[vh.Type](vh))

    def _VerifyVariableHeader(self, header: VariableHeader):
        """
        Verify the validity of a variable header.

        Parameters:
            header (VariableHeader): The variable header to be verified.

        Raises:
            ValueError: If the variable header contains invalid information.

        This method is intended to be used internally by the Reader class.
        """
        if header.Type == NexFileVarType.WAVEFORM and header.NPointsWave <= 0:
            raise ValueError("invalid waveform header: NPointsWave is not positive\n" + str(header))
        if header.Type == NexFileVarType.WAVEFORM and header.SamplingRate <= 0:
            raise ValueError("invalid waveform header: SamplingRate is not positive\n" + str(header))
        if header.Type == NexFileVarType.CONTINUOUS and header.SamplingRate <= 0:
            raise ValueError("invalid continuous header: SamplingRate is not positive\n" + str(header))

    def _ReadNexOrNex5VariableData(self):
        """
        Read variable data from .nex or .nex5 file and populate the data object.

        Raises:
            ValueError: If the variable data is invalid or cannot be read.

        This method is intended to be used internally by the Reader class.
        """
        for var in self.data.variables:
            self.theFile.seek(var.header.DataOffset)
            if isinstance(var, EventVariable) or isinstance(var, NeuronVariable):
                var.timestamps = self._ReadTimestamps(var.header.GetTimestampDataFormat(), var.header.Count)
            elif isinstance(var, IntervalVariable):
                var.interval_starts = self._ReadTimestamps(var.header.GetTimestampDataFormat(), var.header.Count)
                var.interval_ends = self._ReadTimestamps(var.header.GetTimestampDataFormat(), var.header.Count)
            elif isinstance(var, ContinuousVariable):
                self._ReadContinuousData(var)
            elif isinstance(var, WaveformVariable):
                self._ReadWaveformData(var)
            elif isinstance(var, MarkerVariable):
                self._ReadMarkerData(var)
            elif isinstance(var, PopulationVector):
                var.weights = np.fromfile(self.theFile, np.float64, var.header.Count)

    def _ReadMarkerData(self, var: MarkerVariable):
        """
        Read marker data for a MarkerVariable and populate it.

        Parameters:
            var (MarkerVariable): The MarkerVariable object to which the marker data will be added.

        Raises:
            ValueError: If the marker data cannot be read.

        This method is intended to be used internally by the Reader class.
        """
        if var.header.Count == 0:
            return
        var.timestamps = self._ReadTimestamps(var.header.GetTimestampDataFormat(), var.header.Count)
        for fieldNumber in range(var.header.NMarkers):
            name = _ToString(self.theFile.read(64), True).strip()
            var.marker_field_names.append(name)
            if var.header.MarkerDataType == 0:
                length = var.header.MarkerLength
                markers = [_ToString(self.theFile.read(length), True) for m in range(var.header.Count)]
            else:
                markers = self._ReadAndScaleValues(DataFormat.UINT32, var.header.Count)
            var.marker_fields.append(markers)
        var._IfNumberStringsStoreAsNumbers()

    def _ReadWaveformData(self, var: WaveformVariable):
        """
        Read and process waveform data from the file.

        Parameters:
            var (WaveformVariable): The WaveformVariable instance to which the data will be assigned.

        Raises:
            ValueError: If unable to read all required values.

        This method is intended to be used internally by the Reader class.
        """
        if var.header.Count == 0:
            return
        var.timestamps = self._ReadTimestamps(var.header.GetTimestampDataFormat(), var.header.Count)
        contValueType, rawToMv, offset = var.header.GetContDataPars()
        wf = self._ReadAndScaleValues(contValueType, var.header.Count * var.header.NPointsWave, rawToMv)
        var.waveform_values = wf.reshape(var.header.Count, var.header.NPointsWave)
        if offset != 0:
            var.waveform_values += offset
        var._HashContValues()

    def _ReadContinuousData(self, var: ContinuousVariable):
        """
        Read continuous data for a ContinuousVariable and populate it.

        Parameters:
            var (ContinuousVariable): The ContinuousVariable object to which the continuous data will be added.

        Raises:
            ValueError: If the continuous data cannot be read or is invalid.

        This method is intended to be used internally by the Reader class.
        """
        if var.header.Count == 0:
            return
        var.fragment_timestamps = self._ReadTimestamps(var.header.GetTimestampDataFormat(), var.header.Count)
        var.fragment_indexes = np.fromfile(
            self.theFile, DataFormat.GetNumPyType(var.header.GetFragmentIndexDataFormat()), var.header.Count
        )

        contValueType, rawToMv, offset = var.header.GetContDataPars()
        var.continuous_values = self._ReadAndScaleValues(contValueType, var.header.NPointsWave, rawToMv)
        if offset != 0:
            var.continuous_values += offset

        var._CalculateFragmentCountsFromIndexes()
        var._HashContValues()

    def _ReadTimestamps(self, tsDataFormat: DataFormat, count: int) -> list:
        """
        Read and scale timestamps (from ticks to seconds) from the file.

        Parameters:
            tsDataFormat (DataFormat): The data format of the timestamps.
            count (int): The number of timestamps to read.

        Returns:
            List[float]: An array of scaled timestamps.

        This method is intended to be used internally by the Reader class.
        """
        return self._ReadAndScaleValues(tsDataFormat, count, self.data.timestamp_frequencyHz, True)

    def _ReadAndScaleValues(self, valueType: DataFormat, count: int, scale=1.0, divide=False):
        """
        Read and scale values from the file.

        Parameters:
            valueType (DataFormat): The data format of the values.
            count (int): The number of values to read.
            scale (float, optional): The scaling factor to apply to the values. Defaults to 1.0.
            divide (bool, optional): If True, divide values by the scale; if False, multiply values by the scale.
                Defaults to False.

        Returns:
            numpy array: An array of scaled values.

        Raises:
            ValueError: If the values cannot be read or if there's an issue with the data.

        This method is intended to be used internally by the Reader class.
        """
        numpyType = DataFormat.GetNumPyType(valueType)
        values = np.fromfile(self.theFile, numpyType, count)
        if len(values) != count:
            raise ValueError("unable to read all values")
        if scale == 1.0:
            return values
        if divide:
            return values / scale
        else:
            return values * scale

    def _ReadNex5VarHeader(self) -> VariableHeader:
        """
        Read a variable header from a .nex5 file.

        Returns:
            VariableHeader: The variable header read from the file.

        This method is intended to be used internally by the Reader class.
        """
        vh = VariableHeader()
        vh.ReadFromNex5File(self.theFile)
        return vh

    def _ReadNexVarHeader(self) -> VariableHeader:
        """
        Read a variable header from a .nex file.

        Returns:
            VariableHeader: The variable header read from the file.

        This method is intended to be used internally by the Reader class.
        """
        vh = VariableHeader()
        vh.ReadFromNexFile(self.theFile)
        return vh

    def _ReadNex5FileHeader(self):
        """
        Read the file header from a .nex5 file and populate the data object.

        This method is intended to be used internally by the Reader class.
        """
        self.file_header.ReadFromNex5File(self.theFile)
        self.data.comment = self.file_header.Comment
        self.data.timestamp_frequencyHz = self.file_header.Frequency
        self.data.beg_seconds = self.file_header.BegSeconds
        self.data.end_seconds = self.file_header.EndSeconds

    def _ReadNexFileHeader(self):
        """
        Read the file header from a .nex file and populate the data object.

        This method is intended to be used internally by the Reader class.
        """
        self.file_header.ReadFromNexFile(self.theFile)
        self.data.comment = self.file_header.Comment
        self.data.timestamp_frequencyHz = self.file_header.Frequency
        self.data.beg_seconds = self.file_header.BegSeconds
        self.data.end_seconds = self.file_header.EndSeconds

    def _ReadMetadata(self):
        """
        Read and process metadata from the file.

        Raises:
            Warning: If there is an issue with parsing the metadata as JSON, a warning is printed.

        This method is intended to be used internally by the Reader class.
        """
        metaOffset = self.file_header.MetaOffset
        if metaOffset <= 0:
            return
        self.theFile.seek(0, os.SEEK_END)
        size = self.theFile.tell()
        if metaOffset >= size:
            return
        self.theFile.seek(metaOffset)
        metaString = self.theFile.read(size - metaOffset).decode("utf-8").strip("\x00")
        metaString = metaString.strip()
        try:
            self.data.metadata = json.loads(metaString)
            if "variables" in self.data.metadata:
                allVarMeta = self.data.metadata["variables"]
                for varMeta in allVarMeta:
                    name = varMeta["name"]
                    for var in self.data.variables:
                        if var.header.Name == name:
                            var.metadata = varMeta
                            if isinstance(var, NeuronVariable):
                                var._AssignFromVarMeta()

        except Exception as error:
            print("WARNING: Invalid file metadata: " + repr(error))
