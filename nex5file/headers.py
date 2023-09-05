from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from struct import unpack, calcsize, pack


class DataFormat:
    """
    Constants and utility functions for various binary data types.
    """

    INT16 = 0
    UINT16 = 1
    INT32 = 2
    UINT32 = 3
    INT64 = 4
    UINT64 = 5
    FLOAT32 = 6
    FLOAT64 = 7

    @staticmethod
    def NumBytesPerItem(dataType) -> int:
        """
        Get the number of bytes per item for a given data type.

        Parameters:
            dataType : int The data type constant.

        Returns:
            int : The number of bytes per item for the specified data type.
        """
        numBytes = [2, 2, 4, 4, 8, 8, 4, 8]
        return numBytes[dataType]

    @staticmethod
    def StructTypeFromDataType(dataType) -> str:
        """
        Get the struct format string for a given data type.

        Parameters:
            dataType : int The data type constant.

        Returns:
            str : The struct format string for the specified data type.
        """
        formats = ["h", "H", "i", "I", "q", "Q", "f", "d"]
        return formats[dataType]

    @staticmethod
    def GetNumPyType(valueType: int):
        """
        Get the corresponding NumPy data type for a given value type.

        Parameters:
            valueType : int The value type constant.

        Returns:
            numpy.dtype : The NumPy data type corresponding to the specified value type.
        """
        numpyTypes = [
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float32,
            np.float64,
        ]
        return numpyTypes[valueType]


def _ReadStructField(theFile, format) -> any:
    """
    Read a binary field from a file using a specified format.

    Parameters:
        theFile : binary file-like object The file from which to read the binary field.
        format : str The struct format string specifying the binary format of the field.

    Returns:
        any : The binary field read from the file.

    """
    return unpack(format, theFile.read(calcsize(format)))[0]


def _ConvertStringToBytesIfNeeded(stringOrBytes) -> bytes:
    """
    Convert a string to bytes if it's not already in bytes format.

    Parameters:
        stringOrBytes : str or bytes The input string or bytes to convert.

    Returns:
        bytes : The input converted to bytes, or unchanged if it's already in bytes format.

    """
    if isinstance(stringOrBytes, bytes):
        return stringOrBytes
    else:
        return stringOrBytes.encode("utf-8")


def _WriteField(theFile, theFormat: str, theField) -> None:
    """
    Write a field to a file using a specified format.

    Parameters:
        theFile : binary file-like object The file to which to write the field.
        theFormat : str The struct format string specifying the binary format of the field.
        theField : any The field to write to the file.

    """
    if theFormat.endswith("s"):
        theField = _ConvertStringToBytesIfNeeded(theField)
    theFile.write(pack(theFormat, theField))


def _ToString(theBytes, discardAfterFirstZero=False) -> str:
    """
    Convert a byte array read from a file to a string.

    Some .nex file writers may write garbage after the zero-terminated string.
    We need to discard this garbage before converting bytes to a string.

    Parameters:
        theBytes : bytes The byte array read from the file.
        discardAfterFirstZero : bool Whether to discard data after the first zero byte (default is False).

    Returns:
        str : The decoded string.

    """
    if discardAfterFirstZero:
        theBytesBeforeZero = theBytes.split(b"\0", 1)[0]
        try:
            return theBytesBeforeZero.decode("ascii")
        except:
            return theBytesBeforeZero.decode("ascii", errors="replace")

    try:
        str_ = theBytes.decode("utf-8").strip("\x00")
        return str_
    except:
        # try to discard garbage after zero and then decode
        theBytesBeforeZero = theBytes.split(b"\0", 1)[0]
        try:
            return theBytesBeforeZero.decode("utf-8").strip("\x00")
        except:
            return theBytesBeforeZero.decode("utf-8", errors="replace").strip("\x00")


@dataclass
class FileHeader:
    """
    Represents the header of a .nex or .nex5 file.
    """

    MagicNumber: int = 0
    NexFileVersion: int = 0
    Comment: str = ""
    Frequency: float = 0
    BegTicks: int = 0
    EndTicks: int = 0
    NumVars: int = 0
    Padding: str = ""
    MetaOffset: int = 0
    BegSeconds: float = 0
    EndSeconds: float = 0
    NexFileMagicNumber: int = 827868494
    NexFileSpecs: dict = field(default_factory=dict, repr=False)
    Nex5FileMagicNumber: int = 894977358
    Nex5FileSpecs: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """
        Initialize header specifications for .nex and .nex5 files.
        """
        self.NexFileSpecs = {
            "MagicNumber": "<i",
            "NexFileVersion": "<i",
            "Comment": "256s",
            "Frequency": "<d",
            "BegTicks": "<i",
            "EndTicks": "<i",
            "NumVars": "<i",
            "Padding": "260s",
        }

        self.Nex5FileSpecs = {
            "MagicNumber": "<i",
            "NexFileVersion": "<i",
            "Comment": "256s",
            "Frequency": "<d",
            "BegTicks": "<q",
            "NumVars": "<i",
            "MetaOffset": "<Q",
            "EndTicks": "<q",
            "Padding": "56s",
        }

    def WriteToNexFile(self, theFile) -> None:
        """
        Write the header to a .nex file.
        """
        for key in self.NexFileSpecs:
            value = getattr(self, key)
            _WriteField(theFile, self.NexFileSpecs[key], value)

    def WriteToNex5File(self, theFile) -> None:
        """
        Write the header to a .nex5 file.
        """
        for key in self.Nex5FileSpecs:
            value = getattr(self, key)
            _WriteField(theFile, self.Nex5FileSpecs[key], value)

    def ReadFromNexFile(self, theFile) -> None:
        """
        Read the header from a .nex file.
        """
        theDict = {}
        for key in self.NexFileSpecs:
            theDict[key] = _ReadStructField(theFile, self.NexFileSpecs[key])
        if theDict["MagicNumber"] != self.NexFileMagicNumber:
            # do not continue if incorrect magic number
            raise ValueError("invalid .nex file header")

        theDict["Comment"] = _ToString(theDict["Comment"])
        del theDict["Padding"]
        for key in theDict:
            setattr(self, key, theDict[key])
        self.BegSeconds = self.BegTicks / self.Frequency
        self.EndSeconds = self.EndTicks / self.Frequency

    def ReadFromNex5File(self, theFile) -> None:
        """
        Read the header from a .nex5 file.
        """
        theDict = {}
        for key in self.Nex5FileSpecs:
            theDict[key] = _ReadStructField(theFile, self.Nex5FileSpecs[key])
        if theDict["MagicNumber"] != self.Nex5FileMagicNumber:
            # do not continue if incorrect magic number
            raise ValueError("invalid .nex5 file header")

        theDict["Comment"] = _ToString(theDict["Comment"])
        del theDict["Padding"]
        for key in theDict:
            setattr(self, key, theDict[key])
        self.BegSeconds = self.BegTicks / self.Frequency
        self.EndSeconds = self.EndTicks / self.Frequency


@dataclass
class VariableHeader:
    """
    Represents the header of a variable in a .nex or .nex5 file.
    """

    Type: int = -1
    Version: int = 0
    Name: str = ""
    DataOffset: int = 0
    Count: int = 0
    TsDataType: int = 0
    ContDataType: int = 0
    SamplingRate: float = 0
    Units: str = ""
    ADtoMV: float = 0
    MVOffset: float = 0
    NPointsWave: int = 0
    PreThrTime: float = 0
    MarkerDataType: int = 0
    NMarkers: int = 0
    MarkerLength: int = 0
    ContFragIndexType: int = 0
    Padding: str = ""

    Gain: int = 0
    Filter: int = 0
    Wire: int = 0
    Unit: int = 0
    XPos: float = 0
    YPos: float = 0

    NexFileSpecs: dict = field(default_factory=dict, repr=False)

    Nex5FileSpecs: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """
        Initialize header specifications for .nex and .nex5 files.
        """
        self.NexFileSpecs = {
            "Type": "<i",
            "Version": "<i",
            "Name": "64s",
            "DataOffset": "<i",
            "Count": "<i",
            "Wire": "<i",
            "Unit": "<i",
            "Gain": "<i",
            "Filter": "<i",
            "XPos": "<d",
            "YPos": "<d",
            "SamplingRate": "<d",
            "ADtoMV": "<d",
            "NPointsWave": "<i",
            "NMarkers": "<i",
            "MarkerLength": "<i",
            "MVOffset": "<d",
            "PreThrTime": "<d",
            "Padding": "52s",
        }

        self.Nex5FileSpecs = {
            "Type": "<i",
            "Version": "<i",
            "Name": "64s",
            "DataOffset": "<Q",
            "Count": "<Q",
            "TsDataType": "<i",
            "ContDataType": "<i",
            "SamplingRate": "<d",
            "Units": "32s",
            "ADtoMV": "<d",
            "MVOffset": "<d",
            "NPointsWave": "<Q",
            "PreThrTime": "<d",
            "MarkerDataType": "<i",
            "NMarkers": "<i",
            "MarkerLength": "<i",
            "ContFragIndexType": "<i",
            "Padding": "60s",
        }

    def WriteToNexFile(self, theFile) -> None:
        """
        Write the header to a .nex file.
        """
        for key in self.NexFileSpecs:
            value = getattr(self, key)
            _WriteField(theFile, self.NexFileSpecs[key], value)

    def WriteToNex5File(self, theFile) -> None:
        """
        Write the header to a .nex5 file.
        """
        for key in self.Nex5FileSpecs:
            value = getattr(self, key)
            _WriteField(theFile, self.Nex5FileSpecs[key], value)

    def ReadFromNexFile(self, theFile) -> None:
        """
        Read the header from a .nex file.
        """
        theDict = {}
        for key in self.NexFileSpecs:
            theDict[key] = _ReadStructField(theFile, self.NexFileSpecs[key])
        del theDict["Padding"]

        theDict["Name"] = _ToString(theDict["Name"], True)
        for key in theDict:
            setattr(self, key, theDict[key])

    def ReadFromNex5File(self, theFile) -> None:
        """
        Read the header from a .nex5 file.
        """
        theDict = {}
        for key in self.Nex5FileSpecs:
            theDict[key] = _ReadStructField(theFile, self.Nex5FileSpecs[key])
        del theDict["Padding"]

        theDict["Name"] = _ToString(theDict["Name"])
        theDict["Units"] = _ToString(theDict["Units"])

        if theDict["ContDataType"] == 1:
            theDict["ADtoMV"] = 1
            theDict["MVOffset"] = 0

        for key in theDict:
            setattr(self, key, theDict[key])

    def GetContDataPars(self) -> (DataFormat, float, float):
        """
        Get continuous data parameters.

        Returns:
            tuple : A tuple containing continuous data format, raw to MV conversion factor, and offset in MV.
        """
        rawToMv = self.ADtoMV
        offset = self.MVOffset
        contValueType = DataFormat.INT16
        if self.ContDataType == 1:
            contValueType = DataFormat.FLOAT32
            rawToMv = 1.0
            offset = 0.0
        return contValueType, rawToMv, offset

    def GetTimestampDataFormat(self) -> DataFormat:
        """
        Get the timestamp data format.

        Returns:
            DataFormat : The timestamp data format.
        """
        tsValueType = DataFormat.INT32
        if self.TsDataType == 1:
            tsValueType = DataFormat.INT64
        return tsValueType

    def GetFragmentIndexDataFormat(self) -> DataFormat:
        """
        Get the fragment index data format.

        Returns:
            DataFormat : The fragment index data format.

        """
        return DataFormat.UINT32

    def BytesInTimestamp(self) -> int:
        """
        Calculate the number of bytes in a timestamp.

        Returns:
            int : The number of bytes in a timestamp.
        """
        if self.TsDataType == 0:
            return 4
        else:
            return 8

    def BytesInContValue(self) -> int:
        """
        Calculate the number of bytes in a continuous value.

        Returns:
            int : The number of bytes in a continuous value.
        """
        if self.ContDataType == 0:
            return 2
        else:
            return 4

    def BytesInFragmentIndex(self) -> int:
        """
        Calculate the number of bytes in a fragment index.

        Returns:
            int : The number of bytes in a fragment index.
        """
        return 4
