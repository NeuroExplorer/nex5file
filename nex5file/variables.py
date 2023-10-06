from __future__ import annotations
from nex5file.headers import VariableHeader
import numpy as np
from typing import Tuple
import hashlib


class NexFileVarType:
    """
    Constants for .nex and .nex5 variable types.
    """

    NEURON = 0
    EVENT = 1
    INTERVAL = 2
    WAVEFORM = 3
    POPULATION_VECTOR = 4
    CONTINUOUS = 5
    MARKER = 6


def MaxOfNumpyArrayOrZero(x):
    """
    Compute the maximum value in a numpy array or return 0 if the array is empty.

    Parameters:
        x : array_like Input array. An array or array-like object containing numeric elements.

    Returns:
        float : The maximum value in the input array if it is not empty, or 0 if the array is empty.

    :meta private:
    """

    if len(x) == 0:
        return 0
    else:
        return np.max(x)


def NumpyArraysEqual(a, b) -> bool:
    """
    Returns true if numpy arrays equal.

    Parameters:
        a : numpy array
        b : numpy array to compare with 1

    Returns:
        bool : True if numpy arrays equal.

    :meta private:
    """
    return a.shape == b.shape and (a == b).all()


class Variable:
    """
    Represents a variable stored in .nex of .nex5 file.
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}

    def __eq__(self, other: Variable) -> bool:
        return self.header.Name == other.header.Name

    def Metadata(self) -> dict:
        """
        Get the metadata associated with the variable.

        Returns:
            dict : A dictionary containing metadata about the variable (name, wire number for neurons etc.).

        """
        return self.metadata

    def _MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp in seconds.

        Returns:
            float : The maximum timestamp.

        :meta private:

        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def Weights(self) -> list:
        """
        Get a copy of the population vector weights.

        Returns:
            List[float] :A copy of the weights as a numpy array.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def Timestamps(self) -> list:
        """
        Get a copy of the timestamps in seconds.

        Returns:
            numpy array of type np.float64 : A copy of the timestamps (in seconds) as a numpy array.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def Intervals(self) -> list:
        """
        Get a list of intervals represented as two lists:
            the first list is a list of interval starts
            the second list is a list of interval ends

        Returns:
            List[List[float]] : a list of intervals represented as two lists. The first list is a list of interval starts.
                The second list is a list of interval ends.


        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def MarkerFieldNames(self) -> list:
        """
        Get a copy of the marker field names.

        Returns:
            List[str] : A copy of the marker field names.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def Markers(self) -> list:
        """
        Get a copy of the marker fields.

        Returns:
            List[List[str]] : A copy of the marker fields.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def SamplingRate(self) -> float:
        """
        Get the sampling rate of the waveform.

        Returns:
            float : The sampling rate.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def NumPointsInWave(self) -> int:
        """
        Get the number of points in the waveform.

        Returns:
            int : The number of points in the waveform.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def WaveformValues(self) -> list:
        """
        Get a copy of the waveform values.

        Returns:
            numpy array of type np.float32 : A copy of the waveform values as a numpy array.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def PreThresholdTime(self) -> float:
        """
        Get the pre-threshold time.

        Returns:
            float : The pre-threshold time.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def ContinuousValues(self) -> list:
        """
        Get a copy of the continuous values in millivolts.

        Returns:
            numpy array of type np.float64 : A copy of the continuous values as a numpy array.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def FragmentTimestamps(self) -> list:
        """
        Get a copy of the fragment timestamps in seconds.

        Returns:
            numpy array of type np.float64 : A copy of the fragment timestamps as a numpy array.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")

    def FragmentCounts(self) -> list:
        """
        Get a copy of the fragment counts.

        Returns:
            numpy array of type np.int64 : A copy of the fragment counts as a numpy array.

        :meta private:
        """
        raise NotImplementedError("Wrong variable type. This method is not available for this instance of Variable.")


class PopulationVector(Variable):
    """
    Represents a population vector variable (a variable name and a list of weights).
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        self.weights = np.zeros(0)

    def __eq__(self, other: PopulationVector) -> bool:
        return self.header.Name == other.header.Name and NumpyArraysEqual(self.weights, other.weights)

    def Weights(self) -> list:
        """
        Get a copy of the population vector weights.

        Returns:
            List[float] :A copy of the weights as a numpy array.
        """
        return np.copy(self.weights)

    def _BytesInData(self) -> int:
        """
        Get the total number of bytes used to store the weights data.

        Returns:
            int : The total number of bytes used.
        """
        return len(self.weights) * 8

    def _CountForHeader(self) -> int:
        """
        Get the count of weights for header information.

        Returns:
            int : The number of weights.
        """
        return len(self.weights)


class EventVariable(Variable):
    """
    Represents an event variable (event name and a list of timestamps).
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        self.timestamps = np.zeros(0)

    def __eq__(self, other: EventVariable) -> bool:
        return self.header.Name == other.header.Name and NumpyArraysEqual(self.timestamps, other.timestamps)

    def Timestamps(self) -> list:
        """
        Get a copy of the timestamps in seconds.

        Returns:
            numpy array of type np.float64 : A copy of the timestamps (in seconds) as a numpy array.
        """
        return np.copy(self.timestamps)

    def _MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp value in seconds.

        Returns:
            float : The maximum timestamp or 0 if there are no timestamps.
        """
        return MaxOfNumpyArrayOrZero(self.timestamps)

    def _BytesInData(self) -> int:
        """
        Get the total number of bytes used to store the timestamps data.

        Returns:
            int : The total number of bytes to store the timestamps data.
        """
        return self.header_for_writing.BytesInTimestamp() * len(self.timestamps)

    def _CountForHeader(self) -> int:
        """
        Get the number of timestamps for header information.

        Returns:
            int : The number of timestamps.
        """
        return len(self.timestamps)


class NeuronVariable(EventVariable):
    """
    Represents a neuron variable (a variable name, timestamps and wire and unit information).
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        if self.header.Version < 500:
            self.metadata["unitNumber"] = self.header.Unit
            self.metadata["probe"] = {
                "position": {"x": self.header.XPos, "y": self.header.YPos},
                "wireNumber": self.header.Wire,
            }
        else:
            self.metadata["probe"] = {"position": {"x": 0, "y": 0}, "wireNumber": 0}
            self.metadata["unitNumber"] = 0
        self.timestamps = np.zeros(0)

    def __eq__(self, other: NeuronVariable) -> bool:
        return (
            self.header.Name == other.header.Name
            and self.header.Wire == other.header.Wire
            and self.header.Unit == other.header.Unit
            and self.header.XPos == other.header.XPos
            and self.header.YPos == other.header.YPos
            and NumpyArraysEqual(self.timestamps, other.timestamps)
        )

    def _AssignFromVarMeta(self) -> None:
        if "unitNumber" in self.metadata:
            self.header.Unit = self.metadata["unitNumber"]
        if "probe" in self.metadata and "wireNumber" in self.metadata["probe"]:
            self.header.Wire = self.metadata["probe"]["wireNumber"]
        if (
            "probe" in self.metadata
            and "position" in self.metadata["probe"]
            and "x" in self.metadata["probe"]["position"]
        ):
            self.header.XPos = self.metadata["probe"]["position"]["x"]
        if (
            "probe" in self.metadata
            and "position" in self.metadata["probe"]
            and "y" in self.metadata["probe"]["position"]
        ):
            self.header.YPos = self.metadata["probe"]["position"]["y"]


class IntervalVariable(Variable):
    """
    Represents an interval variable (a variable name and interval start and end times).
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        self.timestamps = np.zeros(0)
        self.interval_starts = np.zeros(0)
        self.interval_ends = np.zeros(0)

    def __eq__(self, other: IntervalVariable) -> bool:
        return (
            self.header.Name == other.header.Name
            and NumpyArraysEqual(self.interval_starts, other.interval_starts)
            and NumpyArraysEqual(self.interval_ends, other.interval_ends)
        )

    def Intervals(self) -> list:
        """
        Get a list of intervals represented as two lists -- the first list is a list of interval starts, the second list is a list of interval ends.

        Returns:
            list : a list of intervals represented as two lists -- the first list is a list of interval starts, the second list is a list of interval ends.
        """
        return [self.interval_starts.tolist(), self.interval_ends.tolist()]

    def _MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp among the end times of the intervals.

        Returns:
            float : The maximum timestamp among the end times or 0 if there are no intervals.
        """
        return MaxOfNumpyArrayOrZero(self.interval_ends)

    def _CountForHeader(self) -> int:
        """
        Get the count of intervals for header information.

        Returns:
            int : The count of intervals.
        """
        return len(self.interval_starts)

    def _BytesInData(self) -> int:
        """
        Get the total number of bytes used to store the intervals data.

        Returns
            int : The total number of bytes used to store the intervals data.
        """
        return self.header.BytesInTimestamp() * len(self.interval_starts) * 2


class MarkerVariable(Variable):
    """
    Represents a marker variable with. Each marker is a timestamp with one or more associated string values.
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        self.timestamps = np.zeros(0)
        self.marker_field_names: list = []
        self.marker_fields: list = []

    def __eq__(self, other: MarkerVariable) -> bool:
        return (
            self.header.Name == other.header.Name
            and NumpyArraysEqual(self.timestamps, other.timestamps)
            and self.marker_field_names == other.marker_field_names
            and self.marker_fields == other.marker_fields
        )

    def Timestamps(self) -> list:
        """
        Get a copy of the timestamps associated with the markers.

        Returns:
            numpy array of type np.float64 : A copy of the timestamps as a numpy array.
        """
        return np.copy(self.timestamps)

    def MarkerFieldNames(self) -> list:
        """
        Get a copy of the marker field names.

        Returns:
            List[str] : A copy of the marker field names.
        """
        return self.marker_field_names.copy()

    def Markers(self) -> list:
        """
        Get a copy of the marker fields.

        Returns:
            List[List[str]] : A copy of the marker fields.
        """
        return self.marker_fields.copy()

    def _MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp.

        Returns:
            float : The maximum timestamp.
        """
        return MaxOfNumpyArrayOrZero(self.timestamps)

    def _CountForHeader(self) -> int:
        """
        Get the number of markers for header information.

        Returns:
            int : The number of markers.
        """
        return len(self.timestamps)

    def _BytesInData(self) -> int:
        """
        Get the total number of bytes used to store the marker data and field names.

        Returns:
            int : The total number of bytes used.
        """
        numMarkers = len(self.timestamps)
        return self.header_for_writing.BytesInTimestamp() * numMarkers + len(self.marker_field_names) * (
            64 + self.header_for_writing.MarkerLength * numMarkers
        )

    def _AllMarkerValuesAreNumeric(self) -> bool:
        """
        Check if all marker values are numeric.

        Returns:
            bool : True if all marker values can be converted to integers, False otherwise.
        """
        for field in self.marker_fields:
            for s in field:
                try:
                    int(s)
                except ValueError:
                    return False
        return True

    def _IfNumberStringsStoreAsNumbers(self) -> None:
        """
        If all marker values are numeric, convert number strings to integers.
        """
        if self._AllMarkerValuesAreNumeric():
            for i in range(len(self.marker_fields)):
                self.marker_fields[i] = [int(m) for m in self.marker_fields[i]]

    def _CalcMarkerLength(self) -> None:
        """
        Calculate the maximum marker length and set data type.
        """
        self.header.NMarkers = len(self.marker_field_names)
        if self._AllMarkerValuesAreNumeric():
            self.header.MarkerLength = 6
            self.header.MarkerDataType = 1
        else:
            self.header.MarkerDataType = 0
            maxStringLength = 0
            for field in self.marker_fields:
                for s in field:
                    if not isinstance(s, str):
                        raise ValueError("marker values should be either all numbers or all strings")
                    maxStringLength = max(maxStringLength, len(s))
            self.header.MarkerLength = max(6, maxStringLength + 1)


class WaveformVariable(Variable):
    """
    Represents a waveform variable (each waveform is a timestamps and array of waveform values).
    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        self.timestamps = np.zeros(0)
        self.waveform_values = np.zeros(0)
        self.hashed_wave_values: str = ""

    def __eq__(self, other: WaveformVariable) -> bool:
        return (
            self.header.Name == other.header.Name
            and NumpyArraysEqual(self.timestamps, other.timestamps)
            and NumpyArraysEqual(self.waveform_values, other.waveform_values)
        )

    def SamplingRate(self) -> float:
        """
        Get the sampling rate of the waveform.

        Returns:
            float : The sampling rate.
        """
        return self.header.SamplingRate

    def NumPointsInWave(self) -> int:
        """
        Get the number of points in the waveform.

        Returns:
            int : The number of points in the waveform.
        """
        return self.header.NPointsWave

    def Timestamps(self) -> list:
        """
        Get a copy of the timestamps.

        Returns:
            numpy array of type np.float64 : A copy of the timestamps as a numpy array.
        """
        return np.copy(self.timestamps)

    def WaveformValues(self) -> list:
        """
        Get a copy of the waveform values.

        Returns:
            numpy array of type np.float32 : A copy of the waveform values as a numpy array.
        """
        return np.copy(self.waveform_values)

    def PreThresholdTime(self) -> float:
        """
        Get the pre-threshold time.

        Returns:
            float : The pre-threshold time.
        """
        return self.header.PreThrTime

    def _MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp of all data points in the variable.

        Returns:
            float : The maximum timestamp or 0 if there are no timestamps.
        """
        if len(self.timestamps) == 0:
            return 0
        return MaxOfNumpyArrayOrZero(self.timestamps) + (self.header.NPointsWave - 1) / self.header.SamplingRate

    def _AssignNumPointsWave(self):
        """
        Assign the number of points in the waveform from the waveform values matrix.

        Raises:
            ValueError : If the waveform values matrix is invalid.
        """
        if len(self.timestamps) == 0:
            return
        shape = np.shape(self.waveform_values)
        if shape[0] != len(self.timestamps) or shape[1] == 0:
            raise ValueError("invalid waveform values matrix")
        self.header.NPointsWave = shape[1]

    def _HashContValues(self) -> None:
        """
        Hash the continuous waveform values to generate a unique identifier.
        """
        b = self.waveform_values.view(np.uint8)
        self.hashed_wave_values = hashlib.sha1(b).hexdigest()

    def _ContScaling(self, contDataType: int) -> Tuple[float, float]:
        """
        Calculate continuous scaling factors based on the continuous data type.

        Parameters:
            contDataType : int The continuous data type.

        Returns:
            Tuple[float, float] : A tuple containing the scaling factors (ADtoMV and MVOffset).
        """
        if contDataType == 1 or len(self.timestamps) == 0:
            return 1.0, 0.0
        else:
            # Return existing scaling if available
            # Hash is only calculated when loading var from file
            if len(self.hashed_wave_values) > 0:
                b = self.waveform_values.view(np.uint8)
                if hashlib.sha1(b).hexdigest() == self.hashed_wave_values:
                    return self.header.ADtoMV, self.header.MVOffset

            contMax = np.max(np.abs(self.waveform_values))
            if contMax == 0:
                return 1.0, 0.0
            else:
                return contMax / 32767.0, 0.0

    def _CountForHeader(self) -> int:
        """
        Get the number of timestamps for header information.

        Returns:
            int : The number of timestamps.
        """
        return len(self.timestamps)

    def _BytesInData(self) -> int:
        """
        Get the total number of bytes used to store the waveform data and timestamps.

        Returns:
            int : The total number of bytes used.
        """
        return (
            self.header_for_writing.BytesInTimestamp() * len(self.timestamps)
            + self.header_for_writing.BytesInContValue() * len(self.timestamps) * self.header.NPointsWave
        )


class ContinuousVariable(Variable):
    """
    Represents a continuous variable. The variable contains continuous values and their timestamps.

    """

    def __init__(self, varHeader: VariableHeader = VariableHeader()):
        self.header: VariableHeader = varHeader
        self.header_for_writing: VariableHeader = VariableHeader()
        self.metadata: dict = {"name": self.header.Name, "nameOriginal": self.header.Name}
        self.fragment_timestamps = np.zeros(0)
        self.fragment_indexes = np.zeros(0)
        self.fragment_counts = np.zeros(0)
        self.continuous_values = np.zeros(0)
        self.hashed_cont_values: str = ""

    def __eq__(self, other: ContinuousValues) -> bool:
        return (
            self.header.Name == other.header.Name
            and NumpyArraysEqual(self.fragment_timestamps, other.fragment_timestamps)
            and NumpyArraysEqual(self.fragment_indexes, other.fragment_indexes)
            and NumpyArraysEqual(self.fragment_counts, other.fragment_counts)
            and NumpyArraysEqual(self.continuous_values.astype(np.float32), other.continuous_values.astype(np.float32))
        )

    def SamplingRate(self) -> float:
        """
        Get the sampling rate of the continuous data.

        Returns:
            float : The sampling rate.
        """
        return self.header.SamplingRate

    def ContinuousValues(self) -> list:
        """
        Get a copy of the continuous values in millivolts.

        Returns:
            numpy array of type np.float64 : A copy of the continuous values as a numpy array.
        """
        return np.copy(self.continuous_values)

    def FragmentTimestamps(self) -> list:
        """
        Get a copy of the fragment timestamps in seconds.

        Returns:
            numpy array of type np.float64 : A copy of the fragment timestamps as a numpy array.
        """
        return np.copy(self.fragment_timestamps)

    def FragmentCounts(self) -> list:
        """
        Get a copy of the fragment counts.

        Returns:
            numpy array of type np.int64 : A copy of the fragment counts as a numpy array.
        """
        return np.copy(self.fragment_counts)

    def _HashContValues(self) -> None:
        """
        Hash the continuous waveform values to generate a unique identifier.
        This is an internal method.
        """
        b = self.continuous_values.view(np.uint8)
        self.hashed_cont_values = hashlib.sha1(b).hexdigest()

    def _CalculateFragmentsFromAllTimestamps(self, tsFreq: float) -> None:
        """
        Calculate fragments of continuous data based on timestamps and a given timestamp frequency.

        Parameters:
            tsFreq : float The timestamp frequency.

        """
        if len(self.fragment_timestamps) < 2:
            return

        # if cont freq is not a multiple of the main frequency, we cannot consolidate
        maxDiffToConsolidate = 0.000001
        sr = self.header.SamplingRate
        if abs(round(tsFreq / sr) - (tsFreq / sr)) > maxDiffToConsolidate:
            return

        step = round(tsFreq / sr)  # how many ticks between a/d values
        tsInTicks = np.round(self.fragment_timestamps * tsFreq)
        newFragmentTimestamps = []
        newFragmentStarts = []
        dataPointIndex = 0
        newFragmentTimestamps.append(self.fragment_timestamps[0])
        newFragmentStarts.append(dataPointIndex)

        expectedTimestampOfNextFragment = tsInTicks[0] + step
        for i in range(1, len(tsInTicks)):
            dataPointIndex += 1
            diff = tsInTicks[i] - expectedTimestampOfNextFragment
            if tsInTicks[i] - expectedTimestampOfNextFragment != 0:
                newFragmentTimestamps.append(self.fragment_timestamps[i])
                newFragmentStarts.append(dataPointIndex)
            expectedTimestampOfNextFragment = tsInTicks[i] + step

        self.fragment_timestamps = np.array(newFragmentTimestamps)
        self.fragment_indexes = np.array(newFragmentStarts).astype(np.uint32)
        self._CalculateFragmentCountsFromIndexes()

    def _CalculateFragmentCountsFromIndexes(self) -> None:
        """
        Calculate fragment counts from fragment indexes.
        """
        fragmentCounts = []
        for frag in range(len(self.fragment_indexes)):
            if frag < len(self.fragment_indexes) - 1:
                count = self.fragment_indexes[frag + 1] - self.fragment_indexes[frag]
            else:
                count = len(self.continuous_values) - self.fragment_indexes[frag]
            fragmentCounts.append(count)
        self.fragment_counts = np.array(fragmentCounts).astype(np.int64)

    def _MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp of all the values.

        Returns:
            float : The maximum timestamp or 0 if there are no fragment timestamps.
        """
        if len(self.fragment_timestamps) == 0:
            return 0
        return self.fragment_timestamps[-1] + (self.fragment_counts[-1] - 1) / self.header.SamplingRate

    def _ContScaling(self, contDataType: int) -> Tuple[float, float]:
        """
        Calculate continuous scaling factors based on the continuous data type.

        Parameters:
            contDataType : int The continuous data type.

        Returns:
            Tuple[float, float] : A tuple containing the scaling factors (ADtoMV and MVOffset).
        """
        if contDataType == 1 or len(self.continuous_values) == 0:
            return 1.0, 0.0
        else:
            # Return existing if available
            # Hash is only calculated when loading var from file
            # or when adding continuous variable using 16 bit values
            if len(self.hashed_cont_values) > 0:
                b = self.continuous_values.view(np.uint8)
                if hashlib.sha1(b).hexdigest() == self.hashed_cont_values:
                    return self.header.ADtoMV, self.header.MVOffset
            contMax = np.max(np.abs(self.continuous_values))
            if contMax == 0:
                return 1.0, 0.0
            else:
                return contMax / 32767.0, 0.0

    def _CountForHeader(self) -> int:
        """
        Get the number of fragment timestamps for header information.

        Returns:
            int : The number of fragment timestamps.
        """
        return len(self.fragment_timestamps)

    def _BytesInData(self) -> int:
        """
        Get the total number of bytes used to store the continuous data, fragment timestamps, and fragment indexes.

        Returns:
            int : The total number of bytes used.
        """
        return (self.header_for_writing.BytesInTimestamp() + self.header_for_writing.BytesInFragmentIndex()) * len(
            self.fragment_timestamps
        ) + self.header_for_writing.BytesInContValue() * len(self.continuous_values)
