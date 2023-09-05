from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from nex5file.headers import VariableHeader
from nex5file.variables import (
    NexFileVarType,
    Variable,
    EventVariable,
    NeuronVariable,
    IntervalVariable,
    MarkerVariable,
    WaveformVariable,
    ContinuousVariable,
    PopulationVector,
)
import numpy as np


class FileData:
    """
    FileData: Class for Managing Data in .nex and .nex5 Data Files.

    Parameters:
        timestamp_frequencyHz (float): The timestamp frequency in Hertz. Defaults to 100,000 Hz.

    Raises:
        ValueError: If timestamp_frequencyHz is less than or equal to 0.

    Example:
        ```python
        from nex5file.filedata import FileData

        # Create a FileData instance with a custom timestamp frequency
        file_data = FileData(timestamp_frequencyHz=50000, comment="Sample Data")

        # Add an event variable to the data
        file_data.AddEvent("EventVariable", [1.0, 2.0, 3.0])

        # Retrieve a variable by name
        event_var = file_data["EventVariable"]

        # Get the timestamp frequency
        timestamp_frequency = file_data.GetTimestampFrequency()
        ```
    """

    def __init__(self, tsFrequency: float = 10000, comment: str = ""):
        if tsFrequency <= 0:
            raise ValueError("invalid timestamp frequency")
        self.timestamp_frequencyHz: float = tsFrequency
        self.comment: str = comment
        self.beg_seconds: float = 0
        self.end_seconds: float = 0
        self.metadata: dict = {}
        self.variables: list = []

    def __eq__(self, other: FileData) -> bool:
        return (
            self.timestamp_frequencyHz == other.timestamp_frequencyHz
            and self.comment == other.comment
            and self.beg_seconds == other.beg_seconds
            and self.end_seconds == other.end_seconds
            and self.variables == other.variables
        )

    def __getitem__(self, index: str) -> Variable:
        for var in self.variables:
            if var.header.Name == index:
                return var
        raise ValueError(f'unable to find variable "{index}" in file data')

    def _AddVariable(self, theVar: Variable) -> None:
        """
        Add a Variable to the FileData instance.
        This is an internal method of FileData class.

        Parameters:
            theVar (Variable): The variable to add.

        Raises:
            ValueError: If theVar's type is negative or its name is empty or if a variable with the same name already exists in the FileData instance.
        """
        if theVar.header.Type < 0 or theVar.header.Name == "":
            raise ValueError(f"unable to add variable. Variable is invalid")

        for var in self.variables:
            if var.header.Name == theVar.header.Name:
                raise ValueError(
                    f'unable to add variable with name "{var.header.Name}". Variable with this name already exists'
                )
        self.variables.append(theVar)
        self.end_seconds = self.MaximumTimestamp()

    def DeleteVariable(self, name: str) -> None:
        """
        Delete a variable from the FileData instance by its name.

        Parameters:
            name (str): The name of the Variable to be deleted.

        Raises:
            ValueError: If no Variable with the specified name exists in the FileData instance.
        """
        var = self.__getitem__(name)
        self.variables.remove(var)

    def GetDocComment(self) -> str:
        """
        Get the comment of the FileData instance.

        Returns:
            str: The comment of the FileData instance.
        """
        return self.comment

    def GetTimestampFrequency(self) -> float:
        """
        Get the timestamp frequency of the FileData instance in Hertz.

        Returns:
            float: The timestamp frequency in Hertz.
        """
        return self.timestamp_frequencyHz

    def GetDocStartTime(self) -> float:
        """
        Get the start time of the data in the FileData instance in seconds.

        Returns:
            float: The start time in seconds.
        """
        return self.beg_seconds

    def GetDocEndTime(self) -> float:
        """
        Get the end time of the data in the FileData instance in seconds.

        Returns:
            float: The end time in seconds.
        """
        return self.end_seconds

    def _VarNames(self, varType: NexFileVarType) -> list:
        """
        Get a list of variable names of a specific type in the FileData instance.
        This is an internal method of FileData class.

        Parameters:
            varType (NexFileVarType): The type of variables to retrieve.

        Returns:
            List[str]: A list of variable names of the specified type.
        """
        names = []
        for var in self.variables:
            if var.header.Type == varType:
                names.append(var.header.Name)
        return names

    def NeuronNames(self) -> list:
        """
        Get the list of neuron variable names in the FileData instance.

        Returns:
            List[str]: A list of neuron variable names.
        """
        return self._VarNames(NexFileVarType.NEURON)

    def EventNames(self) -> list:
        """
        Get the list of event variable names in the FileData instance.

        Returns:
            List[str]: A list of event variable names.
        """
        return self._VarNames(NexFileVarType.EVENT)

    def IntervalNames(self) -> list:
        """
        Get the list of interval variable names in the FileData instance.

        Returns:
            List[str]: A list of interval variable names.
        """
        return self._VarNames(NexFileVarType.INTERVAL)

    def WaveNames(self) -> list:
        """
        Get the list of waveform variable names in the FileData instance.

        Returns:
            List[str]: A list of waveform variable names.
        """
        return self._VarNames(NexFileVarType.WAVEFORM)

    def MarkerNames(self) -> list:
        """
        Get the list of marker variable names in the FileData instance.

        Returns:
            List[str]: A list of marker variable names.
        """
        return self._VarNames(NexFileVarType.MARKER)

    def ContinuousNames(self) -> list:
        """
        Get the list of continuous variable names in the FileData instance.

        Returns:
            List[str]: A list of continuous variable names.
        """
        return self._VarNames(NexFileVarType.CONTINUOUS)

    def PopVectorNames(self) -> list:
        """
        Get the list of population vector variable names in the FileData instance.

        Returns:
            List[str]: A list of population vector variable names.
        """
        return self._VarNames(NexFileVarType.POPULATION_VECTOR)

    def MaximumTimestamp(self) -> float:
        """
        Get the maximum timestamp among all variables in the FileData instance.

        Returns:
            float: The maximum timestamp found in the FileData instance.
        """
        maxTs = 0
        for v in self.variables:
            maxTs = max(maxTs, v.MaximumTimestamp())
        return maxTs

    def NumberOfBytesInData(self) -> int:
        """
        Get the total number of bytes required to save data of all variables.

        Returns:
            int: the total number of bytes required to save data of all variables.
        """
        sum = 0
        for v in self.variables:
            sum += v._BytesInData()
        return sum

    def AddEvent(self, evName: str, evTimestamps: list) -> None:
        """
        Add an event variable to the FileData instance.

        Parameters:
            evName (str): The name of the event variable.
            evTimestamps (List[float]): Event timestamps in seconds.

        Example:
            ```python
            file_data.AddEvent("EventVariable", [1.0, 2.0, 3.0])
            ```
        """
        h = VariableHeader(Type=NexFileVarType.EVENT, Name=evName)
        ev = EventVariable(h)
        ev.timestamps = np.array(evTimestamps)
        self.variables.append(ev)

    def AddNeuron(
        self,
        nrName: str,
        nrTimestamps: list,
        wire: int = 0,
        unit: int = 0,
        xPosition: float = 0,
        yPosition: float = 0,
    ) -> None:
        """
        Add a neuron variable to the FileData instance.

        Parameters:
            nrName (str): The name of the neuron variable.
            nrTimestamps (List[float]): Neuron timestamps in seconds.
            wire (int): The wire number. Defaults to 0.
            unit (int): The unit number. Defaults to 0.
            xPosition (float): The x-position in range [0,100]. Defaults to 0.
            yPosition (float): The y-position in range [0,100]. Defaults to 0.

        Example:
            ```python
            file_data.AddNeuron("NeuronVariable", [1.0, 2.0, 3.0])
            ```
        """
        h = VariableHeader(
            Type=NexFileVarType.NEURON, Name=nrName, Wire=wire, Unit=unit, XPos=xPosition, YPos=yPosition
        )
        nr = NeuronVariable(h)
        nr.timestamps = np.array(nrTimestamps)
        self._AddVariable(nr)

    def AddIntervalAsPairsStartEnd(self, intName: str, intervalsAsPairs) -> None:
        """
        Add an interval variable to the FileData instance using start and end pairs.

        Parameters:
            intName (str): The name of the interval variable.
            intervalsAsPairs (list of tuples): List of interval start and end pairs in seconds.

        Example:
            ```python
            intervals = [(0.0, 1.0), (1.5, 2.0)]
            file_data.AddIntervalAsPairsStartEnd("IntervalVariable", intervals)
            ```
        """
        h = VariableHeader(Type=NexFileVarType.INTERVAL, Name=intName)
        starts = []
        ends = []
        for interval in intervalsAsPairs:
            starts.append(interval[0])
            ends.append(interval[1])
        intVar = IntervalVariable(h)
        intVar.interval_starts = np.array(starts)
        intVar.interval_ends = np.array(ends)
        self._AddVariable(intVar)

    def AddMarker(self, markerName, timestamps: list, fieldNames: list, fields):
        """
        Add a marker variable to the FileData instance.

        Parameters:
            markerName (str): The name of the marker variable.
            timestamps (List[float]): The timestamps in seconds.
            fieldNames (List[str]): The names of marker fields.
            fields (list): List of marker fields. Each element of the list contains values for a field.

        Raises:
            ValueError: If the number of field names does not match the number of fields or
                        if the length of any field does not match the length of timestamps.

        Example:
            ```python
            field_names = ["Field1", "Field2"]
            fields = [[1.0, 2.0], [3.0, 'abc']]
            file_data.AddMarker("MarkerVariable", [0.0, 1.0], field_names, fields)
            ```
        """
        if not len(fieldNames) == len(fields):
            raise ValueError("invalid marker parameters")
        for f in fields:
            if not len(f) == len(timestamps):
                raise ValueError("invalid marker parameters")
        h = VariableHeader(Type=NexFileVarType.MARKER, Name=markerName, NMarkers=len(fieldNames))
        markerVar = MarkerVariable(h)
        markerVar.timestamps = np.array(timestamps)
        markerVar.marker_field_names = fieldNames
        markerVar.marker_fields = fields
        self._AddVariable(markerVar)

    def AddContVarWithFloatsSingleFragment(
        self, contName: str, samplingRate: float, startTimestamp: float, contValues: list
    ) -> None:
        """
        Add a continuous variable with float values and a single fragment to the FileData instance.

        Parameters:
            contName (str): The name of the continuous variable.
            samplingRate (float): The sampling rate of the continuous variable in Hz.
            startTimestamp (float): The timestamp at the start of the first fragment in seconds.
            contValues (List[float]): The continuous values in milliVolts.

        Example:
            ```python
            file_data.AddContVarWithFloatsSingleFragment("ContVariable", 1000.0, 0.0, [1.0, 2.0, 3.0])
            ```
        """
        if samplingRate < 0 or samplingRate > self.timestamp_frequencyHz:
            raise ValueError(
                "invalid sampling rate: the rate should be positive and less than FileData timestamp frequency"
            )
        if isinstance(startTimestamp, (list, np.ndarray)):
            raise ValueError("invalid startTimestamp: should be a single value, not a list")

        h = VariableHeader(
            Type=NexFileVarType.CONTINUOUS,
            Name=contName,
            SamplingRate=samplingRate,
            NPointsWave=len(contValues),
            ContDataType=1,
            ADtoMV=1.0,
        )
        cont = ContinuousVariable(h)
        cont.fragment_timestamps = np.array([startTimestamp])
        cont.fragment_indexes = np.array([0])
        cont.continuous_values = np.array(contValues)
        cont._CalculateFragmentCountsFromIndexes()
        self._AddVariable(cont)

    def AddContSingleFragmentValuesInt16(
        self,
        contName: str,
        samplingRate: float,
        startTimestamp: float,
        contValuesAsInt16: list,
        rawToMV: float,
        rawOffset: float,
    ) -> None:
        """
        Add a continuous variable with a single fragment by specifying int16 values and scaling to the FileData instance.

        Parameters:
            contName (str): The name of the continuous variable.
            samplingRate (float): The sampling rate of the continuous variable in Hz.
            startTimestamp (float): The timestamp at the start of the first fragment in seconds.
            contValuesAsInt16 (numpy array of type np.int16): The continuous values as int16 values.
            rawToMV (float): Conversion factor from AD units to milliVolts (MV).
            rawOffset (float): Offset in MV.

        Example:
            ```python
            file_data.AddContSingleFragmentValuesInt16("ContVariable", 1000.0, 0.0, [100, 200, 300], 0.1, 0.0)
            ```
        """
        if samplingRate < 0 or samplingRate > self.timestamp_frequencyHz:
            raise ValueError(
                "invalid sampling rate: the rate should be positive and less than FileData timestamp frequency"
            )
        if isinstance(startTimestamp, (list, np.ndarray)):
            raise ValueError("invalid startTimestamp: should be a single value, not a list")

        h = VariableHeader(
            Type=NexFileVarType.CONTINUOUS,
            Name=contName,
            SamplingRate=samplingRate,
            NPointsWave=len(contValuesAsInt16),
            ADtoMV=rawToMV,
            MVOffset=rawOffset,
            ContDataType=0,
        )
        #  scale cont values
        values = np.array(contValuesAsInt16) * rawToMV + rawOffset
        cont = ContinuousVariable(h)
        cont.fragment_timestamps = np.array([startTimestamp])
        cont.fragment_indexes = np.array([0])
        cont.continuous_values = values
        cont._CalculateFragmentCountsFromIndexes()
        cont._HashContValues()
        self._AddVariable(cont)

    def AddContVarWithFloatsAllTimestamps(
        self,
        contName: str,
        samplingRate: float,
        allTimestamps: list,
        contValues: list,
    ) -> None:
        """
        Add a continuous variable with float values and all the timestamps to the data.

        Parameters:
            contName (str): The name of the continuous variable.
            samplingRate (float): The sampling rate of the continuous variable in Hz.
            allTimestamps (numpy array of type np.float64): The timestamps for all data points in seconds.
            contValues (numpy array of type np.float32): The continuous values in milliVolts.

        Example:
            ```python
            timestamps = [0.0, 0.001, 0.002]
            values = [1.0, 2.0, 3.0]
            file_data.AddContVarWithFloatsAllTimestamps("ContVariable", 1000.0, timestamps, values)
            ```
        """
        if samplingRate < 0 or samplingRate > self.timestamp_frequencyHz:
            raise ValueError(
                "invalid sampling rate: the rate should be positive and less than FileData timestamp frequency"
            )
        if len(allTimestamps) != len(contValues):
            raise ValueError("invalid timestamps and values (both arrays should be the same length)")

        h = VariableHeader(
            Type=NexFileVarType.CONTINUOUS,
            Name=contName,
            SamplingRate=samplingRate,
            NPointsWave=len(contValues),
            ContDataType=1,
        )
        cont = ContinuousVariable(h)
        cont.fragment_timestamps = np.array(allTimestamps)
        cont.continuous_values = np.array(contValues)
        cont._CalculateFragmentsFromAllTimestamps(self.timestamp_frequencyHz)
        cont._CalculateFragmentCountsFromIndexes()
        self.variables.append(cont)

    def AddWaveVarWithFloats(
        self,
        waveName: str,
        samplingRate: float,
        timestamps: list,
        waveValues: list,
    ) -> None:
        """
        Add a waveform variable with float values to the data.

        Parameters:
            waveName (str): The name of the waveform variable.
            samplingRate (float): The sampling rate of the waveform variable in Hz.
            timestamps (numpy array of type np.float64): The timestamps in seconds.
            waveValues (numpy array of type np.float32): The waveform values as a NumPy array. Each column represents a waveform.

        Example:
            ```python
            wave_name = "WaveformVariable"
            sampling_rate = 1000.0  # Hz
            timestamps = [0.0, 1.0, 2.0]
            wave_values = [[2, 3, 4, 1], [5, 6, 7, 2]

            file_data.AddWaveVarWithFloats(wave_name, sampling_rate, timestamps, wave_values)
            ```

        This method adds a waveform variable to the data with the specified name, sampling rate, timestamps, and waveform values.
        """
        if samplingRate < 0 or samplingRate > self.timestamp_frequencyHz:
            raise ValueError(
                "invalid sampling rate: the rate should be positive and less than FileData timestamp frequency"
            )

        h = VariableHeader(
            Type=NexFileVarType.WAVEFORM,
            Name=waveName,
            SamplingRate=samplingRate,
            ContDataType=1,
        )
        wave = WaveformVariable(h)
        wave.timestamps = np.array(timestamps)
        wave.waveform_values = np.array(waveValues)
        wave._AssignNumPointsWave()
        self._AddVariable(wave)


MakeFileVar = {
    NexFileVarType.EVENT: EventVariable,
    NexFileVarType.NEURON: NeuronVariable,
    NexFileVarType.INTERVAL: IntervalVariable,
    NexFileVarType.MARKER: MarkerVariable,
    NexFileVarType.WAVEFORM: WaveformVariable,
    NexFileVarType.CONTINUOUS: ContinuousVariable,
    NexFileVarType.POPULATION_VECTOR: PopulationVector,
}
