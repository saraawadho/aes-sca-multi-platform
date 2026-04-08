import pyvisa as visa
import pyvisa.errors
from pyvisa.resources import MessageBasedResource
import sys
import numpy as np
import re
import time

class TeledyneLeCroyScope:
    def __init__(
            self, 
            ip="172.31.109.19",
            timout_ms = 60_000,
            int16_not_int8=False,
            float_not_int=True):
        self.rm = visa.ResourceManager()
        resources = self.rm.list_resources()
        self.lecroy = None
        for r in resources:
            if (ip in r):
                print('Connecting to ' + str(r))
                self.lecroy = self.rm.open_resource(
                        r, resource_pyclass=MessageBasedResource)
                break
        if self.lecroy is None:
            print("Found instruments: " + str(resources))
            raise ValueError('Scope not found')
        self.lecroy.timeout = timout_ms
        self.lecroy.write("COMM_HEADER OFF")
        self.lecroy.query("*OPC?")
        self.lecroy.write("COMM_FORMAT DEF9,{},BIN".format("WORD" if int16_not_int8 else "BYTE"))
        self.lecroy.query("*OPC?")
        self.dtype = np.int16 if int16_not_int8 else np.int8
        self.int16_not_int8 = int16_not_int8
        self.float_not_int = float_not_int
        self.scale = (1/(2**16)) if int16_not_int8 else (1/(2**8))
        self.lecroy.write("TRIG_MODE NORM")
        self.lecroy.query("*OPC?")
    def __del__(self):
        print('Disconnecting scope')
        self.lecroy.close()
        self.rm.close()
    def arm_single_trace(self):
        self.lecroy.query("*OPC?")
        self.lecroy.write("TRIG_MODE SINGLE")
        self.lecroy.query(r"""vbs? 'return=app.WaitUntilIdle(5)' """)
        self.lecroy.write("WAIT")
    def get_single_trace(self, channel='C2', nb_of_samples = 1E12):
        self.lecroy.query("*OPC?")
        self.lecroy.write("TRIG_MODE STOP")
        self.lecroy.query("*OPC?")
        self.lecroy.write('{}:WF? DAT1'.format(channel))
        header = self.lecroy.read_bytes(16)
        if header[5:6] != b'#':
            raise ValueError('special character not found')
        nb_of_digits = int(header[6:7].decode('ascii'), 16)
        if nb_of_digits > 9:
            header += self.lecroy.read_bytes(nb_of_digits - 9)
        nb_of_bytes = int(header[7:7+nb_of_digits])
        nb_of_bytes = min(nb_of_bytes, nb_of_samples << int(self.int16_not_int8))
        bytestream = self.lecroy.read_bytes(nb_of_bytes)
        trace = np.frombuffer(bytestream, dtype=self.dtype)
        if self.float_not_int:
            trace = trace.astype(float) * self.scale
        return trace
    def init_mean_trace(self, channel='C2'):
        self.lecroy.write("CLEAR_SWEEPS") # restart cumulative processing
        self.lecroy.query("*OPC?")
        self.lecroy.write("F1:DEFINE EQN,'AVG({})',AVERAGETYPE,SUMMED".format(channel))
        self.lecroy.query("*OPC?")
        self.lecroy.write("TRIG_MODE NORM")
        self.lecroy.query("*OPC?")
        self.sweeps_per_acq = 0
    def arm_mean_trace(self):
        for i in range(1000):
            r = self.lecroy.query("F1:INSPECT? SWEEPS_PER_ACQ")
            r = int(re.search(r'\d+', r).group())
            self.lecroy.query(r"""vbs? 'return=app.WaitUntilIdle(5)' """)
            if r == self.sweeps_per_acq:                
                self.sweeps_per_acq += 1 
                return False
            time.sleep(0.05)
        print('Warning: missed a trace')
        return True
    def get_mean_trace(self, nb_of_samples = 1E12):
        self.arm_mean_trace()
        self.lecroy.write("TRIG_MODE STOP")
        self.lecroy.query("*OPC?")
        self.lecroy.write('F1:WF? DAT1')
        header = self.lecroy.read_bytes(16)
        nb_of_bytes = int(header[7:16])
        nb_of_bytes = min(nb_of_bytes, nb_of_samples << int(self.int16_not_int8))
        bytestream = self.lecroy.read_bytes(nb_of_bytes)
        trace = np.frombuffer(bytestream, dtype=self.dtype)
        if self.float_not_int:
            trace = trace.astype(float) * self.scale
        return trace