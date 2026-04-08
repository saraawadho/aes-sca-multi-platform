import chipwhisperer as cw
import numpy as np
import time
from tqdm import tqdm
from Crypto.Cipher import AES

class CW308_STM32F4_AES:
    def __init__(
            self,
            path_to_firmware = "../firmware/AES-CW308_STM32F4.hex",
            timeout_ms = 5_000):
        self.scope = cw.scope(scope_type=cw.scopes.OpenADC)
        self.target = cw.target(self.scope, cw.targets.SimpleSerial)
        self.scope.default_setup()
        self.set_clock_freq_mhz(16)
        self.scope.trigger.module = 'basic'
        self.scope.trigger.triggers = 'tio4'
        self.scope.adc.basic_mode = 'rising_edge'
        self.scope.adc.bits_per_sample = 12
        self.scope.gain.db = 21
        self.scope.clock.adc_mul = 4
        self.scope.adc.samples = 120_000
        cw.program_target(self.scope, cw.programmers.STM32FProgrammer, path_to_firmware)
        self.timeout = timeout_ms
    def reboot_and_flush(self):
        self.scope.io.nrst = "low"
        time.sleep(0.05)
        self.scope.io.nrst = "high_z"
        time.sleep(0.05)
        self.target.flush()
    def set_clock_freq_mhz(self, f):
        self.scope.clock.clkgen_freq = f*1E6
        self.target.baud = 38400*f/7.37
        self.reboot_and_flush()
        if not self.scope.clock.clkgen_locked:
            raise ValueError("setting clock frequency failed")
    def read_key(self):
        self.target.simpleserial_write('R', bytearray([1]))
        return self.target.simpleserial_read('r', 16, timeout=self.timeout, ack=True)
    def read_plaintext(self):
        self.target.simpleserial_write('R', bytearray([2]))
        return self.target.simpleserial_read('r', 16, timeout=self.timeout, ack=True)
    def read_ciphertext(self):
        self.target.simpleserial_write('R', bytearray([3]))
        return self.target.simpleserial_read('r', 16, timeout=self.timeout, ack=True)
    def write_key(self, byte_string):
        self.target.simpleserial_write('W', bytearray([1]) + byte_string)
        if self.target.simpleserial_wait_ack(timeout=self.timeout) != 0:
            raise ValueError("writing key failed")
    def write_plaintext(self, byte_string):
        self.target.simpleserial_write('W', bytearray([2]) + byte_string)
        if self.target.simpleserial_wait_ack(timeout=self.timeout) != 0:
            raise ValueError("writing plaintext failed")
    def write_ciphertext(self, byte_string):
        self.target.simpleserial_write('W', bytearray([3]) + byte_string)
        if self.target.simpleserial_wait_ack(timeout=self.timeout) != 0:
            raise ValueError("writing plaintext failed")
    def encrypt(self):
        self.target.simpleserial_write('E', bytearray([]))
        if self.target.simpleserial_wait_ack(timeout=self.timeout) != 0:
            raise ValueError("AES encryption failed: ")
    def decrypt(self):
        self.target.simpleserial_write('D', bytearray([]))
        if self.target.simpleserial_wait_ack(timeout=self.timeout) != 0:
            raise ValueError("AES encryption failed: ")
    def random_16_bytes(self):
        return np.random.randint(0, high=256, size=16, dtype=np.uint8).tobytes()
    def test_correctness(self,
            nb_of_tests_operands = 10,
            nb_of_tests_consistency = 10,
            nb_of_tests_pycryptodome = 10):
        print("Testing read/write of operands...")
        for i in tqdm(range(nb_of_tests_operands)):
            key = self.random_16_bytes()
            self.write_key(key)
            if self.read_key() != key:
                raise ValueError("key mismatch")
            plaintext = self.random_16_bytes()
            self.write_plaintext(plaintext)
            if self.read_plaintext() != plaintext:
                raise ValueError("plaintext mismatch")
            ciphertext = self.random_16_bytes()
            self.write_ciphertext(ciphertext)
            if self.read_ciphertext() != ciphertext:
                raise ValueError("ciphertext mismatch")
        print("Testing whether AES decryption is the inverse of AES encryption...")
        for i in tqdm(range(nb_of_tests_consistency)):
            plaintext = self.random_16_bytes()
            self.write_plaintext(plaintext)
            self.encrypt()
            self.write_plaintext(self.random_16_bytes())
            self.decrypt()
            if self.read_plaintext() != plaintext:
                raise ValueError("Decryption is not the inverse of encryption")
        print("Testing whether the embedded AES matches with PyCryptodome AES...")
        for i in tqdm(range(nb_of_tests_pycryptodome)):
            key = self.random_16_bytes()
            plaintext = self.random_16_bytes()
            aes = AES.new(key, AES.MODE_ECB)
            ciphertext = aes.encrypt(plaintext)
            self.write_key(key)
            self.write_plaintext(plaintext)
            self.encrypt()
            if self.read_ciphertext() != ciphertext:
                raise ValueError("The embedded AES encryption does not match with the PyCryptodome implementation")
            key = self.random_16_bytes()
            ciphertext = self.random_16_bytes()
            aes = AES.new(key, AES.MODE_ECB)
            plaintext = aes.decrypt(ciphertext)
            self.write_key(key)
            self.write_ciphertext(ciphertext)
            self.decrypt()
            if self.read_plaintext() != plaintext:
                raise ValueError("The embedded AES decryption does not match with the PyCryptodome implementation")
