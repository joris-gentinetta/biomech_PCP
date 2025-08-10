# Mikey Fernandez
# 12/08/2022
# Constant variable definitions for use in EMG USB interface

import numpy as np

"""Data definitions"""
ACTIVE_FRQSC = 1

WIFI_STATUS = 1  #  0: off, 1: on
VISUALIZATION_STATUS = 0  #  0: off, 1: on

EMG_INIT_DELAY = 3000

# Comm port type - buf[1]
COMM_COMMAND_CONFIG_USB = 0x50
COMM_COMMAND_CONFIG_UART = 0x51
COMM_COMMAND_CONFIG_I2C = 0x52
COMM_COMMAND_CONFIG_SPI = 0x53

# Communication on/off
COMM_TX_INACTIVE = 0x0A  # stop stream
COMM_TX_DONE = 0x93  # start stream

# Definition of the datatype
DSP_DATA_RAW = 0x00
DSP_DATA_BPF = 0x01
DSP_DATA_BPFSQ = 0x02

DSP_N_SERIESDATA = 3

DSP_DATA_ENVF = 0x03
DSP_DATA_ACT = 0x04
DSP_DATA_MAX = 0x05
DSP_DATA_MIN = 0x06
DSP_DATA_AUX1 = 0x07
DSP_DATA_AUX2 = 0x08
DSP_DATA_AUX3 = 0x09
DSP_DATA_THMAX = 0x0A
DSP_DATA_THMIN = 0x0B
DSP_DATA_GYROACC = 0x0C
DSP_DATA_RAW8CH_GYROACC = 0x0D
DSP_DATA_FEMGPWR1 = 0x0E  #  TEST
DSP_DATA_FEMGPWR2 = 0x0F  #  TEST
DSP_DATA_FEMGPWR_RAW = 0x10  #  TEST

# buffer type
BUFFER_FULL = 0x00  # full buffer (16ch float): 80byte
BUFFER_HALF = 0x01  # half buffer (16ch int16_t): 40byte
BUFFER_64BYTE = 0x02  # 3/4 buffer (16ch int16_t): 64byte
BUFFER_8CH = 0x03  # half buffer (8ch int16_t): 24 byte
BUFFER_160 = 0x04  # full buffer (16ch float): 160byte - for BeagleBone DMA

BUFFER_FULL_LENGTH = 80
BUFFER_HALF_LENGTH = 40

# #RECEIVER_IP "192.168.50.234" #  JQ
# RECEIVER_IP = "192.168.50.19" #  TSHU
RECEIVER_IP = "192.168.1.3"  # Mikey

RECEIVER_PORT = 8899

WIFI_SPI_CLK = 10000000
WIFI_PACKET_LENGTH = 80

WIFI_SSID = "biomech-robot-2.4G"
WIFI_PASSWORD = "2020leglab"

""" Settings """
ACTIVE_DATA = DSP_DATA_RAW
USB_PACKET_LENGTH = BUFFER_FULL_LENGTH

""" Clasee definitions """


class emgDataHalf:
    def __init__(self):
        self.idx = 0
        self.dataType = 0
        self.emg_os_tick = 0
        self.syncButton = 0
        self.buttonCounter = 0
        self.sw1 = 0
        self.sw2 = 0

        self.dataBuf = np.zeros(16)

        ## the below isn't memcpy'd

        self.first_timestamp = 0
        self.last_timestamp = 0
        self.average_timestamp = 0
        self.cnt = 0
        self.emg_cnt = 0
        self.emg_initCnt = 0
        self.event_time = 0

        self.format = "BBHBBBBHHHHHHHHHHHHHHHH"
        self.fullFormat = "BBHBBBBHHHHHHHHHHHHHHHHIIIIHHQ"

        # uint8_t idx;
        # uint8_t dataType;
        # uint16_t emg_os_tick;
        # uint8_t syncButton;
        # uint8_t buttonCounter;
        # uint8_t sw1;
        # uint8_t sw2;

        # int16_t databuf[16];

        # uint32_t first_timestamp;
        # uint32_t last_timestamp;
        # uint32_t average_timestamp;

        # uint32_t cnt;
        # uint16_t emg_cnt;
        # uint16_t emg_initCnt;

        # uint64_t event_time;

    def emgDataPrint(self):
        print(f"""Time: {self.emg_os_tick:07.3f} ms\n
            {self.dataBuf[0]:07.2f} {self.dataBuf[4]:07.2f} {self.dataBuf[8]:07.2f} {self.dataBuf[12]:07.2f}
            {self.dataBuf[1]:07.2f} {self.dataBuf[5]:07.2f} {self.dataBuf[9]:07.2f} {self.dataBuf[13]:07.2f}
            {self.dataBuf[2]:07.2f} {self.dataBuf[6]:07.2f} {self.dataBuf[10]:07.2f} {self.dataBuf[14]:07.2f}
            {self.dataBuf[3]:07.2f} {self.dataBuf[7]:07.2f} {self.dataBuf[11]:07.2f} {self.dataBuf[15]:07.2f}\n""")


class emgDataFull:
    def __init__(self):
        self.startCode = 0
        self.gpioState = 0
        self.dataType = 0
        self.freqScalar = 0
        self.osTime_us = 0
        self.osTime_ms = 0
        self.processingTime = 0

        self.dataBuf = np.zeros(16)

        self.gpioTimedState = 0
        self.sw1 = 0
        self.sw2 = 0
        self.newline = 0

        ## the below isnt memcpy'd

        self.first_timestamp = 0
        self.last_timestamp = 0
        self.average_timestamp = 0

        self.cnt = 0
        self.emg_cnt = 0
        self.emg_initCnt = 0

        self.event_time = 0

        self.format = "BBBBIHHffffffffffffffffBBBB"
        self.fullFormat = "BBBBIHHffffffffffffffffBBBBIIIIHHQ"

        # uint8_t startCode;
        # uint8_t gpioState;
        # uint8_t dataType;
        # uint8_t freqScalar;
        # uint32_t osTime_us;
        # uint16_t osTime_ms;
        # uint16_t processingTime;
        # float databuf[16];
        # uint8_t gpioTimedState;
        # uint8_t sw1;
        # uint8_t sw2;
        # uint8_t newline;

        # uint32_t first_timestamp;
        # uint32_t last_timestamp;
        # uint32_t average_timestamp;

        # uint32_t cnt;
        # uint16_t emg_cnt;
        # uint16_t emg_initCnt;

        # uint64_t event_time;

    def emgDataFullPrint(self):
        print(f"""Time: {self.osTime_ms:07.3f} ms\n
            {self.dataBuf[0]:07.2f} {self.dataBuf[4]:07.2f} {self.dataBuf[8]:07.2f} {self.dataBuf[12]:07.2f}
            {self.dataBuf[1]:07.2f} {self.dataBuf[5]:07.2f} {self.dataBuf[9]:07.2f} {self.dataBuf[13]:07.2f}
            {self.dataBuf[2]:07.2f} {self.dataBuf[6]:07.2f} {self.dataBuf[10]:07.2f} {self.dataBuf[14]:07.2f}
            {self.dataBuf[3]:07.2f} {self.dataBuf[7]:07.2f} {self.dataBuf[11]:07.2f} {self.dataBuf[15]:07.2f}\n""")
