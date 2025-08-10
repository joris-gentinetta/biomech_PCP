import socket
import threading
import struct
import zmq


class BCI_Data_Receiver(object):
    # def __init__(self, ip, port, gazeboIP, gazeboPort):
    def __init__(self, ip, port, recieveRate):
        self.receiveRate = recieveRate  # Hz
        self.receiveCount = 0

        self.ip = ip
        self.port = port
        self.sock = None
        self.receiveBuff = bytes()
        self.dataBuff = []
        self.dataReadyCallback = None
        self.readingThread = None
        self.address = (self.ip, self.port)

        self.recvType = "USB"  # "WIFI"

        # self.gazeboIP = gazeboIP
        # self.gazeboPort = gazeboPort
        # self.gazeboAddress = (self.gazeboIP, self.gazeboPort)

        self.socketAddr = "tcp://127.0.0.1:1235"
        self.ctx = zmq.Context()
        self.sendingSock = self.ctx.socket(zmq.PUB)
        self.sendingSock.bind(self.socketAddr)

        self.packet_size = 80
        self.packet_num = 15 if self.recvType == "WIFI" else 15
        self.EMG_channel_num = 16
        self.Data_channel_num = 4

    def __del__(self):
        try:
            # close the socket
            self.sendingSock.close()
            self.ctx.term()
        except:
            print("__del__: Socket closing error")

    def startConnection(self):
        """Start the socket connection"""
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # For UDP
        if self.recvType == "WIFI":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(self.address)  # this needs to bind not connect
            # self.sock.connect(("192.168.1.10",35294))
            self.sock.settimeout(10.0)

        elif self.recvType == "USB":
            self.sock = self.ctx.socket(zmq.SUB)
            self.sock.connect("tcp://" + self.ip + ":" + str(self.port))
            self.sock.subscribe("")

        print("Processing incoming data")
        self.processStream()

    def asyncReceiveData(self, dataReadyCallback):
        """Receive the data from ADS1299 asynchronously."""
        if self.readingThread == None:
            self.dataReadyCallback = dataReadyCallback
            self.readingThread = threading.Thread(target=self.startConnection)
            self.readingThread.start()
        else:
            raise Exception("The reading thread is already running!")

    ## Added by Mikey 02/05/21 to allow feeding EMG data to Haptix Gazebo
    def bypassToController(self, data, frameRate):
        """
        Bypass the received data to the virtual limb or hardware for controlling the prosthesis
        """
        sendThreshold = self.receiveRate / frameRate
        self.receiveCount += 1
        if self.receiveCount >= sendThreshold:
            self.receiveCount = 0
            # packed struct, 92 bytes
            # f: time (4 bytes)
            # f: time (4 bytes)
            # f[16]: float EMG signals (floats - 4 bytes each)
            # I: trigger (4 bytes)
            # I: switch 1 (4 bytes)
            # I: switch 2 (4 bytes)
            # I: newline (4 bytes)
            # f: sending frequency (float - 4 bytes)
            byte_data = struct.pack(
                "ffffffffffffffffffIIIIf",
                data[4],
                data[5],
                data[7],
                data[8],
                data[9],
                data[10],
                data[11],
                data[12],
                data[13],
                data[14],
                data[15],
                data[16],
                data[17],
                data[18],
                data[19],
                data[20],
                data[21],
                data[22],
                data[23],
                data[24],
                data[25],
                data[26],
                frameRate,
            )
            # print("%d bytes sent: " % len(byte_data), byte_data, "\n")

            self.sendingSock.send(byte_data)
            print(f"{data[4] / 1e6:0.3f}")

            # self.printData(byte_data=byte_data, data=data, frameRate=frameRate)

    def printData(self, byte_data, data, frameRate):
        print(f"""{len(byte_data)} bytes sent to {self.socketAddr}\nTime: {data[4]:07.3f} sec\nFrequency: {frameRate:07.3f} Hz
            {data[7]:07.2f} {data[8]:07.2f} {data[9]:07.2f} {data[10]:07.2f}
            {data[11]:07.2f} {data[12]:07.2f} {data[13]:07.2f} {data[14]:07.2f}
            {data[15]:07.2f} {data[16]:07.2f} {data[17]:07.2f} {data[18]:07.2f}
            {data[19]:07.2f} {data[20]:07.2f} {data[21]:07.2f} {data[22]:07.2f}\n""")

    def processStream(self):
        while True:
            receiveData = []
            for i in range(self.EMG_channel_num + self.Data_channel_num):
                receiveData.append([])

            # Receive data from sensor
            if self.recvType == "WIFI":
                data, _ = self.sock.recvfrom(self.packet_num * self.packet_size)
                self.receiveBuff = self.receiveBuff + data
            else:
                while len(self.receiveBuff) < self.packet_num * self.packet_size:
                    newData = self.sock.recv()
                    self.receiveBuff = self.receiveBuff + newData

            # print(data, addr)
            # self.receiveBuff = self.receiveBuff + self.sock.recv(40)

            if len(self.receiveBuff) >= self.packet_num * self.packet_size:
                data = self.receiveBuff[0 : self.packet_num * self.packet_size]
                self.receiveBuff = self.receiveBuff[
                    self.packet_num * self.packet_size :
                ]
                for i in range(self.packet_num):
                    # For BrainCo ADS1299
                    # unpacked_data = struct.unpack('qiiiiiiii', data[i*40: (i+1)*40])
                    # For FluidBCInewData
                    # unpacked_data = struct.unpack('iiiiiiiiii', data[i*40: (i+1)*40])
                    # For Foc.us BCI
                    # unpacked_data = struct.unpack('iiffffffff', data[i*40: (i+1)*40])

                    # For Biomech EMG
                    unpacked_data = struct.unpack(
                        "BBBBIHHffffffffffffffffBBBB",
                        data[i * self.packet_size : (i + 1) * self.packet_size],
                    )
                    # print(unpacked_data[2],unpacked_data[21],unpacked_data[22],unpacked_data[4])
                    for channel_index in range(self.EMG_channel_num):
                        receiveData[channel_index].append(
                            unpacked_data[7 + channel_index]
                        )
                    # For Timestamp
                    receiveData[self.EMG_channel_num + 3].append(unpacked_data[4])
                    # print(unpacked_data[4])
                    # For switchs
                    receiveData[self.EMG_channel_num + 0].append(unpacked_data[1])
                    receiveData[self.EMG_channel_num + 1].append(unpacked_data[24])
                    receiveData[self.EMG_channel_num + 2].append(unpacked_data[25])

                    ## Added by Mikey 02/05/21 to allow feeding EMG to Haptix Gazebo
                    # Pass data to use for EMG control
                    self.bypassToController(unpacked_data, self.receiveRate)

                # self.dataReadyCallback(receiveData)
