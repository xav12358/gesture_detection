import sys
import csv
from array import array
import struct


for arg in sys.argv: 
    print(arg)

csv_input_file = sys.argv[1]

with open("train_input.bin", "wb") as binary_file:
    float_array = []
    with open(csv_input_file, 'r') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in spamreader:
            print('[ {} ]'.format(row))
            print("size {}".format(len(row)))
            for data in row:
                float_array.append(float(data))

        # print(">>> {}".format(float_array))
        # newfile.write(bytearray(float_array))
        # float_array.astype('float').tofile(binary_file)
        # for datafloat in float_array:
        #     binary_file.write(struct.pack(datafloat))
    print("size {}".format(len(float_array)))
    b = bytes()
    b = b.join((struct.pack('f', val) for val in float_array))
    binary_file.write(b)