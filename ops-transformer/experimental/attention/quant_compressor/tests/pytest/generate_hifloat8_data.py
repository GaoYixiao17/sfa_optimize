import math
import struct
import time
from multiprocessing import Process, Queue
import numpy as np

def _get_binary_pos_str(f32_binary_str, start_pos, end_pos):
    if start_pos == end_pos:
        return f32_binary_str[start_pos]
    else:
        return f32_binary_str[start_pos:end_pos+1]

def _binary_to_int(binary_str):
    decimal_num = int(binary_str, 2)
    return decimal_num

def test_decode_fp32(input_float):
    sign_start_pos = 0
    sign_end_pos = 0
    exponent_start_pos = 1
    exponent_end_pos = 8
    fraction_start_pos = 9
    fraction_end_pos = 32

    float_binary = _float_to_binary(input_float)
    f_sign_bin_str = _get_binary_pos_str(float_binary, sign_start_pos, sign_end_pos)
    f_exponent_bin_str = _get_binary_pos_str(float_binary, exponent_start_pos, exponent_end_pos)
    f_fraction_bin_str = _get_binary_pos_str(float_binary, fraction_start_pos, fraction_end_pos)
    f_exponent_bin_int = _binary_to_int(f_exponent_bin_str)
    f_fraction_bin_int = _binary_to_int(f_fraction_bin_str)
    print(f"sign:{f_sign_bin_str} exponent:{f_exponent_bin_str} value:{f_exponent_bin_int - 127} fraction:{f_fraction_bin_str} value:{f_fraction_bin_int}")



def _float_to_binary(float_num):
    # 使用struct打包float_num为4字节，然后转换为无符号长整型
    packed = struct.pack('!f', float_num)
    # 将打包结果转换为32位二进制字符串
    return ''.join(f'{c:08b}' for c in packed)

def _binary_to_float(binary_str):
    #这个函数还有点问题，得更换
    # 将二进制字符串转换为bytes类型
    binary_bytes = bytes.fromhex(hex(int(binary_str, 2))[2:])
    # 使用struct模块中的unpack方法将bytes类型转换为float类型
    return struct.unpack('!f', binary_bytes)[0]

def _get_hif8_fraction_bits_number(exponent):
    #return dot value(4bits), exponent size, fraction size
    if exponent < -22:
        #zero
        return -1, 3, 0
    if -22 <= exponent < -15:
        #dml
        return 0, 3, 0
    if exponent == 0:
        #d0
        return 1, 0, 3
    if abs(exponent) == 1:
        #d1
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        #d2
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        #d3
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        #d4
        return 12, 4, 1
    if exponent > 15:
        #over flow
        return 12, 4, -1

def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    #fp32 fraction is 23,keep hif8_bits_num + 1 bits
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num+1))
    #print(f"fraction32_int:{format(fraction32_int, 'b')} hif8_value_tmp:{format(hif8_value_tmp, 'b')}")
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        #carry exponent
        return True, 0
    elif hif8_value_tmp == 0:
        #zero
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        #carrys bits
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1

def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = 16383  # b11111111111111
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192 #10 0000 0000 0000
        t14_values = fraction32_int & t14_mask
        hif8_value = 0

    else:
        #fp32 fraction is 23,keep hif8_bits_num bits
        #print(f"fraction32_int:{format(fraction32_int, 'b')} hif8_bits_num:{hif8_bits_num}")
        hif8_value = fraction32_int >> (23 - hif8_bits_num)
        #print(f"hif8_value:{hif8_value}")
        f14_t14 = fraction32_int - (hif8_value << (23 - hif8_bits_num))
        #print(f"f14_t14:{f14_t14}")
        #current do not deal hif8_bits_num > 9
        f14_values = f14_t14 >> (23 - hif8_bits_num - 14)
        t14_values = f14_t14 & t14_mask
    #print(f"f14_values:{f14_values} t14_values:{t14_values}")
    if f14_values >= t14_values:
        #carry bits
        if hif8_value == pow(2, hif8_bits_num) - 1:
            #carry exponent:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def _fp16_ta_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    #fp16 fraction is 10,keep hif8_bits_num + 1 bits
    hif8_value_tmp = fraction16_int >> (10 - (hif8_bits_num+1))
    #print(f"fraction16_int:{format(fraction16_int, 'b')} hif8_value_tmp:{format(hif8_value_tmp, 'b')}")
    if hif8_value_tmp == pow(2, hif8_bits_num+1) - 1:
        #carry exponent
        return True, 0
    elif hif8_value_tmp == 0:
        #zero
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        #carrys bits
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1

def _fp16_ssr_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
    t2_mask = 1 #b1
    t2_values = (fraction16_int & t2_mask) * 2 + 1
    if exponent == -23:
        f2_values = 2 + fraction16_int >> 9
        hif8_value = 0
    else:
        #fp16 fraction is 10,keep hif8_bits_num bits
        #print(f"fraction32_int:{format(fraction32_int, 'b')} hif8_bits_num:{hif8_bits_num}")
        hif8_value = fraction16_int >> (10 - hif8_bits_num)
        #print(f"hif8_value:{hif8_value}")
        f2_t2 = fraction16_int - (hif8_value << (10 - hif8_bits_num))
        #print(f"f2_t2:{f2_t2}")
        #current do not deal hif8_bits_num > 7
        f2_values = f2_t2 >> (10 - hif8_bits_num - 2)
    #print(f"f14_values:{f14_values} t14_values:{t14_values}")
    if f2_values >= t2_values:
        #carry bits
        if hif8_value == pow(2, hif8_bits_num):
            #carry exponent:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def cvt_float16_to_hifuint8(x, round_mode="round", over_mode=True):
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    if np.isinf(x) or x_abs >= over_value:
        #2^15 = 32768
        if sign:
            if over_mode:
                #b11101111 = 239
                return 239
            else:
                # b11101110 = 238
                return 238
        else:
            if over_mode:
                #b01101111 = 111
                return 111
            else:
                # b01101110 = 110
                return 110
    if np.isnan(x):
        if over_mode:
            #b10000000
            return 128
        else:
            return 0
    if x < 0.0:
        sign = True
        sign_int_value = 128
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))

    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"

    #precheck
    fraction_int = int(x_abs * pow(2, 10)*pow(2, -exponent) - pow(2, 10))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    #print(f"exponent:{exponent} fraction_int:{fraction_int} dot_hif8_value:{dot_hif8_value} exponent_hif8_bits:{exponent_hif8_bits} hif8_f_bits:{fraction_hif8_bits} cut_bit_type:{cut_bit_type}")
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp16_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp16_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        # print(f"unknow round type")
        return 0
    #print(f"hif8_frac_value:{hif8_frac_value}")
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = _get_hif8_fraction_bits_number(exponent)
        #print(f"after carry exponent:{exponent} dot_hif8_value:{dot_hif8_value} exponent_hif8_bits:{exponent_hif8_bits} hif8_f_bits:{fraction_hif8_bits_new} fraction_hif8_bits:{fraction_hif8_bits} fraction_hif8_bits_new:{fraction_hif8_bits_new}")
        #move_bits = fraction_hif8_bits - fraction_hif8_bits_new
        #hif8_frac_value = hif8_frac_value >> (fraction_hif8_bits - fraction_hif8_bits_new)
        fraction_hif8_bits = fraction_hif8_bits_new
    if fraction_hif8_bits == -1:
        #over flow
        if sign:
            if over_mode:
                return 239
            else:
                return 238
        else:
            if over_mode:
                return 111
            else:
                return 110
    if exponent < -23:
        #zero b00000000
        #print("zero")
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value == 0:
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        #d0
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
        #print(f"sign_int_value:{format(sign_int_value, 'b')} dot_int_value:{format(dot_int_value, 'b')} sig_exp:{format(sig_exp, 'b')} exponent_int_value:{format(exponent_int_value, 'b')} hif8_frac_value:{format(hif8_frac_value, 'b')}")
    return hif8_int_value

def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    if x < 0.0:
        sign = True
        sign_int_value = 128
    if np.isinf(x) or x_abs >= over_value:
        #2^15 = 32768
        if sign:
            if over_mode:
                #b11101111 = 239
                return 239
            else:
                # b11101110 = 238
                return 238
        else:
            if over_mode:
                #b01101111 = 111
                return 111
            else:
                # b01101110 = 110
                return 110
    if np.isnan(x):
        if over_mode:
            #b10000000
            return 128
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    #precheck
    fraction_int = int(x_abs * pow(2, 23)*pow(2, -exponent) - pow(2, 23))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    #print(f"exponent:{exponent} fraction_int:{fraction_int} dot_hif8_value:{dot_hif8_value} exponent_hif8_bits:{exponent_hif8_bits} hif8_f_bits:{fraction_hif8_bits} cut_bit_type:{cut_bit_type}")
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknow round type")
        return 0
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = _get_hif8_fraction_bits_number(exponent)
        #print(f"after carry exponent:{exponent} dot_hif8_value:{dot_hif8_value} exponent_hif8_bits:{exponent_hif8_bits} hif8_f_bits:{fraction_hif8_bits_new}")
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < -23:
        #zero b00000000
        #print("zero")
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        #dml
        #print(f"dml exponet:{exponent}")
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        #d0
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
        #print(f"sign_int_value:{format(sign_int_value, 'b')} dot_int_value:{format(dot_int_value, 'b')} sig_exp:{format(sig_exp, 'b')} exponent_int_value:{format(exponent_int_value, 'b')} hif8_frac_value:{format(hif8_frac_value, 'b')}")
    return hif8_int_value

def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120 #b01111000 = 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            #b1100 =12 D4
            exponet = x & 30 #b00011110 = 30
            exponet_int = exponet >> 1
            if exponet_int >= 8:
                #b1000 = 8
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 8

            fra_int = x & 1 #b00000001
            m_value = 1.0 + fra_int * 0.5
            #print(f"dot_4_value:{dot_4_value} exponet_value:{exponet_value} fra_int:{fra_int} value:{sign * pow(2, exponet_value) * (1 + fra_int / pow(2, 1))}")
        elif dot_4_value >= 8:
            #b1000 =8 D3
            exponet = x & 28 #b00011100 = 28
            exponet_int = exponet >> 2
            if exponet_int >= 4:
                #b100 = 4
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 4
            fra_int = x & 3 #b00000011
            m_value = 1.0 + fra_int * 0.25
            #print(f"dot_4_value:{dot_4_value} exponet_int:{exponet_int} exponet_value:{exponet_value} fra_int:{fra_int}")
        elif dot_4_value >= 4:
            #b0100 =8 D2
            exponet = x & 24  # b00011000 = 24
            exponet_int = exponet >> 3
            if exponet_int >= 2:
                # b10 = 2
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 2
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            #b0010 =2 D1
            exponet = x & 8 # b00001000 = 8
            exponet_sign = exponet >> 3
            if exponet_sign >= 1:
                # b10 = 2
                exponet_value = -1
            else:
                exponet_value = 1
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
            #print(f"dot_4_value:{dot_4_value} exponet_value:{exponet_value} fra_int:{fra_int} value:{sign * pow(2, exponet_value) * (1 + fra_int / pow(2, 3))}")
        elif dot_4_value == 1:
            #d0
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
            #print(f"dot_4_value:{dot_4_value} exponet_value:{exponet_value} fra_int:{fra_int} value:{sign*pow(2,exponet_value)*(1+fra_int/pow(2,3))}")
        elif dot_4_value == 0:
            #dml
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
            #print(f"dml exponet_value:{x & 7 - 23} x:{x}")
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign*pow(2.0, exponet_value)*m_value


def worker_cvt_fp32_to_hif8(queuein, queueout):
    while True:
        in_data = queuein.get()
        if in_data is None:
            break
        index = in_data["index"]
        value = in_data["value"]
        out_data = {}
        out_data["index"] = index
        out_data["value"] = cvt_float32_to_hifuint8(value)
        queueout.put(out_data)

def trans_np_float_tensor_to_hifint8_mp(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    print(f"total size:{multi_shape}")
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    processes_num = 4
    queue_in = Queue()
    queue_out = Queue()
    processes = []
    for i in range(processes_num):
        p = Process(target=worker_cvt_fp32_to_hif8, args=(queue_in, queue_out))
        p.start()
        processes.append(p)
    for i in range(multi_shape):
        data = {}
        data["index"] = i
        data["value"] = in_tensor[i]
        queue_in.put(data)

    for i in range(len(processes)):
        queue_in.put(None)

    count = 0
    while count < multi_shape:
        if not queue_out.empty():
            data_out = queue_out.get()
            out_tensor[data_out["index"]] = data_out["value"]
            count += 1
        else:
            continue

    for p in processes:
        p.join()

    out_tensor = out_tensor.reshape(shape_tensor).astype(np.int8)
    return out_tensor


def trans_np_float_tensor_to_hifuint8(in_tensor,round_mode="round", over_mode=True):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    if multi_shape == 1.0: # shape为空时，输出multi_shape可能为浮点数 1.0
        multi_shape = int(multi_shape)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    if in_tensor.dtype == np.float32:
        print(f"[trans_np_float_tensor_to_hifuint8]float32->hif8 total size :{multi_shape}")
        for i in range(multi_shape):
            out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    else:
        print(f"[trans_np_float_tensor_to_hifuint8]float16->hif8 total size :{multi_shape}")
        for i in range(multi_shape):
            out_tensor[i] = cvt_float16_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.astype(np.uint8)
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor

def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    print(f"[trans_np_hifuint8_tensor_to_float32]total size:{multi_shape}")
    out_tensor = np.zeros(multi_shape).astype(np.float32)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_hifuint8_to_float(in_tensor[i])
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor






if __name__ == '__main__':
    #shape = [8, 32, 64, 64]
    #in_tensor = np.fromfile("cast_07_dyn_input_0.bin", dtype=np.float32).reshape(shape)
    #start_time = time.time()
    #out_tensor = trans_np_float_tensor_to_hifint8_mp(in_tensor)
    #out_tensor = trans_np_float_tensor_to_hifuint8(in_tensor)
    #end_time = time.time()
    #print(end_time - start_time)
    #out_tensor.tofile("cast_07_dyn_output_0_hif8.bin")


    #print(_binary_to_float("01011110000000001101100101011001"))
    #print(_binary_to_float("01011110000000001101100101011000"))

    #x = np.array([16],dtype=np.float16)
    #y = trans_np_float_tensor_to_hifuint8(x)
    #print(y)
    #z = trans_np_hifuint8_tensor_to_float32(y)
    #print(z)
    #y = x.astype("float8_e5m2")
    #print(y)

    #x = 1*pow(2,-23)*(1+pow(2,1)/pow(2,3))
    x = -1.45623773e-07
    #x_hif8 = cvt_float16_to_hifuint8(np.float16(x))
    print(f"x:{x} bin:{_float_to_binary(x)} type:{type(x)}")
    test_decode_fp32(x)
    #x_hif8 = cvt_float32_to_hifuint8(x,round_mode="hybrid")
    #x_hif8 = cvt_float32_to_hifuint8(x, round_mode="round")
    x_hif8 = cvt_float32_to_hifuint8(x, round_mode="storound")
    #x_hif8 = cvt_float32_to_hifuint8(x)
    #x_hif8 = cvt_float32_to_hifuint8(x, round_mode="round")
    x_hif8_b_str = '{:08b}'.format(x_hif8)
    print(f"x_hif8:{x_hif8} bin:{x_hif8_b_str}")
    #x_float32 = cvt_hifuint8_to_float(x_hif8)
    #print(f"x_f32:{x_float32} bin:{_float_to_binary(x_float32)}")

    #import numpy as np

    # 创建一个包含随机整数的numpy数组
    #arr = np.random.randint(-10, 10, 10)
    #print("原始数组:", arr)

    # 使用布尔索引来过滤出正数
    #positive_numbers = arr[arr == 1]
    #print("正数数组:", positive_numbers)


    #shape = (2, 32, 100,128)
    #dtype = np.float32
    #in_tensor = np.random.uniform(-1, 1, size=shape).astype(dtype)

    #start_time = time.time()
    #out_tensor = trans_np_float_tensor_to_hifuint8(in_tensor)
    #end_time = time.time()
    #print(end_time - start_time)


    #start_time = time.time()
    #out_tensor2 = trans_np_hifuint8_tensor_to_float32(out_tensor)
    #end_time = time.time()
    #print(end_time - start_time)

    #start_time = time.time()
    #out_tensor2 = trans_np_hifint8_tensor_to_float16(out_tensor)
    #end_time = time.time()
    #print(end_time - start_time)













