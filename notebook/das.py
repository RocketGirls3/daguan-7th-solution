from math import fabs
from math import sqrt
from math import atan2
import math

num = int(input())
pos = input()
zimanum = int(input())
zimapos = input()

zimapos_ = []
zimapos = zimapos.split(',')
for i in range(zimanum):
    zimapos_.append([float(zimapos[i*2]),float(zimapos[i*2+1])])

pos = pos.split(',')
# zimapos = [float(i) for i in zimapos]
pos_ = []
for i in range(num):
    pos_.append([float(pos[i*2]),float(pos[i*2+1])])
# print(zimapos_)

def is_in_2d_polygon(point, vertices):
    px = point[0]
    py = point[1]
    angle_sum = 0

    size = len(vertices)
    if size < 3:
        raise ValueError("len of vertices < 3")
    j = size - 1
    for i in range(0, size):
        sx = vertices[i][0]
        sy = vertices[i][1]
        tx = vertices[j][0]
        ty = vertices[j][1]

        # 通过判断点到通过两点的直线的距离是否为0来判断点是否在边上
        # y = kx + b, -y + kx + b = 0
        k = (sy - ty) / (sx - tx + 0.000000000001)  # 避免除0
        b = sy - k * sx
        dis = fabs(k * px - 1 * py + b) / sqrt(k * k + 1)
        if dis < 0.000001:  # 该点在直线上
            if sx <= px <= tx or tx <= px <= sx:  # 该点在边的两个定点之间，说明顶点在边上
                return True

        # 计算夹角
        angle = atan2(sy - py, sx - px) - atan2(ty - py, tx - px)
        # angle需要在-π到π之内
        if angle >= math.pi:
            angle = angle - math.pi * 2
        elif angle <= -math.pi:
            angle = angle + math.pi * 2

        # 累积
        angle_sum += angle
        j = i

    # 计算夹角和于2*pi之差，若小于一个非常小的数，就认为相等
    return fabs(angle_sum - math.pi * 2) < 0.00000000001
# print(pos_)
s = 0
result = []
for i in zimapos_:
    if is_in_2d_polygon(i, pos_):
        result.append(i)
        s+=1
    else:
        continue
output = ""
if s == 0:
    print("No")
else:
    for i in result:
        output+="("
        output+=str(i[0])
        output += ","
        output += str(i[1])
        output += ")"
print(output)
        # print("("+i[0]+i[1]+")")
# print(is_in_2d_polygon(pos, zimapos_))
