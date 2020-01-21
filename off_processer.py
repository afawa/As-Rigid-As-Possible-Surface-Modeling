import numpy as np

class OffProcesser:
    def read_offFile(self,path):
        #读取mesh文件,输出点坐标和三角标号
        with open(path,'r') as offFile:
            offFile.readline()
            firstline = offFile.readline().split()
            vertex_num = int(firstline[0])
            tris_num = int(firstline[1])
            vertex = np.zeros((vertex_num,3))
            tri = np.zeros((tris_num,3),dtype=np.int)
            for i in range(vertex_num):
                line = offFile.readline().split()
                vertex[i,0] = float(line[0])
                vertex[i,1] = float(line[1])
                vertex[i,2] = float(line[2])
            for i in range(tris_num):
                line = offFile.readline().split()
                tri[i,0] = int(line[1])
                tri[i,1] = int(line[2])
                tri[i,2] = int(line[3])

        return vertex,tri

    def save_offFile(self,vertex,tri,path):
        #保存mesh为off
        with open(path,'w') as offFile:
            offFile.write('OFF\n')
            l = []
            l.append(str(len(vertex)))
            l.append(str(len(tri)))
            l.append(str(0))
            offFile.write(' '.join(l)+'\n')
            for vert in vertex:
                l = []
                l.append(str(vert[0]))
                l.append(str(vert[1]))
                l.append(str(vert[2]))
                offFile.write(' '.join(l)+'\n')
            for T in tri:
                l=[str(3)]
                l.append(str(T[0]))
                l.append(str(T[1]))
                l.append(str(T[2]))
                offFile.write(' '.join(l)+'\n')
if __name__=='__main__':
    processer = OffProcesser()
    vertex,tri= processer.read_offFile('./Meshes/dino.off')
    processer.save_offFile(vertex,tri,'test.off')