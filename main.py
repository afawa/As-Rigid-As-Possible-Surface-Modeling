from arap_deformer import APAR
from off_processer import OffProcesser
import numpy as np
if __name__ == '__main__':
    processer = OffProcesser()
    vertex, tri = processer.read_offFile('./Meshes/armadillo_1k.off')
    deformer = APAR()
    deformer.read_mesh(vertex,tri)
    deformer.model_constrain([],range(200,203),np.stack([np.random.rand(4,4)]*3))
    deformer.process(100)
    processer.save_offFile(deformer.vertexPrime,deformer.triangle,'test.off')
