import numpy as np
import itertools
import math
from scipy.sparse import csgraph

class APAR:
    #根据mesh初始化,计算所有的weight
    def read_mesh(self,vertex,triangle):
        self.vertex = vertex
        self.triangle = triangle
        self.point_num = len(self.vertex)
        self.vertexPrime = np.array(self.vertex)
        self.vertexPrime = np.asmatrix(self.vertexPrime)
        self.vertex2Tri = [[j for j,tri in enumerate(self.triangle) if i in tri] for i in range(self.point_num)]
        self.neighbor = np.zeros((self.point_num,self.point_num))
        self.neighbor[
            tuple(zip(
                *itertools.chain(
                    *map(
                        lambda tri: itertools.permutations(tri,2),
                        self.triangle
                    )
                )
            ))]=1
        self.cellRotaion = np.zeros((self.point_num,3,3))
        self.weight = np.zeros((self.point_num,self.point_num),dtype=np.float)

        for vert in range(self.point_num):
            neighbor = self.findNeighbor(vert)
            for neivert in neighbor:
                self.weightinit(vert,neivert)

    def findNeighbor(self,vert):
        return np.where(self.neighbor[vert]==1)[0]

    def weightinit(self,verti,vertj):
        def calWeight(verti,vertj):
            external_points=[]
            for tri in self.vertex2Tri[verti]:
                triVert = self.triangle[tri]
                if verti in triVert and vertj in triVert:
                    for Vertid in triVert:
                        if Vertid != verti and Vertid !=vertj:
                            external_points.append(Vertid)
            posi = np.array(self.vertex[verti])
            posj = np.array(self.vertex[vertj])
            cot_weight = 0
            for other_point in external_points:
                other_pos = np.array(self.vertex[other_point])
                vA = posi - other_pos
                vB = posj - other_pos
                Cos_value = np.dot(vA,vB)/(np.linalg.norm(vA)*np.linalg.norm(vB))
                theta = math.acos(Cos_value)
                cot_weight += math.cos(theta)/math.sin(theta)
            return cot_weight*0.5

        if self.weight[vertj,verti] == 0:
            weight = calWeight(verti,vertj)
        else:
            weight = self.weight[vertj,verti]
        self.weight[verti,vertj] = weight

    def compute_P(self):
        self.PArray = []
        for verti in range(self.point_num):
            posi =np.array(self.vertex[verti])
            neighbor = self.findNeighbor(verti)
            Pi = np.zeros((3,len(neighbor)))
            for ni in range(len(neighbor)):
                neighborId = neighbor[ni]
                posj = np.array(self.vertex[neighborId])
                Pi[:,ni] = posi-posj
            self.PArray.append(Pi)

    def model_constrain(self,fixId,handleId,deformationMatrix):
        #加入constrain
        deformationMatrix_list = list(map(np.matrix,deformationMatrix))
        self.deformedVerts=[]
        for vert in range(self.point_num):
            if vert in fixId:
                #固定点
                self.deformedVerts.append((vert,self.vertex[vert]))
            elif vert in handleId:
                #先经过deformation matrix
                deformedVert = np.append(self.vertex[vert],1)
                deformedVert = deformedVert.dot(deformationMatrix_list[handleId.index(vert)])
                deformedVert = np.delete(deformedVert,3).flatten()
                deformedVert = np.squeeze(np.asarray(deformedVert))
                self.deformedVerts.append((vert,deformedVert))

        #计算laplacian
        process_num = len(self.deformedVerts)
        self.laplacian = np.zeros((self.point_num+process_num,self.point_num+process_num),dtype=np.float32)
        self.laplacian[:self.point_num,:self.point_num] = csgraph.laplacian(self.weight)
        for i in range(process_num):
            vert = self.deformedVerts[i][0]
            newi = i + self.point_num
            self.laplacian[newi,vert] = 1
            self.laplacian[vert,newi] = 1

        process_num = len(self.deformedVerts)
        self.bArray = np.zeros((self.point_num + process_num, 3))
        for i in range(process_num):
            self.bArray[self.point_num + i] = self.deformedVerts[i][1]
        self.compute_P()

    def process(self, iterations):
        #迭代过程
        for t in range(iterations):
            self.fix_Rotation()
            self.fix_Vertex()

    def fix_Rotation(self):
        for vert in range(self.point_num):
            rotationMatrix = self.computeRoationMatrix(vert)
            self.cellRotaion[vert]=rotationMatrix

    def computeRoationMatrix(self,vert):
        covarianceMatrix = self.computeCovarianceMatrix(vert)
        U,S,V = np.linalg.svd(covarianceMatrix)
        rotation_matrix = V.T.dot(U.T)
        return rotation_matrix

    def computeCovarianceMatrix(self,vert):
        vertPrime = self.vertexPrime[vert]
        neighbor = self.findNeighbor(vert)
        neighborNum = len(neighbor)

        Di = np.zeros((neighborNum,neighborNum))

        P = self.PArray[vert]
        PPrime = np.zeros((3,neighborNum))

        for ni in range(neighborNum):
            nVert = neighbor[ni]
            Di[ni,ni] = self.weight[vert,nVert]
            nvertPrime = self.vertexPrime[nVert]
            PPrime[:,ni] = vertPrime - nvertPrime
        PPrime = PPrime.T

        return P.dot(Di).dot(PPrime)

    def fix_Vertex(self):
        for vert in range(self.point_num):
            self.bArray[vert] = np.zeros((1,3))
            neighbor = self.findNeighbor(vert)
            for nVert in neighbor:
                w = self.weight[vert,nVert] / 2
                r = self.cellRotaion[vert] + self.cellRotaion[nVert]
                p = np.array(self.vertex[vert])-np.array(self.vertex[nVert])
                self.bArray[vert]+=(w*r.dot(p))

        self.vertexPrime = np.linalg.solve(self.laplacian,self.bArray)[:self.point_num]
