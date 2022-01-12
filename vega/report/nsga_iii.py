# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NSGA-III selection algorithm."""
import numpy as np
from scipy.linalg import solve


def UpdateIdealPoint(pop):
    """Update ideal point.

    :param pop: the current population
    :type pop: array
    """
    zmin = pop.min(1, keepdims=True)
    return zmin


def PerformScalarizing(pop):
    """Scaling the population.

    :param pop: the current population
    :type pop: array
    """

    def GetScalarizingVector(nobj, j):
        epsilon = 1e-10
        w = epsilon * np.ones((nobj))
        w[j] = 1
        return w

    nobj, npop = pop.shape
    zmax = np.zeros((nobj, nobj))
    smin = np.ones(nobj) * np.inf
    for j in range(nobj):
        w = GetScalarizingVector(nobj, j)
        s = np.zeros((npop))
        for i in range(npop):
            s[i] = max(pop[:, i] / w)
        smin[j] = min(s)
        zmax[:, j] = pop[:, np.argmin(s)]
    return smin, zmax


def NormalizePopulation(pop):
    """Normalize the population.

    :param pop: the current population
    :type pop: array
    """

    def FindHyperplaneIntercepts(zmax):
        target = np.ones(zmax.shape[1])
        if np.linalg.matrix_rank(zmax):
            w = solve(zmax, target)
            a = (1 / w)
        else:
            a = target
        return a

    _, npop = pop.shape
    pop_norm = pop.copy()
    zmin = UpdateIdealPoint(pop_norm)
    pop_norm = pop_norm - zmin
    _, zmax = PerformScalarizing(pop_norm)
    a = FindHyperplaneIntercepts(zmax)
    for i in range(npop):
        pop_norm[:, i] = pop_norm[:, i] / a
    return pop_norm


def Dominates(x, y):
    """Check if x dominates y.

    :param x: a sample
    :type x: array
    :param y: a sample
    :type y: array
    """
    return np.all(x <= y) & np.any(x < y)


def NonDominatedSorting(pop):
    """Perform non-dominated sorting.

    :param pop: the current population
    :type pop: array
    """
    _, npop = pop.shape
    rank = np.zeros(npop)
    dominatedCount = np.zeros(npop)
    dominatedSet = [[] for i in range(npop)]
    F = [[]]
    for i in range(npop):
        for j in range(i + 1, npop):
            p = pop[:, i]
            q = pop[:, j]
            if Dominates(p, q):
                dominatedSet[i].append(j)
                dominatedCount[j] += 1
            if Dominates(q, p):
                dominatedSet[j].append(i)
                dominatedCount[i] += 1
        if dominatedCount[i] == 0:
            rank[i] = 1
            F[0].append(i)
    k = 0
    while (True):
        Q = []
        for i in F[k]:
            p = pop[:, i]
            for j in dominatedSet[i]:
                dominatedCount[j] -= 1
                if dominatedCount[j] == 0:
                    Q.append(j)
                    rank[j] = k + 1
        if len(Q) == 0:
            break
        F.append(Q)
        k += 1
    return F


def GenerateReferencePoint(M, p):
    """Generate reference point.

    :param M: number of population
    :type M: int
    :param p: number of reference points
    :type p: int
    """

    def GetFixedRowSumIntegerMatrix(M, RowSum):
        if M == 1:
            A = np.array([[RowSum]])
            return A
        A = np.zeros((0, M))
        for i in range(RowSum):
            B = GetFixedRowSumIntegerMatrix(M - 1, RowSum - i)
            A = np.vstack((A, np.hstack((i * np.ones((B.shape[0], 1)), B))))
        return A

    Zr = GetFixedRowSumIntegerMatrix(M, p)
    Zr = Zr.transpose() / p
    return Zr


def AssociateToReferencePoint(pop_norm):
    """Associate current population to reference points.

    :param pop_norm: the current population
    :type pop_norm: array
    """
    nZr = 10
    _, npop = pop_norm.shape
    Zr = GenerateReferencePoint(pop_norm.shape[0], nZr)
    rho = np.zeros((nZr))
    d = np.zeros((npop, nZr))
    pop_ref = np.zeros(npop)
    pop_dis = np.zeros(npop)
    for i in range(npop):
        for j in range(nZr):
            w = Zr[:, j] / np.linalg.norm(Zr[:, j]).reshape(-1, 1)
            z = pop_norm[:, i].reshape(-1, 1)
            d[i, j] = np.linalg.norm(z - w.transpose() * z * w)
        dmin = np.min(d[i])
        jmin = np.argmin(d[i])
        pop_ref[i] = jmin
        pop_dis[i] = dmin
        rho[jmin] += 1
    return rho, pop_ref, pop_dis


def SortAndSelectPopulation(pop, N):
    """NSGA-III selection.

    :param pop: the current population
    :type pop: arraySortAndSelectPopulation
    :param N: number of population
    :type N: int
    """
    selected = np.zeros(0)
    # pop: nobj * npop matrix
    nobj, _ = pop.shape
    pop_norm = NormalizePopulation(pop)
    F = NonDominatedSorting(pop)
    rho, pop_ref, _ = AssociateToReferencePoint(pop_norm)
    newpop = np.zeros((nobj, 0))
    lastFront = list()
    for i in range(len(F)):
        if newpop.shape[1] + len(F[i]) > N:
            lastFront = F[i]
            break
        newpop = np.hstack((newpop, pop[:, F[i]]))
        selected = np.hstack((selected, np.array(F[i])))

    if newpop.shape[1] < N:
        while (True):
            j = np.argmin(rho)
            AssocitedFromLastFront = []
            for i in lastFront:
                if pop_ref[i] == j:
                    AssocitedFromLastFront.append(i)
            if len(AssocitedFromLastFront) == 0:
                rho[j] = np.inf
                continue
            new_member_ind = np.random.choice(
                list(range(len(AssocitedFromLastFront))), 1)
            MemberToAdd = AssocitedFromLastFront[new_member_ind[0]]
            lastFront = [item for item in lastFront if item != MemberToAdd]
            newpop = np.hstack((newpop, pop[:, MemberToAdd].reshape(-1, 1)))
            selected = np.hstack((selected, MemberToAdd))
            rho[j] = rho[j] + 1
            if newpop.shape[1] >= N:
                break
    F = NonDominatedSorting(newpop)
    return F, newpop, selected.astype(np.int32)
