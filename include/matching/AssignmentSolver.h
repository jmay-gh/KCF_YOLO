#ifndef KCF_YOLO_ASSIGNMENTSOLVER_H
#define KCF_YOLO_ASSIGNMENTSOLVER_H

#pragma once

#include "hungarian_algo/matrix.h"
#include <vector>
#include <utility>
#include <set>


class AssignmentSolver {
public:
    static std::vector<std::pair<int, int>> solveHungarian(Matrix<float>& costMatrix);
    static std::vector<std::pair<int, int>> solveGreedy(const Matrix<float>& costMatrix);
};

#endif //KCF_YOLO_ASSIGNMENTSOLVER_H
