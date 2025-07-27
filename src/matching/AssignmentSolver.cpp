#include "../include/matching/AssignmentSolver.h"
#include "../include/hungarian_algo/Munkres.h"

std::vector<std::pair<int, int>> AssignmentSolver::solveHungarian(Matrix<float>& costMatrix) {

    Munkres<float> munkres;
    munkres.solve(costMatrix);

    std::vector<std::pair<int, int>> matches;
    for (int i = 0; i < costMatrix.rows(); ++i) {
        for (int j = 0; j < costMatrix.columns(); ++j) {
            if (costMatrix(i, j) == 0.0f) {
                matches.emplace_back(i, j);
                break;
            }
        }
    }
    return matches;
}


std::vector<std::pair<int, int>> AssignmentSolver::solveGreedy(const Matrix<float>& costMatrix) {
    std::vector<std::pair<int, int>> matches;
    std::set<int> usedCols;

    for (int i = 0; i < costMatrix.rows(); ++i) {
        float best = std::numeric_limits<float>::max();
        int bestCol = -1;
        for (int j = 0; j < costMatrix.columns(); ++j) {
            if (usedCols.count(j)) continue;
            if (costMatrix(i, j) < best) {
                best = costMatrix(i, j);
                bestCol = j;
            }
        }
        if (bestCol != -1) {
            usedCols.insert(bestCol);
            matches.emplace_back(i, bestCol);
        }
    }

    return matches;
}
