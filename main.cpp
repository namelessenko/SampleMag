#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

MatrixXd sumNNi_vectorized(MatrixXd& Sx, MatrixXd& Sy, MatrixXd& Sz) {
    MatrixXd Sx_ip1 = Sx.rightCols(Sx.cols() - 1);
    Sx_ip1.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_ip1 = Sy.rightCols(Sy.cols() - 1);
    Sy_ip1.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_ip1 = Sz.rightCols(Sz.cols() - 1);
    Sz_ip1.conservativeResize(Sz.rows(), Sz.cols());
    MatrixXd Sx_im1 = Sx.leftCols(Sx.cols() - 1);
    Sx_im1.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_im1 = Sy.leftCols(Sy.cols() - 1);
    Sy_im1.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_im1 = Sz.leftCols(Sz.cols() - 1);
    Sz_im1.conservativeResize(Sz.rows(), Sz.cols());
    return Sx.array() * Sx_ip1.array() + Sy.array() * Sy_ip1.array() +
           Sz.array() * Sz_ip1.array() + Sx.array() * Sx_im1.array() +
           Sy.array() * Sy_im1.array() + Sz.array() * Sz_im1.array();
}

MatrixXd sumNNj_vectorized(MatrixXd& Sx, MatrixXd& Sy, MatrixXd& Sz) {
    MatrixXd Sx_jp1 = Sx.bottomRows(Sx.rows() - 1);
    Sx_jp1.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_jp1 = Sy.bottomRows(Sy.rows() - 1);
    Sy_jp1.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_jp1 = Sz.bottomRows(Sz.rows() - 1);
    Sz_jp1.conservativeResize(Sz.rows(), Sz.cols());
    MatrixXd Sx_jm1 = Sx.topRows(Sx.rows() - 1);
    Sx_jm1.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_jm1 = Sy.topRows(Sy.rows() - 1);
    Sy_jm1.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_jm1 = Sz.topRows(Sz.rows() - 1);
    Sz_jm1.conservativeResize(Sz.rows(), Sz.cols());
    return Sx.array() * Sx_jp1.array() + Sy.array() * Sy_jp1.array() +
           Sz.array() * Sz_jp1.array() + Sx.array() * Sx_jm1.array() +
           Sy.array() * Sy_jm1.array() + Sz.array() * Sz_jm1.array();
}

MatrixXd sumNNij_vectorized(MatrixXd& Sx, MatrixXd& Sy, MatrixXd& Sz) {
    MatrixXd Sx_ip1 = Sx.rightCols(Sx.cols() - 1);
    Sx_ip1.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_ip1 = Sy.rightCols(Sy.cols() - 1);
    Sy_ip1.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_ip1 = Sz.rightCols(Sz.cols() - 1);
    Sz_ip1.conservativeResize(Sz.rows(), Sz.cols());
    MatrixXd Sx_im1 = Sx.leftCols(Sx.cols() - 1);
    Sx_im1.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_im1 = Sy.leftCols(Sy.cols() - 1);
    Sy_im1.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_im1 = Sz.leftCols(Sz.cols() - 1);
    Sz_im1.conservativeResize(Sz.rows(), Sz.cols());
    MatrixXd Sx_ip1_jp1 = Sx_ip1.bottomRows(Sx_ip1.rows() - 1);
    Sx_ip1_jp1.conservativeResize(Sx_ip1.rows(), Sx_ip1.cols());
    MatrixXd Sy_ip1_jp1 = Sy_ip1.bottomRows(Sy_ip1.rows() - 1);
    Sy_ip1_jp1.conservativeResize(Sy_ip1.rows(), Sy_ip1.cols());
    MatrixXd Sz_ip1_jp1 = Sz_ip1.bottomRows(Sz_ip1.rows() - 1);
    Sz_ip1_jp1.conservativeResize(Sz_ip1.rows(), Sz_ip1.cols());
    MatrixXd Sx_im1_jm1 = Sx_im1.topRows(Sx_im1.rows() - 1);
    Sx_im1_jm1.conservativeResize(Sx_im1.rows(), Sx_im1.cols());
    MatrixXd Sy_im1_jm1 = Sy_im1.topRows(Sy_im1.rows() - 1);
    Sy_im1_jm1.conservativeResize(Sy_im1.rows(), Sy_im1.cols());
    MatrixXd Sz_im1_jm1 = Sz_im1.topRows(Sz_im1.rows() - 1);
    Sz_im1_jm1.conservativeResize(Sz_im1.rows(), Sz_im1.cols());
    MatrixXd Sx_ip1_jm1 = Sx_ip1.topRows(Sx_ip1.rows() - 1);
    Sx_ip1_jm1.conservativeResize(Sx_ip1.rows(), Sx_ip1.cols());
    MatrixXd Sy_ip1_jm1 = Sy_ip1.topRows(Sy_ip1.rows() - 1);
    Sy_ip1_jm1.conservativeResize(Sy_ip1.rows(), Sy_ip1.cols());
    MatrixXd Sz_ip1_jm1 = Sz_ip1.topRows(Sz_ip1.rows() - 1);
    Sz_ip1_jm1.conservativeResize(Sz_ip1.rows(), Sz_ip1.cols());
    MatrixXd Sx_im1_jp1 = Sx_im1.bottomRows(Sx_im1.rows() - 1);
    Sx_im1_jp1.conservativeResize(Sx_im1.rows(), Sx_im1.cols());
    MatrixXd Sy_im1_jp1 = Sy_im1.bottomRows(Sy_im1.rows() - 1);
    Sy_im1_jp1.conservativeResize(Sy_im1.rows(), Sy_im1.cols());
    MatrixXd Sz_im1_jp1 = Sz_im1.bottomRows(Sz_im1.rows() - 1);
    Sz_im1_jp1.conservativeResize(Sz_im1.rows(), Sz_im1.cols());
    return Sx.array() * Sx_ip1_jp1.array() + Sy.array() * Sy_ip1_jp1.array() +
           Sz.array() * Sz_ip1_jp1.array() + Sx.array() * Sx_im1_jm1.array() +
           Sy.array() * Sy_im1_jm1.array() + Sz.array() * Sz_im1_jm1.array() +
           Sx.array() * Sx_ip1_jm1.array() + Sy.array() * Sy_ip1_jm1.array() +
           Sz.array() * Sz_ip1_jm1.array() + Sx.array() * Sx_im1_jp1.array() +
           Sy.array() * Sy_im1_jp1.array() + Sz.array() * Sz_im1_jp1.array();
}

MatrixXd sum2Ni_vectorized(MatrixXd& Sx, MatrixXd& Sy, MatrixXd& Sz) {
    MatrixXd Sx_ip2 = Sx.rightCols(Sx.cols() - 2);
    Sx_ip2.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_ip2 = Sy.rightCols(Sy.cols() - 2);
    Sy_ip2.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_ip2 = Sz.rightCols(Sz.cols() - 2);
    Sz_ip2.conservativeResize(Sz.rows(), Sz.cols());
    MatrixXd Sx_im2 = Sx.leftCols(Sx.cols() - 2);
    Sx_im2.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_im2 = Sy.leftCols(Sy.cols() - 2);
    Sy_im2.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_im2 = Sz.leftCols(Sz.cols() - 2);
    Sz_im2.conservativeResize(Sz.rows(), Sz.cols());
    return Sx.array() * Sx_ip2.array() + Sy.array() * Sy_ip2.array() +
           Sz.array() * Sz_ip2.array() + Sx.array() * Sx_im2.array() +
           Sy.array() * Sy_im2.array() + Sz.array() * Sz_im2.array();
}

MatrixXd sum2Nj_vectorized(MatrixXd& Sx, MatrixXd& Sy, MatrixXd& Sz) {
    MatrixXd Sx_jp2 = Sx.bottomRows(Sx.rows() - 2);
    Sx_jp2.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_jp2 = Sy.bottomRows(Sy.rows() - 2);
    Sy_jp2.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_jp2 = Sz.bottomRows(Sz.rows() - 2);
    Sz_jp2.conservativeResize(Sz.rows(), Sz.cols());
    MatrixXd Sx_jm2 = Sx.topRows(Sx.rows() - 2);
    Sx_jm2.conservativeResize(Sx.rows(), Sx.cols());
    MatrixXd Sy_jm2 = Sy.topRows(Sy.rows() - 2);
    Sy_jm2.conservativeResize(Sy.rows(), Sy.cols());
    MatrixXd Sz_jm2 = Sz.topRows(Sz.rows() - 2);
    Sz_jm2.conservativeResize(Sz.rows(), Sz.cols());
    return Sx.array() * Sx_jp2.array() + Sy.array() * Sy_jp2.array() +
           Sz.array() * Sz_jp2.array() + Sx.array() * Sx_jm2.array() +
           Sy.array() * Sy_jm2.array() + Sz.array() * Sz_jm2.array();
}

int main() {
    // Define your N value
    int N = 5;

    // Create Ri and Rj matrices
    MatrixXd Ri = MatrixXd::Zero(N, N);
    MatrixXd Rj = MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Ri(i, j) = i + j;
            Rj(i, j) = i - j;
        }
    }

    // Calculate Sx, Sy, Sz matrices
    MatrixXd Sx = Ri.array() + Rj.array();
    MatrixXd Sy = Ri.array() + Rj.array();
    MatrixXd Sz = Ri.array() + Rj.array();

    // Calculate the sums
    MatrixXd sum_NNi = sumNNi_vectorized(Sx, Sy, Sz);
    MatrixXd sum_NNj = sumNNj_vectorized(Sx, Sy, Sz);
    MatrixXd sum_NNij = sumNNij_vectorized(Sx, Sy, Sz);
    MatrixXd sum_2Ni = sum2Ni_vectorized(Sx, Sy, Sz);
    MatrixXd sum_2Nj = sum2Nj_vectorized(Sx, Sy, Sz);

    // Calculate the final summation
    double J1 = 1.0;  // Set your J1 value
    double J2 = 1.0;  // Set your J2 value
    double J4 = 1.0;  // Set your J4 value
    double summation = -J1/2 * (sum_NNi.sum() + sum_NNj.sum())
                       + J2/2 * sum_NNij.sum()
                       + J4/2 * (sum_2Ni.sum() + sum_2Nj.sum());

    // Print the results
    std::cout << "Summation: " << summation << std::endl;

    return 0;
}

