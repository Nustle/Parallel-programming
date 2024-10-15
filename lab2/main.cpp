#include <vector>
#include <cmath>
#include <mpi.h>
#include <ctime>

using namespace std;

void initMatrixA(vector<vector<double> >& a) {
    int n = (int)a.size();
    for (int i = 0; i < n; i++) {
        a[i].resize(n);
        for (int j = 0; j < n; j++) {
            if (i == j) {
                a[i][j] = 2.0;
            } else {
                a[i][j] = 1.0;
            }
        }
    }
}

void initColumnGivenSolution(vector<double>& b, vector<double>& ans) {
    int n = (int)b.size();
    for (int i = 0; i < n; i++) {
        b[i] = n + 1;
        ans[i] = 1.0;
    }
}

void initColumnVariableSolution(vector<vector<double> >& a, vector<double>& b, vector<double>& ans) {
    int n = (int)b.size();
    for (int i = 0; i < n; i++) {
        double arg = (2 * M_PI * i) / n;
        if ((2 * i) % n == 0) {
            arg = 0;
        }
        ans[i] = sin(arg);
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            b[i] += a[i][j] * ans[j];
        }
    }
}


bool compareGivenSolution(const vector<double>& x, const vector<double>& ans, double e) {
    int n = (int)x.size();
    for (int i = 0; i < n; i++) {
        if (abs(x[i] - ans[i]) >= e) {
            return false;
        }
    }
    return true;
}


double scalarMultiply(const vector<double>& u, const vector<double>& v) {
    int n = (int)u.size();
    double ans = 0.0;
    for (int i = 0; i < n; i++) {
        ans += u[i] * v[i];
    }
    return ans;
}

double norma(const vector<double>& u) {
    double ans = 0.0;
    for (double val : u) {
        ans += val * val;
    }
    return sqrt(ans);
}

void multiplyMatrixByVector(const vector<vector<double>>& a, const vector<double>& x, vector<double>& result, int n, int rank, int size) {
    int rows_proc = n / size;
    int remainder = n % size;

    vector<int> local_rows_proc(size);
    for (int i = 0; i < size; i++) {
        local_rows_proc[i] = rows_proc + (i < remainder ? 1 : 0);
    }

    // кол-во строк разрезанной матрицы A для процесса с номером rank.
    int local_rows = local_rows_proc[rank];

    // кол-во элементов в каждой части разрезанной матрицы A (т.к. A - матрица, то кол-во элементов как бы из развёрнутой матрицы A).
    // Столько элементов отправляется в каждый процесс.
    vector<int> sendcounts(size);
    // то, с какого элемента начинается текущая строка развёрнутой матрицы A (для 0 процесса - с 0, для 1-ого - с rows_proc * n).
    // Каждое расположение в массиве относительно соответствующего элемента массива a_flat.
    vector<int> displs(size);

    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = local_rows_proc[i] * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // разворачиваем матрицу A в одномерный массив, т.к. его удобнее передавать через MPI.
    vector<double> a_flat;
    if (rank == 0) {
        a_flat.resize(n * n);
        for (int i = 0; i < n; i++) {
            copy(a[i].begin(), a[i].end(), a_flat.begin() + i * n);
        }
    }

    // часть развёрнутой матрицы A, которая соответствует процессу с номером rank.
    // такой буфер получает каждый процесс.
    vector<double> local_a_flat(local_rows * n);
    MPI_Scatterv(a_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_a_flat.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // вектор x отправляем всем процессам без разрезания.
    MPI_Bcast(const_cast<double*>(x.data()), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // умножаем часть матрицы A, соответствующей процессу с номером rank, на вектор x.
    vector<double> local_result(local_rows, 0.0);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_result[i] += local_a_flat[i * n + j] * x[j];
        }
    }

    vector<int> recvcounts(size);
    vector<int> recvdispls(size);
    offset = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = local_rows_proc[i];
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }

    vector<double> result_flat(n);

    // собираем от всех процессов все части матрицы A, умноженной на x, и передаём их процессу с номером 0.
    MPI_Gatherv(local_result.data(), local_rows, MPI_DOUBLE,
                result_flat.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Bcast(result_flat.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    result = result_flat;
}

vector<double> solveWithConjugateGradients(const vector<vector<double>>& a, const vector<double>& b, double e, int size, int rank) {
    int n = (int)b.size();
    vector<double> x_n(n, 0.0);
    vector<double> r_n(n);
    vector<double> z_n(n);
    vector<double> temp(n);

    multiplyMatrixByVector(a, x_n, temp, n, rank, size);
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            r_n[i] = b[i] - temp[i];
            z_n[i] = r_n[i];
        }
    }

    MPI_Bcast(r_n.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(z_n.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double b_norm = 0.0;
    if (rank == 0) {
        b_norm = norma(b);
    }
    MPI_Bcast(&b_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double condition = 0.0;
    if (rank == 0) {
        condition = norma(r_n) / b_norm;
    }
    MPI_Bcast(&condition, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int max_iter = 1000;
    int iter = 0;

    while (condition >= e && iter < max_iter) {
        multiplyMatrixByVector(a, z_n, temp, n, rank, size);
        if (rank == 0) {
            double rs = scalarMultiply(r_n, r_n);
            double a_n = rs / scalarMultiply(temp, z_n);
            for (int i = 0; i < n; i++) {
                x_n[i] += a_n * z_n[i];
                r_n[i] -= a_n * temp[i];
            }
            double b_n = scalarMultiply(r_n, r_n) / rs;
            for (int i = 0; i < n; i++) {
                z_n[i] = r_n[i] + b_n * z_n[i];
            }
            condition = norma(r_n) / b_norm;
        }

        // condition, x_n, r_n, z_n в размере count будут скопированы в локальные буферы x_n, r_n, z_n каждого процесса.
        MPI_Bcast(&condition, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(x_n.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(r_n.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(z_n.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        iter++;
    }

    return x_n;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;

    const char* mode = "given";

    if (argc > 1) {
        mode = argv[1];
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = (int)pow(2, 15);

    vector<vector<double> > a;
    vector<double> b(n);
    vector<double> ans(n);

    if (rank == 0) {
        a.resize(n);
        initMatrixA(a);
        if (strcmp(mode, "given") == 0) {
            initColumnGivenSolution(b, ans);
        } else {
            initColumnVariableSolution(a, b, ans);
        }
    }

    double e = 1e-6;

    time_t start_time, end_time;
    if (rank == 0) {
        time(&start_time);
    }

    vector<double> x = solveWithConjugateGradients(a, b, e, size, rank);

    if (rank == 0) {
        time(&end_time);
        printf("The result of work is %d with time: %f seconds\n", compareGivenSolution(x, ans, e), difftime(end_time, start_time));
    }

    MPI_Finalize();
    return 0;
}
