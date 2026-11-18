#include <mpi.h>
#include <omp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *allocate_array(size_t count, size_t item_size) {
    void *array = calloc(count, item_size);

    if (array == NULL) {
        fprintf(stderr, "Cannot allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    return array;
}

void read_graph(const char *filename, int *out_n, int *out_m,
                int **out_offsets, int **out_edges) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        perror(filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    int n;
    int m;

    if (fscanf(file, "%d%d", &n, &m) != 2 ||
        n < 1 || m < 0 || m > INT_MAX / 2) {
        fprintf(stderr, "Invalid graph size\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    int *from = allocate_array((size_t)m + 1, sizeof(int));
    int *to = allocate_array((size_t)m + 1, sizeof(int));
    int *next = allocate_array(n, sizeof(int));
    int *offsets = allocate_array((size_t)n + 1, sizeof(int));
    int *edges = allocate_array(2 * (size_t)m + 1, sizeof(int));

    // считаем соседей. Каждое ребро учитываем в обе стороны
    for (int i = 0; i < m; i++) {
        if (fscanf(file, "%d%d", &from[i], &to[i]) != 2 ||
            from[i] < 0 || from[i] >= n || to[i] < 0 || to[i] >= n) {
            fprintf(stderr, "Invalid edge %d\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }

        offsets[from[i] + 1]++;
        offsets[to[i] + 1]++;
    }

    fclose(file);

    // накопленная сумма задаёт границы списков соседей
    for (int v = 0; v < n; v++) {
        offsets[v + 1] += offsets[v];
        next[v] = offsets[v];
    }

    // next[v] указывает на свободное место в списке соседей v
    for (int i = 0; i < m; i++) {
        edges[next[from[i]]++] = to[i];
        edges[next[to[i]]++] = from[i];
    }

    free(from);
    free(to);
    free(next);

    *out_n = n;
    *out_m = m;
    *out_offsets = offsets;
    *out_edges = edges;
}

void bfs_sequential(const int *offsets, const int *edges, int start,
                    int *dist, int *queue) {
    int head = 0;
    int tail = 0;

    dist[start] = 0;
    queue[tail++] = start;

    while (head < tail) {
        int v = queue[head++];

        for (int j = offsets[v]; j < offsets[v + 1]; j++) {
            int u = edges[j];

            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank;
    int nprocs;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: bfs graph.txt start_vertex\n");
        }

        MPI_Finalize();
        return 1;
    }

    int n;
    int m;
    int *offsets;
    int *edges;

    read_graph(argv[1], &n, &m, &offsets, &edges);

    char *end;
    long input_start = strtol(argv[2], &end, 10);

    if (end == argv[2] || *end != '\0' || input_start < 0 || input_start >= n) {
        if (rank == 0) {
            fprintf(stderr, "Invalid start vertex\n");
        }

        free(offsets);
        free(edges);
        MPI_Finalize();
        return 1;
    }

    int start = (int)input_start;
    int *dist_seq = NULL;
    int *dist = allocate_array(n, sizeof(int));
    int *frontier = allocate_array(n, sizeof(int));
    int *next_frontier = allocate_array(n, sizeof(int));
    int *local_candidate = allocate_array(n, sizeof(int));
    int *global_candidate = allocate_array(n, sizeof(int));

    omp_set_dynamic(0);
    int threads = omp_get_max_threads();

    // каждый поток получает свой массив признаков
    unsigned char *thread_candidates =
        allocate_array(threads, (size_t)n);

    int *frontier_sizes = NULL;
    int *processed_by_rank = NULL;
    int local_processed = 0;
    int max_frontier = 0;
    double time_seq = 0.0;

    if (rank == 0) {
        frontier_sizes = allocate_array(n, sizeof(int));
        processed_by_rank = allocate_array(nprocs, sizeof(int));
        dist_seq = allocate_array(n, sizeof(int));
        int *queue = allocate_array(n, sizeof(int));

        for (int v = 0; v < n; v++) {
            dist_seq[v] = -1;
        }

        double begin = MPI_Wtime();
        bfs_sequential(offsets, edges, start, dist_seq, queue);
        time_seq = MPI_Wtime() - begin;

        free(queue);
    }

    for (int v = 0; v < n; v++) {
        dist[v] = -1;
    }

    dist[start] = 0;
    frontier[0] = start;

    int frontier_size = 1;
    int level = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    while (frontier_size > 0) {
        int local_size = frontier_size / nprocs;

        if (rank < frontier_size % nprocs) {
            local_size++;
        }

        local_processed += local_size;

        // фронт одинаков у всех процессов, сохраняем размеры на ранге 0
        if (rank == 0) {
            frontier_sizes[level] = frontier_size;

            if (frontier_size > max_frontier) {
                max_frontier = frontier_size;
            }
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            unsigned char *candidates = thread_candidates + (size_t)tid * n;

            memset(candidates, 0, (size_t)n);

            #pragma omp for schedule(runtime)
            for (int k = 0; k < local_size; k++) {
                int index = rank + k * nprocs;
                int v = frontier[index];

                for (int j = offsets[v]; j < offsets[v + 1]; j++) {
                    int u = edges[j];

                    if (dist[u] == -1) {
                        candidates[u] = 1;
                    }
                }
            }
        }

        // объединяем признаки потоков одного процесса
        memset(local_candidate, 0, (size_t)n * sizeof(int));

        for (int t = 0; t < threads; t++) {
            unsigned char *candidates = thread_candidates + (size_t)t * n;

            for (int v = 0; v < n; v++) {
                if (candidates[v] == 1) {
                    local_candidate[v] = 1;
                }
            }
        }

        MPI_Allreduce(local_candidate, global_candidate, n,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // каждый процесс строит одинаковый следующий фронт
        int next_size = 0;

        for (int v = 0; v < n; v++) {
            if (global_candidate[v] == 1 && dist[v] == -1) {
                dist[v] = level + 1;
                next_frontier[next_size++] = v;
            }
        }

        int *temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;
        frontier_size = next_size;
        level++;
    }

    double local_time = MPI_Wtime() - begin;
    double time_par = 0.0;

    MPI_Reduce(&local_time, &time_par, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gather(&local_processed, 1, MPI_INT, processed_by_rank, 1,
               MPI_INT, 0, MPI_COMM_WORLD);

    int mismatches = 0;

    if (rank == 0) {
        int unreachable = 0;

        for (int v = 0; v < n; v++) {
            if (dist_seq[v] != dist[v]) {
                mismatches++;
            }

            if (dist[v] == -1) {
                unreachable++;
            }

            if (n <= 20) {
                printf("dist[%d]: seq = %d, par = %d\n", v, dist_seq[v], dist[v]);
            }
        }

        printf("N = %d  M = %d  start = %d  levels = %d\n", n, m, start, level);
        printf("MPI processes = %d  OpenMP threads per process = %d\n", nprocs, threads);
        printf("Unreachable = %d  Mismatches = %d\n", unreachable, mismatches);
        printf("Sequential = %.9f s  Hybrid = %.9f s\n", time_seq, time_par);

        if (time_par > 0.0) {
            double speedup = time_seq / time_par;
            double efficiency = speedup / ((double)nprocs * threads);
            printf("Speedup = %.6f  Efficiency = %.6f\n", speedup, efficiency);
        }

        printf("Frontier sizes by level:");

        for (int k = 0; k < level; k++) {
            printf(" %d", frontier_sizes[k]);
        }

        printf("\nMax frontier = %d\n", max_frontier);

        for (int p = 0; p < nprocs; p++) {
            printf("Rank %d processed %d vertices\n", p, processed_by_rank[p]);
        }
    }

    MPI_Bcast(&mismatches, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(offsets);
    free(edges);
    free(dist_seq);
    free(dist);
    free(frontier);
    free(next_frontier);
    free(local_candidate);
    free(global_candidate);
    free(thread_candidates);
    free(frontier_sizes);
    free(processed_by_rank);

    MPI_Finalize();
    return mismatches != 0;
}