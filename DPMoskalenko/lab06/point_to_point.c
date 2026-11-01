#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define TAG_MESSAGE 1 // тег сообщения от процесса 0
#define TAG_REPLY 2 // тег ответа процессу 0
#define MAX_LEN 100 // размер буфера

#define WARMUP 10 // пробные обмены
#define REPS 1000 // обмены в замере

char buf[1048576]; // буфер для замера, вмещает самое большое

// обмен сообщениями туда и обратно
void round_trip(int rank, int n) {
    if (rank == 0) {
        MPI_Send(buf, n, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
        MPI_Recv(buf, n, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Recv(buf, n, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf, n, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

// эксперимент 3.3: среднее время обмена для сообщений разного размера
void measure(int rank) {
    int sizes[] = {1, 1024, 65536, 1048576};

    if (rank == 0) {
        printf("size, bytes   reps   avg round-trip time, s\n");
        fflush(stdout);
    }

    for (int s = 0; s < 4; s++) {
        int n = sizes[s];

        for (int k = 0; k < WARMUP; k++) {
            round_trip(rank, n);
        }

        double start = MPI_Wtime();

        for (int k = 0; k < REPS; k++) {
            round_trip(rank, n);
        }

        double avg = (MPI_Wtime() - start) / REPS;

        if (rank == 0) {
            printf("%11d   %4d   %.3e\n", n, REPS, avg);
            fflush(stdout);
        }
    }
}

// fflush(stdout) после printf нужен
// чтобы строки разных процессов не перемешивались на экране
int main(int argc, char **argv) {
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("Need at least 2 processes: mpiexec -n 2 ./point_to_point\n");
        fflush(stdout);
        MPI_Finalize();
        return 1;
    }

    // режим замера времени
    if (argc > 1 && strcmp(argv[1], "time") == 0) {
        measure(rank);
        MPI_Finalize();
        return 0;
    }

    char message[MAX_LEN];
    char reply[MAX_LEN];

    if (rank == 0) {
        // рассылка: каждому процессу r своё сообщение с его номером
        for (int r = 1; r < size; r++) {
            sprintf(message, "Hello, rank %d, from rank 0", r);

            printf("[rank 0] send to rank %d: \"%s\"\n", r, message);
            fflush(stdout);
            MPI_Send(message, strlen(message) + 1, MPI_CHAR, r, TAG_MESSAGE, MPI_COMM_WORLD);
        }

        int received = 0;

        for (int r = 1; r < size; r++) {
            MPI_Recv(reply, MAX_LEN, MPI_CHAR, r, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[rank 0] received from rank %d: \"%s\"\n", r, reply);
            fflush(stdout);
            received++;
        }

        printf("[rank 0] sent %d messages, received %d replies\n", size - 1, received);
        fflush(stdout);
    } else {
        MPI_Recv(message, MAX_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank %d] received from rank 0: \"%s\"\n", rank, message);
        fflush(stdout);

        sprintf(reply, "Message received by rank %d", rank);

        printf("[rank %d] send reply to rank 0: \"%s\"\n", rank, reply);
        fflush(stdout);
        MPI_Send(reply, strlen(reply) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}