#include <iostream>
#include <thread>
#include <queue>
#include <unordered_map>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <random>
#include <cmath>
#include <iomanip>

// Функции для вычислений
template<typename T>
T fun_sin(T arg) {
    return std::sin(arg);
}

template <typename T>
T fun_sqrt(T arg) {
    return std::sqrt(arg);
}

template <typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}

template<typename T>
class TaskServer {
public:
    using Task = std::function<T()>;

    TaskServer() : running(false), task_id(0) {}

    void start() {
        running = true;
        worker = std::thread(&TaskServer::process_tasks, this);
    }

    void stop() {
        running = false;
        cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    size_t add_task(Task task, const std::string& operation, double arg1, double arg2,
                    const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_queue);
        size_t id = task_id++;
        std::promise<T> prom;
        results[id] = prom.get_future();
        tasks.push({id, std::move(task), std::move(prom), operation, arg1, arg2, filename});
        cv.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::future<T>& fut = results.at(id);
        return fut.get();  // Блокирующий вызов
    }

private:
    struct TaskItem {
        size_t id;
        Task task;
        std::promise<T> prom;
        std::string operation;
        double arg1;
        double arg2;
        std::string filename;
    };

    void process_tasks() {
        while (running) {
            std::unique_lock<std::mutex> lock(mutex_queue);
            cv.wait(lock, [&]() { return !tasks.empty() || !running; });

            if (!tasks.empty()) {
                TaskItem item = std::move(tasks.front());
                tasks.pop();
                lock.unlock();

                T result = item.task();
                item.prom.set_value(result);

                // Записываем информацию в соответствующий файл
                std::ofstream fout(item.filename, std::ios::app);
                fout << std::fixed << std::setprecision(6); // Одинаковая точность
                fout << item.operation << " " << item.arg1;
                if (item.operation == "pow") {
                    fout << " " << item.arg2;
                }
                fout << " = " << result << "\n";
            }
        }
    }

    std::atomic<bool> running;
    std::atomic<size_t> task_id;
    std::thread worker;

    std::queue<TaskItem> tasks;
    std::unordered_map<size_t, std::future<T>> results;
    std::mutex mutex_queue;
    std::condition_variable cv;
};

// Аргумент filename добавлен
void client(TaskServer<double>& server, int task_type, int N, const std::string& filename) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 10.0);

    for (int i = 0; i < N; ++i) {
        if (task_type == 1) {  // sin
            double val = dis(gen);
            server.add_task([val]() { return std::sin(val); }, "sin", val, 0.0, filename);
        } else if (task_type == 2) {  // sqrt
            double val = dis(gen);
            server.add_task([val]() { return std::sqrt(val); }, "sqrt", val, 0.0, filename);
        } else if (task_type == 3) {  // pow
            double base = dis(gen), exp = dis(gen);
            server.add_task([base, exp]() { return std::pow(base, exp); }, "pow", base, exp, filename);
        }
    }
}

int main() {
    TaskServer<double> server;
    server.start();

    const int N = 10000;  // Число задач для каждого клиента

    std::thread client1(client, std::ref(server), 1, N, "sin_output.txt");
    std::thread client2(client, std::ref(server), 2, N, "sqrt_output.txt");
    std::thread client3(client, std::ref(server), 3, N, "pow_output.txt");

    client1.join();
    client2.join();
    client3.join();

    server.stop();

    return 0;
}
