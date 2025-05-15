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
#include <algorithm>

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
        {
            std::lock_guard<std::mutex> lock(mutex_queue);
            running = false;
        }
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
        return fut.get();
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
        while (true) {
            TaskItem item;
            {
                std::unique_lock<std::mutex> lock(mutex_queue);
                cv.wait(lock, [&]() { return !tasks.empty() || !running; });

                if (!running && tasks.empty()) break;
                if (tasks.empty()) continue;

                item = std::move(tasks.front());
                tasks.pop();
            }

            try {
                T result = item.task();
                item.prom.set_value(result);

                std::lock_guard<std::mutex> file_lock(file_mutex);
                std::ofstream fout(item.filename, std::ios::app);
                fout << std::fixed << std::setprecision(6);
                fout << item.operation << " " << item.arg1;
                if (item.operation == "pow") {
                    fout << " " << item.arg2;
                }
                fout << " = " << result << "\n";
            } catch (...) {
                item.prom.set_exception(std::current_exception());
            }
        }
    }

    std::atomic<bool> running;
    std::atomic<size_t> task_id;
    std::thread worker;

    std::queue<TaskItem> tasks;
    std::unordered_map<size_t, std::future<T>> results;
    std::mutex mutex_queue;
    std::mutex file_mutex;
    std::condition_variable cv;
};

void client(TaskServer<double>& server, int task_type, int N, const std::string& filename) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 10.0);

    for (int i = 0; i < N; ++i) {
        if (task_type == 1) {
            double val = dis(gen);
            server.add_task([val]() { return fun_sin(val); }, "sin", val, 0.0, filename);
        } else if (task_type == 2) {
            double val = dis(gen);
            server.add_task([val]() { return fun_sqrt(val); }, "sqrt", val, 0.0, filename);
        } else if (task_type == 3) {
            double base = dis(gen), exp = dis(gen);
            server.add_task([base, exp]() { return fun_pow(base, exp); }, "pow", base, exp, filename);
        }
    }
}

int main() {
    TaskServer<double> server;
    server.start();

    const int N = 100;

    std::ofstream("sin_output.txt", std::ios::trunc).close();
    std::ofstream("sqrt_output.txt", std::ios::trunc).close();
    std::ofstream("pow_output.txt", std::ios::trunc).close();

    std::thread client1(client, std::ref(server), 1, N, "sin_output.txt");
    std::thread client2(client, std::ref(server), 2, N, "sqrt_output.txt");
    std::thread client3(client, std::ref(server), 3, N, "pow_output.txt");

    client1.join();
    client2.join();
    client3.join();

    server.stop();


    auto count_lines = [](const std::string& filename) {
        std::ifstream fin(filename);
        return std::count(std::istreambuf_iterator<char>(fin),
                        std::istreambuf_iterator<char>(), '\n');
    };

    std::cout << "sin_output.txt lines: " << count_lines("sin_output.txt") << std::endl;
    std::cout << "sqrt_output.txt lines: " << count_lines("sqrt_output.txt") << std::endl;
    std::cout << "pow_output.txt lines: " << count_lines("pow_output.txt") << std::endl;

    return 0;
}