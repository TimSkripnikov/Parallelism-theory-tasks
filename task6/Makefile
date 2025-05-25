# Название исполняемых файлов
TARGET_CPU = heat_cpu
TARGET_MULTICORE = heat_multicore
TARGET_GPU = heat_gpu

CXX = pgc++
CXXFLAGS = -O3 -Minfo=all -std=c++14

ACC_MULTICORE = -acc=multicore
ACC_GPU = -acc=gpu

BOOST_FLAGS = -lboost_program_options

SRC = main.cpp

all: $(TARGET_CPU) $(TARGET_MULTICORE) $(TARGET_GPU)

$(TARGET_CPU): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BOOST_FLAGS)

$(TARGET_MULTICORE): $(SRC)
	$(CXX) $(CXXFLAGS) -acc $(ACC_MULTICORE) -o $@ $^ $(BOOST_FLAGS)

$(TARGET_GPU): $(SRC)
	$(CXX) $(CXXFLAGS) -acc $(ACC_GPU) -o $@ $^ $(BOOST_FLAGS)

clean:
	rm -f $(TARGET_CPU) $(TARGET_MULTICORE) $(TARGET_GPU) output.txt
