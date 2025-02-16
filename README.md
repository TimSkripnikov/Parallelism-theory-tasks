# Parallelism Theory Tasks

This repository contains various tasks related to the **Parallelism Theory**. 

## Task 1

### Project Structure

There are two branches for different build systems: `make` and `cmake`. This README file provides instructions for building the project using each system.

- `main.cpp`: The main source code file for the project.
- `Makefile`: Build file for the `make` system located in the `make` branch.
- `CMakeLists.txt`: Build file for the `CMake` system located in the `cmake` branch.

### Branches

#### `main` Branch
The `main` branch contains the source code and documentation files. The files for building the project (using `make` or `cmake`) are located in their respective branches.

#### `make` Branch
The `make` branch contains the `Makefile` for building the project using the `make` build system.

##### To build with `make`:
1. Switch to the `make` branch:
    ```bash
    git checkout make
    ```

    ```bash
    cd task1
    ```

2. Compile the project using the `make` command:
    
    With double type

    ```bash
    make TYPE=double
    ```

    With float type

    ```bash
    make 
    ```

   This will compile the `main.cpp` file and generate the executable `sum_sin`.

#### `cmake` Branch
The `cmake` branch contains the `CMakeLists.txt` file for building the project using the `CMake` build system.

##### To build with `CMake`:
1. Switch to the `cmake` branch:
    ```bash
    git checkout cmake
    ```

    ```bash
    cd task1
    ```

2. Create a build directory and configure the project with `CMake`:

    with double type

    ```bash
    mkdir build
    cmake -S . -B build -D TYPE=double
    cmake --build ./build
    ```

    with float type

    ```bash
    mkdir build
    cmake -S . -B build
    cmake --build ./build
    ```

   This will compile the `main.cpp` file and generate the executable `sum_sin`.

