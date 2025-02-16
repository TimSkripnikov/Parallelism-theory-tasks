# Parallelism Theory Tasks

This repository contains various tasks related to the **Parallelism Theory**. The `main.cpp` file is used for the core functionality of each task, and there are two branches for different build systems: `make` and `cmake`. This README file provides instructions for building the project using each system.

## Task 1

### Project Structure

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
2. Compile the project using the `make` command:
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
2. Create a build directory and configure the project with `CMake`:
    ```bash
    mkdir build
    cd build
    cmake ..
    ```
3. Compile the project:
    ```bash
    make
    ```
   This will compile the `main.cpp` file and generate the executable `sum_sin`.

## Notes

- If you want to switch between the `make` and `cmake` branches, ensure that all untracked changes are either committed or stashed to avoid conflicts.
- Each build system has its own specific setup, but both will produce the same executable `sum_sin` from the `main.cpp` file.

Feel free to modify and extend the `README.md` based on your future needs.
