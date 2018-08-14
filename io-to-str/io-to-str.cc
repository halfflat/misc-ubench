extern "C" {
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
}

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "benchmark/benchmark.h"

char temp_file[] = "/tmp/iotest_XXXXXX";

struct throw_syserr { const char* what; };

template <typename R>
R operator||(R r, throw_syserr err) {
    return r==R(-1)? throw std::system_error(errno, std::system_category(), err.what): r;
}

void make_temp(std::size_t bytes) {
    std::vector<char> contents(bytes, 'x');

    int fd;
    fd = ::mkstemp(temp_file) || throw_syserr{"mkstemp"};

    std::FILE* file;
    file = ::fdopen(fd, "w") || throw_syserr{"fdopen"};

    std::fwrite(&contents[0], 1, contents.size(), file);
    std::fclose(file);
}

void rm_temp() {
    std::remove(temp_file) || throw_syserr{"remove"};
    std::size_t n = std::strlen(temp_file);
    if (n>=6) {
        std::memset(temp_file+n-6, 'X', 6);
    }
}

std::string run_mmap() {
    int fd;
    fd = open(temp_file, O_RDONLY) || throw_syserr{"open"};

    struct stat st;
    fstat(fd, &st) || throw_syserr{"fstat"};

    void* addr;
    addr = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) || throw_syserr{"mmap"};

    std::string s((char*)addr , (char*)addr+st.st_size);

    munmap(addr, st.st_size);
    close(fd);

    return s;
}

std::string run_fstream_read() {
    std::string s;

    std::ifstream fs;
    fs.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    fs.open(temp_file);

    fs.seekg(0, std::ios::end);
    s.resize(fs.tellg());
    fs.seekg(0, std::ios::beg);
    fs.read(&s[0], s.size());
    return s;
}

std::string run_fstream_rdbuf() {
    std::stringstream s;

    std::ifstream fs;
    fs.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    fs.open(temp_file);

    s << fs.rdbuf();
    return s.str();
}

std::string run_fstream_iter() {
    std::ifstream fs;
    fs.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    fs.open(temp_file);

    using SI = std::istreambuf_iterator<char>;
    return std::string(SI(fs), SI());
}

void string_reader(benchmark::State& state, std::string (*fn)()) {
    std::size_t sz = state.range(0);
    make_temp(sz);

    for (auto _: state) {
        benchmark::DoNotOptimize(fn());
    }

    rm_temp();
}

BENCHMARK_CAPTURE(string_reader, mmap, run_mmap)->Range(1<<10, 1<<28);
BENCHMARK_CAPTURE(string_reader, fstream_read, run_fstream_read)->Range(1<<10, 1<<28);
BENCHMARK_CAPTURE(string_reader, fstream_rdbuf, run_fstream_rdbuf)->Range(1<<10, 1<<28);
BENCHMARK_CAPTURE(string_reader, fstream_iter, run_fstream_rdbuf)->Range(1<<10, 1<<28);

BENCHMARK_MAIN();

